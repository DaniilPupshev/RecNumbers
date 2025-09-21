import os
import csv
import re
from datetime import datetime, timedelta

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

VIDEO_SOURCE = 0
YOLO_WEIGHTS = "best.pt"

CSV_PATH   = "plates.csv"
CSV_FIELDS = ["plate_text", "first_seen", "detections"]
CSV_DELIM  = ","
CSV_ENC    = "utf-8-sig"

DET_CONF = 0.20
N_RECENT = 5
COOLDOWN_SECONDS = 15

STABLE_WINDOW = 1.0        # сек
REPEAT_VOTES  = 2
MIN_OCR_CONF  = 0.50       # OCR confidence
GRID          = 40

# DEBUG-Переключатели
DEBUG = True
RECTIFY_ENABLED = True     # если True — пробуем выпрямлять табличку; если нет — берём чистый crop
SHOW_RAW_OCR = True        # показывать сырые OCR-кандидаты в углу экрана
SHOW_REASONS = True        # показывать причины отбрасывания

# ========= ВАЛИДАЦИЯ РФ =========
RU_LETTERS = "ABEKMHOPCTYX"
RU_PLATE_RE = re.compile(rf"^[{RU_LETTERS}]\d{{3}}[{RU_LETTERS}]{{2}}\d{{2,3}}$")

def clean_text(s: str) -> str:
    s = s.upper()
    s = (s.replace("А","A").replace("В","B").replace("Е","E").replace("К","K")
           .replace("М","M").replace("Н","H").replace("О","O").replace("Р","P")
           .replace("С","C").replace("Т","T").replace("У","Y").replace("Х","X"))
    return re.sub(r"[^A-Z0-9]", "", s)

def is_valid_ru_plate(s: str) -> bool:
    return bool(RU_PLATE_RE.match(s))

# ========= CSV =========
def ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding=CSV_ENC) as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=CSV_DELIM)
            w.writeheader()

def load_rows():
    ensure_csv()
    with open(CSV_PATH, "r", newline="", encoding=CSV_ENC) as f:
        rdr = csv.DictReader(f, delimiter=CSV_DELIM)
        return list(rdr)

def save_rows(rows):
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding=CSV_ENC) as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=CSV_DELIM)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, CSV_PATH)

def parse_detections(cell: str) -> list[str]:
    cell = (cell or "").strip()
    if not cell: return []
    if cell.startswith("[") and cell.endswith("]"):
        try:
            import json
            return [t.strip().strip('"') for t in json.loads(cell)]
        except Exception:
            pass
    return [t for t in cell.split("|") if t.strip()]

def dump_detections(ts_list: list[str]) -> str:
    return "|".join(ts_list)

def upsert_plate(plate: str, now_iso: str):
    rows = load_rows()
    idx = None
    for i, row in enumerate(rows):
        if row.get("plate_text") == plate:
            idx = i
            break

    if idx is None:
        rows.append({
            "plate_text": plate,
            "first_seen": now_iso,
            "detections": dump_detections([now_iso]),
        })
        save_rows(rows)
        if DEBUG: print(f"[NEW]  {plate} @ {now_iso}")
        return

    row = rows[idx]
    times = parse_detections(row.get("detections", ""))
    last_dt = None
    if times:
        try: last_dt = datetime.fromisoformat(times[-1])
        except Exception: last_dt = None

    now_dt = datetime.fromisoformat(now_iso)
    if (last_dt is None) or (now_dt - last_dt >= timedelta(seconds=COOLDOWN_SECONDS)):
        times.append(now_iso)
        if len(times) > N_RECENT: times = times[-N_RECENT:]
        row["detections"] = dump_detections(times)
        rows[idx] = row
        save_rows(rows)
        if DEBUG: print(f"[SEEN] {plate} + {now_iso} (last {len(times)}/{N_RECENT})")

# ========= ПРЕДОБРАБОТКА =========
def rectify_plate(crop_bgr, expand=0.08):
    if crop_bgr is None or crop_bgr.size == 0: return crop_bgr
    h, w = crop_bgr.shape[:2]
    pad = int(max(h, w) * expand)
    roi = cv2.copyMakeBorder(crop_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 40, 40)
    clahe = cv2.createCLAHE(3.0, (8,8)).apply(gray)
    edges = cv2.Canny(clahe, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return roi

    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)

    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)
    tl = box[np.argmin(s)]; br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]; bl = box[np.argmax(diff)]
    ordered = np.array([tl,tr,br,bl], dtype=np.float32)

    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    maxW = int(max(wA, wB)); maxH = int(max(hA, hB))
    if maxW < 60 or maxH < 20: return roi

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(roi, M, (maxW, maxH))
    return warped

def preprocess_roi_for_ocr(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return []
    out = []
    # 1) исходный
    out.append(roi_bgr.copy())
    # 2) LAB-CLAHE + median
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8,8)).apply(l)
    rgb = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    g1 = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    g1 = cv2.medianBlur(g1, 3)
    out.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))
    # 3) Otsu binary
    _, bin_ = cv2.threshold(g1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out.append(cv2.cvtColor(bin_, cv2.COLOR_GRAY2BGR))
    return out

# ========= СТАБИЛИЗАЦИЯ =========
stabilizer = {}

def bbox_key(x1, y1, x2, y2):
    cx = (x1+x2)//2; cy = (y1+y2)//2
    return f"{(cx//GRID)*GRID}_{(cy//GRID)*GRID}"

def purge_stabilizer(now_dt: datetime):
    to_del = [k for k,s in stabilizer.items() if (now_dt - s["last_time"]).total_seconds() > STABLE_WINDOW]
    for k in to_del: stabilizer.pop(k, None)

# ========= МОДЕЛИ =========
model = YOLO(YOLO_WEIGHTS)
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', det=False, rec=True)

def to_rgb(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def run_ocr_candidates(img_list):
    best_txt, best_conf = None, 0.0
    raw_samples = []
    for img in img_list:
        rgb = to_rgb(img)
        res = ocr_engine.ocr(rgb, cls=True, det=False)
        if not res:
            continue
        for entry in res:
            for item in entry:
                txt, conf = None, None
                if isinstance(item, (list, tuple)):
                    if len(item) == 2 and isinstance(item[0], str):
                        txt, conf = item[0], float(item[1])
                    elif len(item) == 2 and not isinstance(item[0], str):
                        pair = item[1]
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            txt, conf = pair[0], float(pair[1])
                if txt is not None and conf is not None:
                    raw_samples.append((txt, conf))
                    if conf > best_conf:
                        best_conf, best_txt = conf, txt
    return best_txt, best_conf, raw_samples

# ========= ВИДЕОЦИКЛ =========
print("Нажми 'q' для выхода.")
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видеоисточник: {VIDEO_SOURCE}")

ensure_csv()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, conf=DET_CONF, verbose=False)[0]
    yolo_boxes = results.boxes
    now_dt = datetime.now()
    purge_stabilizer(now_dt)

    # DEBUG: счётчик боксов
    if DEBUG:
        cv2.putText(frame, f"YOLO boxes: {0 if yolo_boxes is None else len(yolo_boxes)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    if not yolo_boxes or len(yolo_boxes) == 0:
        if DEBUG and SHOW_REASONS:
            cv2.putText(frame, "No YOLO boxes (check weights/path/conf)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)
        cv2.imshow("RecNumbres", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        continue

    raw_dump_lines = []  # для сырых OCR-кандидатов

    for box in yolo_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0,x1); y1 = max(0,y1); x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (50,200,50), 2)  # рисуем ВСЕ боксы

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 1) выпрямление (можно отключить)
        roi = rectify_plate(crop) if RECTIFY_ENABLED else crop

        # 2) подготовка вариантов и OCR
        variants = preprocess_roi_for_ocr(roi)
        if not variants:
            continue
        raw_text, conf_ocr, raw_samples = run_ocr_candidates(variants)

        if SHOW_RAW_OCR and raw_samples:
            # покажем первые 3 варианты сырых чтений
            for i,(t,c) in enumerate(raw_samples[:3]):
                raw_dump_lines.append(f"{t} ({c:.2f})")

        if not raw_text:
            if DEBUG and SHOW_REASONS:
                cv2.putText(frame, "OCR empty", (x1, max(0, y1-25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            continue

        if conf_ocr < MIN_OCR_CONF:
            if DEBUG and SHOW_REASONS:
                cv2.putText(frame, f"low conf {conf_ocr:.2f}", (x1, max(0, y1-25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            continue

        plate = clean_text(raw_text)
        if not is_valid_ru_plate(plate):
            if DEBUG and SHOW_REASONS:
                cv2.putText(frame, f"regex fail: {plate}", (x1, max(0, y1-25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            # Покажем всё равно, чтобы видеть, ЧТО читает OCR
            cv2.putText(frame, plate, (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
            continue

        # 3) стабилизация по ключу bbox
        key = f"{( (x1+x2)//2 // GRID )*GRID}_{( (y1+y2)//2 // GRID )*GRID}"
        rec = stabilizer.get(key)
        if rec is None:
            stabilizer[key] = {"text": plate, "conf": conf_ocr,
                               "first_time": now_dt, "last_time": now_dt, "count": 1}
        else:
            if plate == rec["text"]:
                rec["count"] += 1
                rec["last_time"] = now_dt
                if conf_ocr > rec["conf"]:
                    rec["conf"] = conf_ocr
            else:
                stabilizer[key] = {"text": plate, "conf": conf_ocr,
                                   "first_time": now_dt, "last_time": now_dt, "count": 1}

        # решение о фиксации
        rec = stabilizer[key]
        if rec["count"] >= REPEAT_VOTES or (now_dt - rec["first_time"]).total_seconds() >= STABLE_WINDOW:
            upsert_plate(rec["text"], now_dt.isoformat(timespec="seconds"))
            stabilizer.pop(key, None)

        # Визуализация финального текста на боксе
        cv2.putText(frame, plate, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # DEBUG: сырые OCR
    if SHOW_RAW_OCR and raw_dump_lines:
        y = 80
        cv2.putText(frame, "RAW OCR:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        for line in raw_dump_lines[:5]:
            y += 18
            cv2.putText(frame, line[:60], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    cv2.imshow("RecNumbers", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
