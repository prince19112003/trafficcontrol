"""
Real-time Vehicle + Number Plate Detection and OCR using YOLOv8 + EasyOCR
Author: Prince Thakur (Final Year Project)
Dependencies: ultralytics, opencv-python, easyocr, pyserial
"""

# ----------------- Imports -----------------
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import time

# Uncomment this if using Arduino
# import serial

# ----------------- Configuration -----------------
MODEL_VEHICLE = "yolov8n.pt"       # YOLOv8 model for vehicles (COCO pretrained)
MODEL_PLATE = "plate_model.pt"     # Optional YOLOv8 model for license plates
USE_PLATE_YOLO = False             # Set True if you have a plate model
VIDEO_SOURCE = "./video.mp4"                   # 0 = webcam, or path to video file
SERIAL_PORT = None                 # e.g. "COM3" for Arduino
SERIAL_BAUD = 9600
DETECT_VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, motorcycle, bus, truck (COCO)
OCR_LANGS = ['en']                 # Language for EasyOCR
# -------------------------------------------------

# ----------------- Initialization -----------------
print("[INFO] Loading YOLOv8 vehicle model...")
vehicle_model = YOLO(MODEL_VEHICLE)

plate_model = None
if USE_PLATE_YOLO:
    try:
        plate_model = YOLO(MODEL_PLATE)
        print("[INFO] License plate model loaded successfully.")
    except Exception as e:
        print("[WARN] Could not load plate model:", e)
        plate_model = None

print("[INFO] Initializing OCR engine...")
reader = easyocr.Reader(OCR_LANGS, gpu=False)

# Optional Arduino serial connection
arduino = None
if SERIAL_PORT:
    try:
        import serial
        arduino = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        print("[INFO] Arduino connected on", SERIAL_PORT)
    except Exception as e:
        print("[WARN] Could not connect to Arduino:", e)
        arduino = None

# ----------------- Helper Functions -----------------
def preprocess_plate_roi(roi):
    """Enhance image for better OCR reading"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def find_plate_candidates(vehicle_img):
    """Find possible license plate areas if no YOLO plate model is used"""
    gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradX = cv2.convertScaleAbs(gradX)
    _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    H, W = gray.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / (h + 1e-6)
        area = w * h
        if area < 400: continue
        if 2.0 < aspect < 6.5 and 0.01*W*H < area < 0.5*W*H:
            candidates.append((x, y, w, h))
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    return candidates[:3]

# ----------------- Main Loop -----------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("[ERROR] Cannot open video source!")
    exit()

print("[INFO] Starting real-time detection... Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of stream or camera error.")
        break

    orig = frame.copy()
    results = vehicle_model(frame)
    boxes = results[0].boxes

    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        cls_id = int(cls)
        if cls_id not in DETECT_VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        vehicle_roi = orig[y1:y2, x1:x2]
        plate_text, plate_conf = None, 0.0
        plate_box_abs = None

        # --- Use YOLO Plate Model if available ---
        if plate_model:
            res_plate = plate_model(vehicle_roi)
            for p in res_plate:
                for pb, pconf in zip(p.boxes.xyxy, p.boxes.conf):
                    px1, py1, px2, py2 = map(int, pb.tolist())
                    plate_roi = vehicle_roi[py1:py2, px1:px2]
                    if plate_roi.size == 0:
                        continue
                    proc = preprocess_plate_roi(plate_roi)
                    ocr_res = reader.readtext(proc)
                    if ocr_res:
                        text = ocr_res[0][1]
                        conf_ocr = ocr_res[0][2]
                        if conf_ocr > plate_conf:
                            plate_text = text
                            plate_conf = conf_ocr
                            plate_box_abs = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)

        # --- Fallback: OpenCV + OCR ---
        else:
            candidates = find_plate_candidates(vehicle_roi)
            for (cx, cy, cw, ch) in candidates:
                roi = vehicle_roi[cy:cy+ch, cx:cx+cw]
                proc = preprocess_plate_roi(roi)
                ocr_res = reader.readtext(proc)
                for det in ocr_res:
                    text = det[1]
                    conf_ocr = det[2]
                    candidate_text = ''.join(ch for ch in text if ch.isalnum())
                    if len(candidate_text) >= 4 and conf_ocr > plate_conf:
                        plate_text = candidate_text
                        plate_conf = conf_ocr
                        plate_box_abs = (x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch)

        # --- Draw Results ---
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0,255,0), 2)
        if plate_text and plate_box_abs:
            px1, py1, px2, py2 = plate_box_abs
            cv2.rectangle(orig, (px1, py1), (px2, py2), (0,165,255), 2)
            cv2.putText(orig, f"{plate_text}", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
            print(f"[DETECTED] Plate: {plate_text} | Confidence: {plate_conf:.2f}")

            # Send to Arduino if connected
            if arduino:
                try:
                    arduino.write(f"PLATE:{plate_text}\n".encode())
                except:
                    pass

    cv2.imshow("Real-Time Vehicle + Plate Detection", orig)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
print("[INFO] Program ended.")
