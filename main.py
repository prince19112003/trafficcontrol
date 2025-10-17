"""
Real-time Vehicle + Number Plate Detection + CSV Logging (Codespace Safe Version)
Author: Prince Thakur
"""

from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import time
import csv
import os

# Uncomment this if using Arduino
# import serial

# ---------------- CONFIG ----------------
MODEL_VEHICLE = "yolov8n.pt"
VIDEO_SOURCE = "video2.mp4"  # Change to 0 for webcam on local PC
SERIAL_PORT = None
SERIAL_BAUD = 9600
DETECT_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
OCR_LANGS = ['en']
OUTPUT_VIDEO = "processed_output.mp4"
CSV_FILE = "detected_plates.csv"
# ----------------------------------------

print("[INFO] Loading YOLOv8 vehicle model...")
vehicle_model = YOLO(MODEL_VEHICLE)

print("[INFO] Initializing OCR engine...")
reader = easyocr.Reader(OCR_LANGS, gpu=False)

# Optional Arduino serial
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


# ---------------- CSV SETUP ----------------
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Plate_Number", "Confidence"])
print(f"[INFO] CSV logging enabled → {CSV_FILE}")
# -------------------------------------------


def log_to_csv(plate, conf):
    """Save detected number plate details into CSV."""
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), plate, f"{conf:.2f}"])


def preprocess_plate_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def find_plate_candidates(vehicle_img):
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


cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("[ERROR] Cannot open video source!")
    exit()

# Prepare video writer (output)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

print("[INFO] Starting real-time detection (headless mode)...")

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

        cv2.rectangle(orig, (x1, y1), (x2, y2), (0,255,0), 2)
        if plate_text and plate_box_abs:
            px1, py1, px2, py2 = plate_box_abs
            cv2.rectangle(orig, (px1, py1), (px2, py2), (0,165,255), 2)
            cv2.putText(orig, f"{plate_text}", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
            print(f"[DETECTED] Plate: {plate_text} | Confidence: {plate_conf:.2f}")
            
            # Save to CSV
            log_to_csv(plate_text, plate_conf)

            # Send to Arduino if connected
            if arduino:
                try:
                    arduino.write(f"PLATE:{plate_text}\n".encode())
                except:
                    pass

    out.write(orig)  # Save each processed frame
    print(f"[INFO] Frame processed @ {time.strftime('%H:%M:%S')}")

cap.release()
out.release()
if arduino:
    arduino.close()

print(f"[INFO] Processing complete.\n[INFO] Output video → {OUTPUT_VIDEO}\n[INFO] CSV log → {CSV_FILE}")
