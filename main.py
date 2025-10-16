# ============================================================
# AI-Based Real-Time Traffic Signal Control using YOLOv8
# Author: Nikhil Dubey (Final Year Project)
# ============================================================

# Install required libraries (run once):
# pip install ultralytics opencv-python pyserial

from ultralytics import YOLO
import cv2
import time
import serial   # optional if Arduino is connected

# ----------------------------------------------
# CONFIGURATION
# ----------------------------------------------
MODEL_PATH = "yolov8n.pt"  # YOLOv8 model file
VIDEO_SOURCE = "./video.mp4"  # 0 for webcam, or "traffic.mp4" for video

# If using Arduino for actual LEDs, connect serial port
# Comment this line if no Arduino connected
# arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
# time.sleep(2)

# ----------------------------------------------
# INITIALIZATION
# ----------------------------------------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

# ----------------------------------------------
# TRAFFIC LIGHT SIMULATION FUNCTION
# ----------------------------------------------
def control_traffic_lights(vehicle_count):
    """
    Decide green light duration based on detected vehicle count
    """
    if vehicle_count > 20:
        green_time = 40
    elif 10 < vehicle_count <= 20:
        green_time = 25
    else:
        green_time = 15

    print(f"[INFO] Vehicles: {vehicle_count} → Green Light: {green_time} seconds")

    # Optional: Send signal to Arduino
    # arduino.write(f"{green_time}\n".encode())

    # Simulate traffic light sequence
    print("🟢 GREEN light ON")
    time.sleep(green_time)
    print("🟡 YELLOW light ON (3 sec)")
    time.sleep(3)
    print("🔴 RED light ON (3 sec)")
    time.sleep(3)
    print("-" * 50)

# ----------------------------------------------
# MAIN LOOP
# ----------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        annotated_frame = result.plot()

        # Count vehicles (car=2, motorbike=3, bus=5, truck=7)
        vehicle_count = sum(1 for cls in boxes.cls if int(cls) in [2, 3, 5, 7])

        # Display count on frame
        cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Traffic Detection", annotated_frame)

        # Every 10 seconds (or 1 cycle), control lights
        current_time = time.time()
        if int(current_time) % 10 == 0:  # every 10 seconds
            control_traffic_lights(vehicle_count)

    # Press ESC to stop
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
