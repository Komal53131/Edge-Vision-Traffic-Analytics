import cv2
import csv
from datetime import datetime
from ultralytics import YOLO
# 1. Model Load
model = YOLO("yolov8n.pt")
# 2. Source Setup (0 for webcam, ya "traffic.mp4")
cap = cv2.VideoCapture(0)
# 3. Class Mapping & Filtering
TARGET_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
# Line Position & Trackers
LINE_Y = 250
counted_ids = set()
category_counts = {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
# 4. CSV File Setup (Data Logging)
csv_file = open("traffic_log.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)
if csv_file.tell() == 0:
    csv_writer.writerow(["Timestamp", "Track_ID", "Vehicle_Type", "Total_Crossed"])
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # YOLO Tracking
    results = model.track(frame, persist=True, verbose=False, classes=TARGET_CLASSES, conf=0.5)
    # Virtual Counting Line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 3)
    cv2.putText(frame, "COUNTING LINE", (10, LINE_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Detection & Processing
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().tolist()
        clss = results[0].boxes.cls.int().tolist()
        ids = results[0].boxes.id.int().tolist()
        for box, cls_id, track_id in zip(boxes, clss, ids):
            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)
            vehicle_type = CLASS_NAMES.get(cls_id, 'Vehicle')
            # Render Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_type} ID:{track_id}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Category-wise Trigger & Logging Logic
            if (LINE_Y - 15) < center_y < (LINE_Y + 15):
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    if vehicle_type in category_counts:
                        category_counts[vehicle_type] += 1
                    # CSV Record Entry
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([timestamp, track_id, vehicle_type, len(counted_ids)])
                    csv_file.flush()
    # Analytics Dashboard Overlay
    y_offset = 30
    cv2.putText(frame, f"Total Crossed: {len(counted_ids)}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for v_type, count in category_counts.items():
        y_offset += 25
        cv2.putText(frame, f"{v_type}s: {count}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("EdgeVision - Traffic Analytics", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Resource Cleanups
csv_file.close()
cap.release()
cv2.destroyAllWindows()