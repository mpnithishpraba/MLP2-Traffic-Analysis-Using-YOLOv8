import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

vehicle_classes = ['car', 'motorbike', 'bus', 'truck']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    centroids = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centroids.append((cx, cy))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    centroids.sort(key=lambda x: x[0])

    gaps = []
    for i in range(1, len(centroids)):
        dx = centroids[i][0] - centroids[i - 1][0]
        dy = centroids[i][1] - centroids[i - 1][1]
        gaps.append(np.sqrt(dx**2 + dy**2))

    avg_gap = np.mean(gaps) if gaps else 0
    vehicle_count = len(centroids)

    if vehicle_count < 5 and avg_gap > 150:
        status, color = "LOW TRAFFIC", (0, 255, 0)
    elif vehicle_count < 15 and avg_gap > 50:
        status, color = "MEDIUM TRAFFIC", (0, 255, 255)
    else:
        status, color = "HIGH TRAFFIC", (0, 0, 255)

    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.putText(frame, f"Avg Gap: {avg_gap:.1f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.putText(frame, status, (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)

    out.write(frame)

cap.release()
out.release()

print("output.mp4 generated")