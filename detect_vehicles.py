from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # lightweight model

# Use a sample video
video_path = "https://github.com/ultralytics/assets/raw/main/videos/traffic.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Vehicle Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
