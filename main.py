from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection on a sample image (downloaded automatically)
results = model.predict(source="https://ultralytics.com/images/bus.jpg", show=False)

# Print detections
for box in results[0].boxes:
    print(box)
