from ultralytics import YOLO
from collections import Counter
from PIL import Image
import torch

# Load YOLOv8 model (replace with your custom model if trained on food)
model = YOLO("yolov8n.pt")  # You can replace with "best.pt" if you trained a model

# Load the image
image_path = "food3.jpg"  # Replace with your image path
image = Image.open(image_path)

# Run inference
results = model(image_path)

# Display the result with bounding boxes (optional)
results[0].show()

# Extract bounding boxes and class IDs
detected_classes = results[0].boxes.cls.tolist()
bounding_boxes = results[0].boxes.xyxy.tolist()

# Count detected items
class_counts = Counter(detected_classes)

# Map class IDs to names
names = model.names

print("\n🔍 Detected Food Items:")
for class_id, count in class_counts.items():
    class_name = names[int(class_id)]
    print(f"- {class_name}: {count}")

print("\n📏 Estimated Portion Sizes (bounding box area):")
for i, box in enumerate(bounding_boxes):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    class_name = names[int(detected_classes[i])]
    print(f"- {class_name} approx. area: {area:.2f} px²")
