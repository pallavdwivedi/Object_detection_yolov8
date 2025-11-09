import cv2
import numpy as np
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("Model loaded!")

print("Creating dummy frame...")
dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
print(f"Frame shape: {dummy_frame.shape}")

print("Running inference...")
results = model(dummy_frame, verbose=False)
print("Inference successful!")
print(f"Results: {len(results)} object(s) detected")
print("YOLO is working!")
