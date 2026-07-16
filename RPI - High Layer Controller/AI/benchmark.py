import time
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO("/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n_ncnn_model")

# Create a blank dummy frame (640x640, 3 channels)
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

print("Warming up...")
for _ in range(5):
    model.predict(dummy_frame, imgsz=640, verbose=False)

print("Running pure inference benchmark (100 frames)...")
start_time = time.time()

for _ in range(100):
    model.predict(dummy_frame, imgsz=640, verbose=False)

total_time = time.time() - start_time
print(f"Total Time: {total_time:.2f}s")
print(f"RAW MAXIMUM NCNN FPS: {100 / total_time:.2f}")
