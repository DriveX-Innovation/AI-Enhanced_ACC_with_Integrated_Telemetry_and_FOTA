import torch

# Intercept PyTorch's load function and force weights_only=False
original_load = torch.load
def bypass_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = bypass_load

from ultralytics import YOLO

print("Loading YOLOv8n...")
# Pointing directly to the file you transferred
model = YOLO("/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n.pt")

print("Exporting to NCNN...")
model.export(format="ncnn")

print("Export complete!")
