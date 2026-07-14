import cv2
import time
import os
from ultralytics import YOLO
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# Paths
MODEL_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n_ncnn_model"
INPUT_VIDEO = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/temp/test.mp4"
OUTPUT_VIDEO = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/temp/output_fps.mp4"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    print("--- ADAS HEADLESS PROCESSOR STARTING ---")
    model = YOLO(MODEL_PATH)
    classes = [model.names[i] for i in range(len(model.names))]
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (width, height))

    distanceDetector = SingleCamDistanceMeasure()
    
    frame_count = 0
    total_start = time.time()
    batch_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)

        # ADAS Processing
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append((x1, y1, x2, y2, conf, cls_id))

        distanceDetector.updateDistance(detections, classes)
        
        # Annotate & Write
        annotated_frame = results[0].plot()
        distanceDetector.DrawDetectedOnFrame(annotated_frame)
        out.write(annotated_frame)
        
        frame_count += 1
        
        # Print FPS to terminal every 30 frames
        if frame_count % 30 == 0:
            now = time.time()
            batch_fps = 30 / (now - batch_start)
            print(f"Progress: {frame_count} frames | Live Speed: {batch_fps:.2f} FPS")
            batch_start = now

    cap.release()
    out.release()
    
    total_duration = time.time() - total_start
    avg_fps = frame_count / total_duration
    print("\n--- PROCESSING COMPLETE ---")
    print(f"Total Frames: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Final Video Saved: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
