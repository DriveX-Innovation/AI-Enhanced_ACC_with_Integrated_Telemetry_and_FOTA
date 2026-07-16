import cv2
import time
from ultralytics import YOLO
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# ==========================
# ABSOLUTE CONFIGURATION
# ==========================
MODEL_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n_ncnn_model"
VIDEO_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/temp/test.mp4"

def main():
    print("Loading YOLOv8 NCNN Model...")
    # Initialize the NCNN model natively
    model = YOLO(MODEL_PATH,task='detect')
    
    
    # Extract class names as a list for your distance detector
    classes = [model.names[i] for i in range(len(model.names))]

    # Load video with absolute path
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = 0
    frame_count = 0
    start_time = time.time()
    distanceDetector = SingleCamDistanceMeasure()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # 1. Run Inference (verbose=False keeps your terminal clean)
        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)

        # 2. Reformat detections for your Distance Measure class
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append((x1, y1, x2, y2, conf, cls_id))

        # 3. Update ADAS Logic
        distanceDetector.updateDistance(detections, classes)

        # 4. Draw bounding boxes automatically using Ultralytics
        annotated_frame = results[0].plot()

        # 5. Draw distance overlays
        distanceDetector.DrawDetectedOnFrame(annotated_frame)

        # FPS Calculation
        frame_count += 1
        if frame_count >= 30:
            fps = 30 / (time.time() - start_time)
            frame_count = 0
            start_time = time.time()

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("ADAS System - NCNN", annotated_frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
