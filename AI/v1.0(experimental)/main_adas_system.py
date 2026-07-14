import cv2
import time
import numpy as np
from ultralytics import YOLO
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# Lane Detection Imports
from TrafficLaneDetector.ufldDetector.utils import LaneModelType
from TrafficLaneDetector.ufldDetector.ultrafastLaneDetector import UltrafastLaneDetector

# ==========================
# ABSOLUTE CONFIGURATION
# ==========================
MODEL_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n_ncnn_model"
VIDEO_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/temp/test.mp4"

# FIXED: Removed /openvino/ from the middle so the class builds the path correctly
LANE_MODEL_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/ultra_falst_lane_detection_tusimple_288x800.xml"
LANE_MODEL_TYPE = LaneModelType.UFLD_TUSIMPLE

def main():
    print("Loading YOLOv8 NCNN Model...")
    # Initialize the NCNN model natively
    model = YOLO(MODEL_PATH, task='detect')

    # Extract class names as a list for your distance detector
    classes = [model.names[i] for i in range(len(model.names))]

    print("Loading OpenVINO Lane Detector...")
    # Initialize the OpenVINO Lane Detector
    lane_detector = UltrafastLaneDetector(LANE_MODEL_PATH, LANE_MODEL_TYPE)

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

        # A. LANE DETECTION
        # Detect lanes on the clean frame
        lane_detector.DetectFrame(frame)

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

        # B. DRAW LANES ON ANNOTATED FRAME
        lane_detector.DrawAreaOnFrame(annotated_frame)
        lane_detector.DrawDetectedOnFrame(annotated_frame)

        # 5. Draw distance overlays (Visual only)
        distanceDetector.DrawDetectedOnFrame(annotated_frame)

        # C. TERMINAL-ONLY COLLISION LOGIC
        # Grab the ego-lane polygon
        ego_lane_poly = lane_detector.lane_info.area_points
        if ego_lane_poly is not None and len(ego_lane_poly) > 0:
            poly_np = np.array(ego_lane_poly, dtype=np.int32)
            collision_target = distanceDetector.calcCollisionPoint(poly_np)
            
            if collision_target:
                # collision_target returns [Xcenter, Ybottom, distance]
                dist_to_front_car = collision_target[2]
                # Print alert strictly to terminal as requested
                print(f"\033[91m⚠️  FCW ALERT: Object in Lane at {dist_to_front_car:.2f} meters!\033[0m")

        # FPS Calculation
        frame_count += 1
        if frame_count >= 30:
            fps = 30 / (time.time() - start_time)
            frame_count = 0
            start_time = time.time()

        # Display FPS on frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show Output
        cv2.imshow("ADAS System - NCNN & OpenVINO", annotated_frame)
        
        # Press ESC to exit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
