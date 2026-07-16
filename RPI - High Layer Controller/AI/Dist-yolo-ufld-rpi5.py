import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
import numpy as np
from ultralytics import YOLO

from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n_ncnn_model"
VIDEO_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/temp/test.mp4"

YOLO_SKIP_FRAMES = 2
INFER_W = 640
INFER_H = 640

# ==========================
# ROI
# ==========================
def get_roi_vertices(width, height):
    return np.array([[
        (100, height),
        (width // 2 - 150, height // 2 + 50),
        (width // 2 + 150, height // 2 + 50),
        (width - 100, height)
    ]], dtype=np.int32)

# ==========================
# 🔥 IMPROVED LANE DETECTION
# ==========================
def detect_lanes_traditional(frame, roi_vertices):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Color filtering (white + yellow) ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    masked = cv2.bitwise_and(gray, gray, mask=color_mask)

    blur = cv2.GaussianBlur(masked, (5, 5), 0)

    # Slightly stricter thresholds = less noise
    edges = cv2.Canny(blur, 75, 200)

    # ROI mask
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    edges = cv2.bitwise_and(edges, mask)

    # Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                            minLineLength=40, maxLineGap=150)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Remove horizontal noise
            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

    line_img = np.zeros_like(frame)

    def average_lane(lines):
        if len(lines) == 0:
            return None

        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]

        poly = np.polyfit(y_coords, x_coords, 1)

        y1 = frame.shape[0]
        y2 = int(frame.shape[0] * 0.6)

        x1 = int(poly[0]*y1 + poly[1])
        x2 = int(poly[0]*y2 + poly[1])

        return x1, y1, x2, y2

    left_lane = average_lane(left_lines)
    right_lane = average_lane(right_lines)

    if left_lane is not None:
        cv2.line(line_img, (left_lane[0], left_lane[1]),
                 (left_lane[2], left_lane[3]), (255, 0, 0), 6)

    if right_lane is not None:
        cv2.line(line_img, (right_lane[0], right_lane[1]),
                 (right_lane[2], right_lane[3]), (255, 0, 0), 6)

    return line_img

# ==========================
# MAIN
# ==========================
def main():
    print("Loading YOLOv8 NCNN Model...")
    model = YOLO(MODEL_PATH, task='detect')
    classes = [model.names[i] for i in range(len(model.names))]

    print("Loading Video Stream...")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ego_poly = get_roi_vertices(vid_w, vid_h)

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    frame_idx = 0

    distanceDetector = SingleCamDistanceMeasure()

    last_detections = []
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_idx += 1

        # ==========================
        # 1. LANE DETECTION (FAST)
        # ==========================
        lane_mask = detect_lanes_traditional(frame, ego_poly)
        base_frame = cv2.addWeighted(frame, 1.0, lane_mask, 1.0, 0)

        # ==========================
        # 2. YOLO (SKIPPED FRAMES)
        # ==========================
        if frame_idx % YOLO_SKIP_FRAMES == 0:
            results = model.predict(
                frame,
                imgsz=INFER_W,
                conf=0.25,
                verbose=False
            )

            last_detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                last_detections.append((x1, y1, x2, y2, conf, cls_id))

            distanceDetector.updateDistance(last_detections, classes)
            last_annotated = results[0].plot(img=base_frame)

        else:
            last_annotated = base_frame.copy()
            for x1, y1, x2, y2, conf, cls_id in last_detections:
                label = f"{classes[cls_id]} {conf:.2f}"
                cv2.rectangle(last_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(last_annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        annotated_frame = last_annotated.copy()

        # ==========================
        # EGO ZONE
        # ==========================
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, ego_poly, (0, 100, 255))
        annotated_frame = cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0)

        distanceDetector.DrawDetectedOnFrame(annotated_frame)

        # ==========================
        # FCW
        # ==========================
        collision_target = distanceDetector.calcCollisionPoint(ego_poly[0])

        if collision_target:
            dist = collision_target[2]
            alert = f"!! FCW: {dist:.1f}m !!"

            (tw, th), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
            ax = vid_w // 2 - tw // 2

            cv2.rectangle(annotated_frame,
                          (ax - 8, 55), (ax + tw + 8, 55 + th + 16),
                          (0, 0, 0), -1)

            cv2.putText(annotated_frame, alert, (ax, 55 + th + 8),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        # ==========================
        # FPS
        # ==========================
        frame_count += 1
        if frame_count >= 10:
            fps = frame_count / (time.time() - start_time)
            frame_count = 0
            start_time = time.time()

        cv2.rectangle(annotated_frame, (8, 8), (170, 45), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("ADAS System - Optimized Lane + YOLO", annotated_frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
