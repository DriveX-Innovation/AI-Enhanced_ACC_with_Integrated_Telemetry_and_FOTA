import cv2
import time
import numpy as np
import random
import onnxruntime as ort
import serial

# --- Custom Module Imports ---
# Make sure these match your project's folder structure
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
from TrafficLaneDetector.ufldDetector.utils import LaneModelType
from TrafficLaneDetector.ufldDetector.ultrafastLaneDetector import UltrafastLaneDetector
from TrafficLaneDetector.ufldDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2

# ==========================
# UART CONFIGURATION (PC)
# ==========================
# Change 'COM3' to whatever port your USB-to-TTL adapter uses. 
# You can check "Device Manager -> Ports (COM & LPT)" on Windows.
COM_PORT = 'COM3' 
BAUD_RATE = 115200

try:
    ser = serial.Serial(port=COM_PORT, baudrate=BAUD_RATE, timeout=1)
    print(f"✅ UART initialized on PC ({COM_PORT})")
except Exception as e:
    print(f"⚠️ UART Warning: Could not open {COM_PORT}. Is the USB adapter plugged in? ({e})")
    ser = None

# ==========================
# ADAS CONFIGURATION
# ==========================
VIDEO_PATH = "./temp/test.mp4"

# YOLO config
MODEL_PATH_YOLO = "models/yolov8n.onnx"
CLASSES_PATH = "models/coco_label.txt"
CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT_SIZE = (640, 640) # Changed to 640 for standard YOLOv8 models

# UFLD config
MODEL_PATH_UFLD = "models/ultra_falst_lane_detection_culane_288x800.onnx"
MODEL_TYPE_UFLD = LaneModelType.UFLD_CULANE


# ==========================
# UTILITY FUNCTIONS (YOLO)
# ==========================
def load_classes(path):
    with open(path, "r") as f:
        return [c.strip() for c in f.readlines()]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(w * r), int(h * r)
    img_resized = cv2.resize(img, (nw, nh))
    pad_w, pad_h = new_shape[1] - nw, new_shape[0] - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, left, top

def nms(boxes, scores, iou_thres):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=CONF_THRES, nms_threshold=iou_thres
    )
    return indices.flatten() if len(indices) else []


# ==========================
# YOLOv8 CPU DETECTOR
# ==========================
class YOLOv8_CPU:
    def __init__(self, model_path):
        print(" Loading YOLOv8 ONNX (CPU only)...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        img, r, pad_x, pad_y = letterbox(frame, INPUT_SIZE)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, r, pad_x, pad_y

    def postprocess(self, output, r, pad_x, pad_y, orig_shape):
        output = output.squeeze(0).T
        boxes, scores, class_ids = [], [], []
        for det in output:
            x, y, w, h = det[:4]
            cls_scores = det[4:]
            class_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[class_id])
            if conf < CONF_THRES:
                continue
            x1 = max(0, min((x - w / 2 - pad_x) / r, orig_shape[1]))
            y1 = max(0, min((y - h / 2 - pad_y) / r, orig_shape[0]))
            x2 = max(0, min((x + w / 2 - pad_x) / r, orig_shape[1]))
            y2 = max(0, min((y + h / 2 - pad_y) / r, orig_shape[0]))
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(conf)
            class_ids.append(class_id)

        keep = nms(boxes, scores, IOU_THRES)
        results = [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], scores[i], class_ids[i]) for i in keep]
        return results

    def detect(self, frame):
        img, r, pad_x, pad_y = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: img})[0]
        return self.postprocess(output, r, pad_x, pad_y, frame.shape)


# ==========================
# MAIN ADAS LOOP
# ==========================
if __name__ == "__main__":
    # Initialize Models
    classes = load_classes(CLASSES_PATH)
    colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(len(classes))}
    yolo_model = YOLOv8_CPU(MODEL_PATH_YOLO)
    distanceDetector = SingleCamDistanceMeasure()

    print("Model Type : ", MODEL_TYPE_UFLD.name)
    if "UFLDV2" in MODEL_TYPE_UFLD.name:
        lane_detector = UltrafastLaneDetectorV2(MODEL_PATH_UFLD, MODEL_TYPE_UFLD)
    else:
        lane_detector = UltrafastLaneDetector(MODEL_PATH_UFLD, MODEL_TYPE_UFLD)

    # Initialize Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"The video path [{VIDEO_PATH}] cannot be found!")
        exit()

    cv2.namedWindow("ADAS Output", cv2.WINDOW_NORMAL)

    fps = 0
    frame_count = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_img = frame.copy()

        # ==========================
        # A. LANE DETECTION
        # ==========================
        lane_detector.DetectFrame(output_img)
        lane_detector.DrawAreaOnFrame(output_img)
        lane_detector.DrawDetectedOnFrame(output_img)

        # ==========================
        # B. OBJECT & DISTANCE DETECTION
        # ==========================
        detections = yolo_model.detect(frame)
        distanceDetector.updateDistance(detections, classes)

        # Draw YOLO boxes
        for x1, y1, x2, y2, conf, cls_id in detections:
            label = f"{classes[cls_id]} {conf:.2f}"
            color = colors[cls_id]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw distances on screen
        distanceDetector.DrawDetectedOnFrame(output_img)

        # ==========================
        # C. EGO-LANE COLLISION CHECK & UART
        # ==========================
        ego_lane_poly = lane_detector.lane_info.area_points
        
        if ego_lane_poly is not None and len(ego_lane_poly) > 0:
            collision_target = distanceDetector.calcCollisionPoint(ego_lane_poly)
            
            if collision_target:
                dist_to_front_car = collision_target[2]
                
                # Format string for UART (e.g., "D:15.52\n")
                uart_msg = f"D:{dist_to_front_car:.2f}\n"
                
                # Print to PC Terminal
                print(f"⚠️ TARGET IN LANE: Front car distance: {dist_to_front_car:.2f}m | Sending -> {uart_msg.strip()}")
                
                # Send via UART
                if ser and ser.is_open:
                    ser.write(uart_msg.encode('utf-8'))
            else:
                # If lane is clear, you can optionally send a clear signal
                if ser and ser.is_open:
                    ser.write(b"D:CLEAR\n")

        # ==========================
        # D. FPS & DISPLAY
        # ==========================
        frame_count += 1
        if frame_count >= 30:
            end = time.time()
            fps = frame_count / (end - start)
            frame_count = 0
            start = time.time()

        cv2.putText(output_img, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("ADAS Output", output_img)

        # Press 'q' or 'ESC' to stop
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    # Cleanup
    if ser and ser.is_open:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()