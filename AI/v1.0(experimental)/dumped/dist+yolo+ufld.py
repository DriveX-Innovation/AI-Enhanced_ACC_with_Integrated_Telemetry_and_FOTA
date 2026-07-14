import cv2
import time
import numpy as np
import random
import onnxruntime as ort

# --- Custom Module Imports ---
# Assuming your YOLO classes/functions are either in this file or imported here.
# If they are in this file, keep your YOLOv8_CPU and letterbox functions above this.
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
from TrafficLaneDetector.ufldDetector.utils import LaneModelType
from TrafficLaneDetector.ufldDetector.ultrafastLaneDetector import UltrafastLaneDetector
from TrafficLaneDetector.ufldDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2

# ==========================
# CONFIGURATION
# ==========================
# Video config
VIDEO_PATH = "./temp/test.mp4"

# YOLO config
MODEL_PATH_YOLO = "models/yolov8n.onnx"
CLASSES_PATH = "models/coco_label.txt"
CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT_SIZE = (480, 640)

# UFLD Lane Detection config
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
    # 1. Initialize YOLO and Classes
    classes = load_classes(CLASSES_PATH)
    colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(len(classes))}
    yolo_model = YOLOv8_CPU(MODEL_PATH_YOLO)
    
    # 2. Initialize Distance Measure
    distanceDetector = SingleCamDistanceMeasure()

    # 3. Initialize UFLD Lane Detector
    print("Model Type : ", MODEL_TYPE_UFLD.name)
    if "UFLDV2" in MODEL_TYPE_UFLD.name:
        lane_detector = UltrafastLaneDetectorV2(MODEL_PATH_UFLD, MODEL_TYPE_UFLD)
    else:
        lane_detector = UltrafastLaneDetector(MODEL_PATH_UFLD, MODEL_TYPE_UFLD)

    # 4. Initialize Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"The video path [{VIDEO_PATH}] cannot be found!")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow("ADAS Output", cv2.WINDOW_NORMAL)

    fps = 0
    frame_count = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy of the frame to draw on, keeping the original clean for YOLO
        output_img = frame.copy()

        # ==========================
        # A. LANE DETECTION
        # ==========================
        # Run inference (updates internal lane_info)
        lane_detector.DetectFrame(output_img)
        
        # Draw the ego-lane area (polygon) and the lane lines on the image
        lane_detector.DrawAreaOnFrame(output_img)
        lane_detector.DrawDetectedOnFrame(output_img)

        # ==========================
        # B. OBJECT & DISTANCE DETECTION
        # ==========================
        # Run YOLO inference
        detections = yolo_model.detect(frame)
        
        # Calculate distances for all detected objects (passing classes array!)
        distanceDetector.updateDistance(detections, classes)

        # Draw YOLO bounding boxes and labels
        for x1, y1, x2, y2, conf, cls_id in detections:
            label = f"{classes[cls_id]} {conf:.2f}"
            color = colors[cls_id]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw all distances on the screen
        distanceDetector.DrawDetectedOnFrame(output_img)

        # ==========================
        # C. EGO-LANE COLLISION CHECK (FCW)
        # ==========================
        # Grab the polygon representing your current lane from UFLD
        ego_lane_poly = lane_detector.lane_info.area_points
        
        # Check if we successfully generated a polygon for the lane
        if ego_lane_poly is not None and len(ego_lane_poly) > 0:
            
            # Check if any object falls inside this polygon
            collision_target = distanceDetector.calcCollisionPoint(ego_lane_poly)
            
            if collision_target:
                # collision_target returns [Xcenter, Ybottom, distance]
                dist_to_front_car = collision_target[2]
                
                # ONLY print to terminal if the object is in our lane!
                print(f"⚠️ TARGET IN LANE: Front car distance: {dist_to_front_car:.2f} meters")

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

    cap.release()
    cv2.destroyAllWindows()