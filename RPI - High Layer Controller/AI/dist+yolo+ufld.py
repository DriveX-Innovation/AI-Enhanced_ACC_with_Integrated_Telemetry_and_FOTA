import cv2
import time
import numpy as np
import random
import onnxruntime as ort
import traceback

from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
from TrafficLaneDetector.ufldDetector.utils import LaneModelType
from TrafficLaneDetector.ufldDetector.ultrafastLaneDetector import UltrafastLaneDetector
from TrafficLaneDetector.ufldDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2

# ==========================
# CONFIGURATION
# ==========================
VIDEO_PATH = r"D:\graduation-project-related\Vehicle-CV-ADAS-master\temp\test.mp4"

MODEL_PATH_YOLO = "models/yolov8n.onnx"
CLASSES_PATH    = "models/coco_label.txt"
CONF_THRES      = 0.25
IOU_THRES       = 0.45
INPUT_SIZE      = (480, 640)

MODEL_PATH_UFLD = "models/ultra_falst_lane_detection_tusimple_288x800.bin"
MODEL_TYPE_UFLD = LaneModelType.UFLD_TUSIMPLE

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

def draw_fps(frame, fps):
    """Draw a clean FPS counter with black background in top-left corner."""
    label = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (8, 8), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(frame, label, (14, th + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


# ==========================
# YOLOv8 CPU DETECTOR
# ==========================
class YOLOv8_CPU:
    def __init__(self, model_path):
        print("  Loading YOLOv8 ONNX (CPU only)...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
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
        results = [
            (
                boxes[i][0],
                boxes[i][1],
                boxes[i][0] + boxes[i][2],
                boxes[i][1] + boxes[i][3],
                scores[i],
                class_ids[i],
            )
            for i in keep
        ]
        return results

    def detect(self, frame):
        img, r, pad_x, pad_y = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: img})[0]
        return self.postprocess(output, r, pad_x, pad_y, frame.shape)


# ==========================
# MAIN ADAS LOOP
# ==========================
if __name__ == "__main__":
    try:
        print("Step 1: Loading classes...")
        classes = load_classes(CLASSES_PATH)
        print(f"  OK - {len(classes)} classes loaded")
        colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(len(classes))}

        print("Step 2: Loading YOLOv8...")
        yolo_model = YOLOv8_CPU(MODEL_PATH_YOLO)
        print("  OK")

        print("Step 3: Loading distance detector...")
        distanceDetector = SingleCamDistanceMeasure()
        print("  OK")

        print(f"Step 4: Loading lane detector ({MODEL_TYPE_UFLD.name})...")
        if "UFLDV2" in MODEL_TYPE_UFLD.name:
            lane_detector = UltrafastLaneDetectorV2(MODEL_PATH_UFLD, MODEL_TYPE_UFLD)
        else:
            lane_detector = UltrafastLaneDetector(MODEL_PATH_UFLD, MODEL_TYPE_UFLD)
        print("  OK")

        print(f"Step 5: Opening video: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("  FAILED - video not found or unreadable.")
            input("Press Enter to exit...")
            exit()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS)
        vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  OK - {vid_w}x{vid_h} @ {video_fps:.1f} fps  |  {total_frames} frames")

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("  FAILED to decode first frame.")
            print(f"  ffmpeg -i \"{VIDEO_PATH}\" -vcodec libx264 -acodec aac temp/test_fixed.mp4")
            input("Press Enter to exit...")
            exit()

        print("  First frame decoded OK — starting pipeline...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("Step 6: Entering main loop... (press Q to quit)")
        cv2.namedWindow("ADAS Output", cv2.WINDOW_NORMAL)

        # FPS tracking — updated every 10 frames for a stable reading
        fps         = 0.0
        frame_count = 0
        fps_start   = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("  Video ended.")
                break

            output_img = frame.copy()

            # A. Lane detection
            lane_detector.DetectFrame(output_img)
            lane_detector.DrawAreaOnFrame(output_img)
            lane_detector.DrawDetectedOnFrame(output_img)

            # B. Object & distance detection
            detections = yolo_model.detect(frame)
            distanceDetector.updateDistance(detections, classes)

            for x1, y1, x2, y2, conf, cls_id in detections:
                label = f"{classes[cls_id]} {conf:.2f}"
                color = colors[cls_id]
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output_img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            distanceDetector.DrawDetectedOnFrame(output_img)

            # C. FCW ego-lane collision check
            ego_lane_poly = lane_detector.lane_info.area_points
            if ego_lane_poly is not None and len(ego_lane_poly) > 0:
                poly_np = np.array(ego_lane_poly, dtype=np.int32)
                collision_target = distanceDetector.calcCollisionPoint(poly_np)
                if collision_target:
                    dist_to_front = collision_target[2]
                    alert = f"!! FCW: {dist_to_front:.1f}m !!"
                    (tw, th), _ = cv2.getTextSize(
                        alert, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3
                    )
                    ax = vid_w // 2 - tw // 2
                    # Black background behind alert text
                    cv2.rectangle(output_img,
                                  (ax - 8, 55),
                                  (ax + tw + 8, 55 + th + 16),
                                  (0, 0, 0), -1)
                    cv2.putText(output_img, alert, (ax, 55 + th + 8),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2,
                                (0, 0, 255), 3, cv2.LINE_AA)

            # D. FPS counter — update every 10 frames
            frame_count += 1
            if frame_count >= 10:
                elapsed     = time.time() - fps_start
                fps         = frame_count / elapsed
                frame_count = 0
                fps_start   = time.time()

            draw_fps(output_img, fps)

            cv2.imshow("ADAS Output", output_img)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                print("  User quit.")
                break

    except Exception as e:
        traceback.print_exc()
        input("Press Enter to exit...")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()