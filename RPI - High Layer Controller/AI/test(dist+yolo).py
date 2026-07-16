import cv2
import numpy as np
import onnxruntime as ort
import time
import random
from typing import List, Tuple, Optional
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# ==========================
# CONFIG
# ==========================
MODEL_PATH        = "models/yolov8n.onnx"
CLASSES_PATH      = "models/coco_label.txt"
TWIN_MODEL_PATH   = "models/TwinLiteNet.onnx"

CONF_THRES        = 0.25
IOU_THRES         = 0.45
YOLO_INPUT_SIZE   = (480, 640)   # (H, W) – YOLOv8 default

# TwinLiteNet expects 360×640 input (standard BDD100K resolution)
TWIN_INPUT_H      = 360
TWIN_INPUT_W      = 640

# Overlay alpha values
LANE_ALPHA        = 0.55         # lane-line mask blend
DRIVE_ALPHA       = 0.35         # drivable-area mask blend

# Colours (BGR)
LANE_COLOR        = (0,   255, 255)   # cyan
DRIVE_COLOR       = (0,   180,   0)   # green


# ==========================
# UTILS
# ==========================
def load_classes(path: str) -> List[str]:
    with open(path, "r") as f:
        return [c.strip() for c in f.readlines()]


def letterbox(img: np.ndarray,
              new_shape: Tuple[int, int] = (640, 640),
              color: Tuple[int, int, int] = (114, 114, 114)):
    """Resize + pad image to new_shape while preserving aspect ratio."""
    h, w = img.shape[:2]
    r    = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(w * r), int(h * r)
    img_resized = cv2.resize(img, (nw, nh))

    pad_w, pad_h = new_shape[1] - nw, new_shape[0] - nh
    top,  bottom = pad_h // 2, pad_h - pad_h // 2
    left, right  = pad_w // 2, pad_w - pad_w // 2

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, left, top


def nms(boxes, scores, iou_thres: float):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=CONF_THRES,
        nms_threshold=iou_thres
    )
    return indices.flatten() if len(indices) else []


# ==========================
# YOLOv8 CPU DETECTOR
# ==========================
class YOLOv8_CPU:
    def __init__(self, model_path: str):
        print("[YOLOv8] Loading ONNX model (CPU)…")
        so = ort.SessionOptions()
        so.intra_op_num_threads       = 4
        so.graph_optimization_level   = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session    = ort.InferenceSession(
            model_path, sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray):
        img, r, pad_x, pad_y = letterbox(frame, YOLO_INPUT_SIZE)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, r, pad_x, pad_y

    # ------------------------------------------------------------------
    def postprocess(self, output: np.ndarray,
                    r: float, pad_x: int, pad_y: int,
                    orig_shape: Tuple) -> List:
        output = output.squeeze(0).T  # (num_boxes, 84)

        boxes, scores, class_ids = [], [], []

        for det in output:
            x, y, w, h   = det[:4]
            cls_scores   = det[4:]
            class_id     = int(np.argmax(cls_scores))
            conf         = float(cls_scores[class_id])

            if conf < CONF_THRES:
                continue

            x1 = (x - w / 2 - pad_x) / r
            y1 = (y - h / 2 - pad_y) / r
            x2 = (x + w / 2 - pad_x) / r
            y2 = (y + h / 2 - pad_y) / r

            x1 = int(max(0, min(x1, orig_shape[1])))
            y1 = int(max(0, min(y1, orig_shape[0])))
            x2 = int(max(0, min(x2, orig_shape[1])))
            y2 = int(max(0, min(y2, orig_shape[0])))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(class_id)

        keep    = nms(boxes, scores, IOU_THRES)
        results = []
        for i in keep:
            x, y, w, h = boxes[i]
            results.append((x, y, x + w, y + h, scores[i], class_ids[i]))
        return results

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List:
        img, r, pad_x, pad_y = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: img})[0]
        return self.postprocess(output, r, pad_x, pad_y, frame.shape)


# ==========================
# TwinLiteNet LANE DETECTOR
# ==========================
class TwinLiteNetDetector:
    """
    Lightweight dual-head segmentation network that simultaneously predicts:
      • Drivable area  (head 0)
      • Lane lines     (head 1)

    Expected ONNX output shapes (standard TwinLiteNet):
      da_seg  : (1, 2, H, W)   – drivable-area logits
      ll_seg  : (1, 2, H, W)   – lane-line logits

    If your export produces a single tensor (1, 4, H, W), set
    TWIN_SINGLE_OUTPUT = True in the constructor call.
    """

    def __init__(self, model_path: str, single_output: bool = False):
        print("[TwinLiteNet] Loading ONNX model (CPU)…")
        so = ort.SessionOptions()
        so.intra_op_num_threads     = 2
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session       = ort.InferenceSession(
            model_path, sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.input_name    = self.session.get_inputs()[0].name
        self.single_output = single_output   # handle merged-head exports

        # cache output names
        self.output_names  = [o.name for o in self.session.get_outputs()]
        print(f"[TwinLiteNet] Outputs: {self.output_names}")

    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize → normalise → NCHW float32 blob."""
        img = cv2.resize(frame, (TWIN_INPUT_W, TWIN_INPUT_H))
        img = img[:, :, ::-1].astype(np.float32) / 255.0

        # ImageNet mean / std normalisation (matches training pre-proc)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std

        img  = np.transpose(img, (2, 0, 1))
        img  = np.expand_dims(img, axis=0)          # (1, 3, H, W)
        return img.astype(np.float32)

    # ------------------------------------------------------------------
    def postprocess(self, outputs, orig_h: int, orig_w: int):
        """
        Returns two boolean masks in **original frame resolution**:
          drive_mask : drivable area
          lane_mask  : lane lines
        """
        if self.single_output:
            # merged tensor (1, 4, H, W) → split channels 0-1 / 2-3
            raw        = outputs[0]                 # (1, 4, H, W)
            da_logits  = raw[:, :2, :, :]
            ll_logits  = raw[:, 2:, :, :]
        else:
            # separate tensors
            da_logits  = outputs[0]                 # (1, 2, H, W)
            ll_logits  = outputs[1] if len(outputs) > 1 else outputs[0]

        # argmax over class dim → (H, W) uint8
        da_pred = np.argmax(da_logits.squeeze(0), axis=0).astype(np.uint8)
        ll_pred = np.argmax(ll_logits.squeeze(0), axis=0).astype(np.uint8)

        # resize back to original frame size
        drive_mask = cv2.resize(da_pred, (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
        lane_mask  = cv2.resize(ll_pred, (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)

        return drive_mask.astype(bool), lane_mask.astype(bool)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray):
        """End-to-end inference. Returns (drive_mask, lane_mask)."""
        blob    = self.preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self.postprocess(outputs, frame.shape[0], frame.shape[1])


# ==========================
# OVERLAY HELPERS
# ==========================
def overlay_mask(frame: np.ndarray,
                 mask: np.ndarray,
                 color: Tuple[int, int, int],
                 alpha: float) -> np.ndarray:
    """Alpha-blend a boolean mask colour onto the frame in-place."""
    colored          = np.zeros_like(frame, dtype=np.uint8)
    colored[mask]    = color
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)


def draw_lane_contours(frame: np.ndarray,
                       lane_mask: np.ndarray,
                       color: Tuple[int, int, int] = LANE_COLOR,
                       thickness: int = 3) -> np.ndarray:
    """Draw lane-line contours (cleaner than a raw mask overlay)."""
    lane_u8   = (lane_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(lane_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    classes = load_classes(CLASSES_PATH)
    colors  = {i: [random.randint(0, 255) for _ in range(3)]
               for i in range(len(classes))}

    # ---- models -------------------------------------------------------
    yolo_model = YOLOv8_CPU(MODEL_PATH)
    twin_model = TwinLiteNetDetector(
        TWIN_MODEL_PATH,
        single_output=False   # set True if your export is a single 4-ch tensor
    )
    distance_detector = SingleCamDistanceMeasure()

    # ---- video --------------------------------------------------------
    cap = cv2.VideoCapture("./temp/test.mp4")
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source.")

    fps, frame_count = 0.0, 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # ---------- 1. Lane / drivable-area segmentation ---------------
        drive_mask, lane_mask = twin_model.detect(frame)

        # Draw drivable area first (bottom layer)
        frame = overlay_mask(frame, drive_mask, DRIVE_COLOR, DRIVE_ALPHA)

        # Draw lane contours on top
        frame = draw_lane_contours(frame, lane_mask)

        # ---------- 2. Object detection --------------------------------
        detections = yolo_model.detect(frame)
        distance_detector.updateDistance(detections, classes)

        for x1, y1, x2, y2, conf, cls_id in detections:
            label = f"{classes[cls_id]} {conf:.2f}"
            color = colors[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        distance_detector.DrawDetectedOnFrame(frame)

        # ---------- 3. FPS counter -------------------------------------
        frame_count += 1
        if frame_count >= 30:
            t_end       = time.time()
            fps         = frame_count / (t_end - t_start)
            frame_count = 0
            t_start     = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ---------- 4. Legend ------------------------------------------
        cv2.rectangle(frame, (10, orig_h - 50), (22, orig_h - 38),
                      DRIVE_COLOR, -1)
        cv2.putText(frame, "Drivable", (26, orig_h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, DRIVE_COLOR, 1)
        cv2.rectangle(frame, (10, orig_h - 30), (22, orig_h - 18),
                      LANE_COLOR, -1)
        cv2.putText(frame, "Lane", (26, orig_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, LANE_COLOR, 1)

        cv2.imshow("YOLOv8 + TwinLiteNet", frame)
        if cv2.waitKey(1) == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()