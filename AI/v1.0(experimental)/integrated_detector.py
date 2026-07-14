import cv2
import numpy as np
import onnxruntime as ort
import time
import random
import os
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# ==========================
# CONFIG
# ==========================
MODEL_PATH      = "models/yolov8n.onnx"
CLASSES_PATH    = "models/coco_label.txt"
TWIN_MODEL_PATH = "models/TwinLiteNet.onnx"

CONF_THRES      = 0.25
IOU_THRES       = 0.45
YOLO_INPUT_SIZE = (480, 640)  # (H, W)
TWIN_INPUT_H    = 360
TWIN_INPUT_W    = 640
TWIN_SKIP       = 3           # run TwinLiteNet every N frames

LANE_COLOR      = (0,   255, 255)
DRIVE_COLOR     = (0,   180,   0)
DRIVE_ALPHA     = 0.35

CPU_COUNT       = os.cpu_count() or 4


# ==========================
# CPU SESSION FACTORY
# ==========================
def cpu_session(model_path: str, intra: int, inter: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode           = ort.ExecutionMode.ORT_PARALLEL
    so.intra_op_num_threads     = intra   # parallelism within each op
    so.inter_op_num_threads     = inter   # parallelism between independent ops
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


# ==========================
# UTILS
# ==========================
def load_classes(path):
    with open(path) as f:
        return [c.strip() for c in f]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r    = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh))
    pw, ph  = new_shape[1] - nw, new_shape[0] - nh
    t, b    = ph // 2, ph - ph // 2
    l, r2   = pw // 2, pw - pw // 2
    return (cv2.copyMakeBorder(resized, t, b, l, r2,
                               cv2.BORDER_CONSTANT, value=color),
            r, l, t)


def nms(boxes, scores, iou_thres):
    idx = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, iou_thres)
    return idx.flatten() if len(idx) else []


# ==========================
# YOLOv8
# ==========================
class YOLOv8_CPU:
    def __init__(self, model_path):
        print("[YOLOv8] Loading...")
        # Runs every frame → give it all cores
        self.session    = cpu_session(model_path,
                                      intra=CPU_COUNT,
                                      inter=CPU_COUNT // 2)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[YOLOv8] Ready  (intra={CPU_COUNT}, inter={CPU_COUNT//2})")

    def preprocess(self, frame):
        img, r, px, py = letterbox(frame, YOLO_INPUT_SIZE)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(img, (2, 0, 1)), 0), r, px, py

    def postprocess(self, out, r, px, py, orig_shape):
        out = out.squeeze(0).T
        boxes, scores, class_ids = [], [], []
        for det in out:
            x, y, w, h = det[:4]
            cls_id = int(np.argmax(det[4:]))
            conf   = float(det[4 + cls_id])
            if conf < CONF_THRES:
                continue
            x1 = int(max(0, min((x - w/2 - px) / r, orig_shape[1])))
            y1 = int(max(0, min((y - h/2 - py) / r, orig_shape[0])))
            x2 = int(max(0, min((x + w/2 - px) / r, orig_shape[1])))
            y2 = int(max(0, min((y + h/2 - py) / r, orig_shape[0])))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(cls_id)
        keep = nms(boxes, scores, IOU_THRES)
        return [(boxes[i][0], boxes[i][1],
                 boxes[i][0] + boxes[i][2],
                 boxes[i][1] + boxes[i][3],
                 scores[i], class_ids[i]) for i in keep]

    def detect(self, frame):
        img, r, px, py = self.preprocess(frame)
        out = self.session.run(None, {self.input_name: img})[0]
        return self.postprocess(out, r, px, py, frame.shape)


# ==========================
# TwinLiteNet
# ==========================
class TwinLiteNetDetector:
    _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path):
        print("[TwinLiteNet] Loading...")
        # Throttled model → half the cores is enough
        self.session      = cpu_session(model_path,
                                        intra=max(2, CPU_COUNT // 2),
                                        inter=2)
        self.input_name   = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        dummy        = np.zeros((1, 3, TWIN_INPUT_H, TWIN_INPUT_W), np.float32)
        outputs      = self.session.run(self.output_names, {self.input_name: dummy})
        self._layout = self._detect_layout(outputs)
        print(f"[TwinLiteNet] Ready  layout={self._layout}")

    @staticmethod
    def _detect_layout(outputs):
        if len(outputs) >= 2:  return "dual"
        sh = outputs[0].shape
        if len(sh) == 4:
            if sh[1] == 4:  return "merged4_cf"
            if sh[1] == 2:  return "single2_cf"
            if sh[3] == 4:  return "merged4_cl"
            if sh[3] == 2:  return "single2_cl"
        if len(sh) == 3:    return "single_mask"
        return "unknown"

    def preprocess(self, frame):
        img = cv2.resize(frame, (TWIN_INPUT_W, TWIN_INPUT_H))
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        return np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype(np.float32)

    def postprocess(self, outputs, orig_h, orig_w):
        def up(pred):
            return cv2.resize(pred.astype(np.uint8), (orig_w, orig_h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        def decode(logits):
            if logits is None:
                return np.zeros((orig_h, orig_w), bool)
            return up(np.argmax(logits.squeeze(0), axis=0))

        L = self._layout
        if L == "dual":
            return decode(outputs[0]), decode(outputs[1])
        if L == "merged4_cf":
            return decode(outputs[0][:, :2]), decode(outputs[0][:, 2:])
        if L == "merged4_cl":
            t = np.transpose(outputs[0], (0, 3, 1, 2))
            return decode(t[:, :2]), decode(t[:, 2:])
        if L in ("single2_cf", "single2_cl"):
            arr = outputs[0] if L == "single2_cf" \
                  else np.transpose(outputs[0], (0, 3, 1, 2))
            return np.zeros((orig_h, orig_w), bool), decode(arr)
        if L == "single_mask":
            return np.zeros((orig_h, orig_w), bool), up(outputs[0].squeeze(0))
        z = np.zeros((orig_h, orig_w), bool)
        return z, z

    def detect(self, frame):
        outputs = self.session.run(self.output_names,
                                   {self.input_name: self.preprocess(frame)})
        return self.postprocess(outputs, frame.shape[0], frame.shape[1])


# ==========================
# DRAW HELPERS
# ==========================
def apply_mask(frame, mask, color, alpha):
    if mask is None or not mask.any():
        return
    colored       = np.zeros_like(frame)
    colored[mask] = color
    cv2.addWeighted(colored, alpha, frame, 1.0, 0, dst=frame)


def draw_lane_contours(frame, mask, color=LANE_COLOR, thickness=3):
    if mask is None or not mask.any():
        return
    contours, _ = cv2.findContours((mask.astype(np.uint8)) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, thickness)


def draw_legend(frame):
    h = frame.shape[0]
    cv2.rectangle(frame, (10, h - 48), (20, h - 38), DRIVE_COLOR, -1)
    cv2.putText(frame, "Drivable", (24, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, DRIVE_COLOR, 1)
    cv2.rectangle(frame, (10, h - 28), (20, h - 18), LANE_COLOR, -1)
    cv2.putText(frame, "Lane", (24, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, LANE_COLOR, 1)


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    classes  = load_classes(CLASSES_PATH)
    colors   = {i: [random.randint(0, 255) for _ in range(3)]
                for i in range(len(classes))}

    yolo_model        = YOLOv8_CPU(MODEL_PATH)
    twin_model        = TwinLiteNetDetector(TWIN_MODEL_PATH)
    distance_detector = SingleCamDistanceMeasure()

    cap = cv2.VideoCapture("./temp/test.mp4")
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    print(f"\n[INFO] CPU cores available: {CPU_COUNT}")
    print(f"[INFO] TwinLiteNet runs every {TWIN_SKIP} frames\n")

    fps, frame_count = 0.0, 0
    t_start          = time.time()
    cached_drive     = None
    cached_lane      = None
    twin_frame       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Segmentation (throttled)
        if twin_frame % TWIN_SKIP == 0:
            cached_drive, cached_lane = twin_model.detect(frame)
        twin_frame += 1

        apply_mask(frame, cached_drive, DRIVE_COLOR, DRIVE_ALPHA)
        draw_lane_contours(frame, cached_lane)

        # 2. Object detection
        detections = yolo_model.detect(frame)
        distance_detector.updateDistance(detections, classes)

        for x1, y1, x2, y2, conf, cls_id in detections:
            color = colors[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{classes[cls_id]} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        distance_detector.DrawDetectedOnFrame(frame)

        # 3. FPS
        frame_count += 1
        if frame_count >= 30:
            fps         = frame_count / (time.time() - t_start)
            frame_count = 0
            t_start     = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        draw_legend(frame)

        cv2.imshow("YOLOv8 + TwinLiteNet", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
