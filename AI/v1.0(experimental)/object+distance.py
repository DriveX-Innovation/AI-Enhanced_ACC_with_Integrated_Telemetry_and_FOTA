import cv2
import numpy as np
import onnxruntime as ort
import time
import random
import typing

# ==========================================================
# RECTINFO ADAPTER (COMPATIBLE WITH YOUR DISTANCE MODULE)
# ==========================================================
class RectInfo:
    def __init__(self, xmin, ymin, xmax, ymax, label, conf):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
        self.conf = conf

    def tolist(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]


# ==========================================================
# YOUR ORIGINAL DISTANCE MODULE (UNCHANGED)
# ==========================================================
from ObjectTracker.core import putText_shadow  # keep your function

class SingleCamDistanceMeasure(object):
    INCH = 0.39
    RefSizeDict = {
        "person": (160 * INCH, 50 * INCH),
        "bicycle": (98 * INCH, 65 * INCH),
        "motorbike": (100 * INCH, 100 * INCH),
        "car": (150 * INCH, 180 * INCH),
        "bus": (319 * INCH, 250 * INCH),
        "truck": (346 * INCH, 250 * INCH),
    }

    def __init__(self, object_list=["person", "bicycle", "car", "motorbike", "bus", "truck"]):
        self.object_list = object_list
        self.f = 100  # focal length (you should calibrate)
        self.distance_points = []

    def updateDistance(self, boxes: typing.List[RectInfo]) -> None:
        self.distance_points = []
        if len(boxes) != 0:
            for box in boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                label = box.label

                if label in self.object_list and ymax <= 650:
                    point_x = (xmax + xmin) // 2
                    point_y = ymax
                    try:
                        distance = (self.RefSizeDict[label][0] * self.f) / (ymax - ymin)
                        distance = distance / 12 * 0.3048
                        self.distance_points.append([point_x, point_y, distance])
                    except:
                        pass

    def DrawDetectedOnFrame(self, frame_show):
        if len(self.distance_points) != 0:
            for x, y, d in self.distance_points:
                cv2.circle(frame_show, (x, y), 4, (255, 255, 255), -1)
                text = f"{d:.2f} m" if d > 0 else "unknown m"
                fontScale = max(0.4, min(1, 1 / max(d, 0.1)))
                putText_shadow(frame_show, text, (x - 30, y - 5),
                               fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                               fontScale=fontScale,
                               color=(255, 255, 255),
                               thickness=1,
                               shadow_color=(150, 150, 150))


# ==========================================================
# YOLOv8 CONFIG
# ==========================================================
MODEL_PATH = "models/yolov8n.onnx"
CLASSES_PATH = "models/coco_label.txt"
CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT_SIZE = (480, 640)


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
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, iou_thres)
    return indices.flatten() if len(indices) else []


class YOLOv8_CPU:
    def __init__(self, model_path):
        print("Loading YOLOv8 ONNX (CPU)...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
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

            x1 = (x - w / 2 - pad_x) / r
            y1 = (y - h / 2 - pad_y) / r
            x2 = (x + w / 2 - pad_x) / r
            y2 = (y + h / 2 - pad_y) / r

            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(conf)
            class_ids.append(class_id)

        keep = nms(boxes, scores, IOU_THRES)

        results = []
        for i in keep:
            x, y, w, h = boxes[i]
            results.append((x, y, x + w, y + h, scores[i], class_ids[i]))
        return results

    def detect(self, frame):
        img, r, pad_x, pad_y = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: img})[0]
        return self.postprocess(output, r, pad_x, pad_y, frame.shape)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    classes = load_classes(CLASSES_PATH)
    colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(len(classes))}

    model = YOLOv8_CPU(MODEL_PATH)
    distanceDetector = SingleCamDistanceMeasure()

    cap = cv2.VideoCapture("./temp/test.mp4")

    fps = 0
    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = model.detect(frame)

        object_info = []
        for x1, y1, x2, y2, conf, cls_id in detections:
            label = classes[cls_id]
            object_info.append(RectInfo(x1, y1, x2, y2, label, conf))

            color = colors[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        distanceDetector.updateDistance(object_info)
        distanceDetector.DrawDetectedOnFrame(frame)

        frame_count += 1
        if frame_count >= 30:
            fps = frame_count / (time.time() - start)
            start = time.time()
            frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 + Distance", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
