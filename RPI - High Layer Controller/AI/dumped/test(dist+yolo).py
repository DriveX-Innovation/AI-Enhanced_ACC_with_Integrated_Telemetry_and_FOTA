import cv2
import numpy as np
import onnxruntime as ort
import time
import random
from typing import List, Tuple
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
# ==========================
# CONFIG
# ==========================
MODEL_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n.onnx"
CLASSES_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/coco_label.txt"
CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT_SIZE = (480, 640)  # YOLOv8 default


# ==========================
# UTILS
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
	def __init__(self, model_path):
		print(" Loading YOLOv8 ONNX (CPU only)...")

		so = ort.SessionOptions()
		so.intra_op_num_threads = 4
		so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

		self.session = ort.InferenceSession(
			model_path,
			sess_options=so,
			providers=["CPUExecutionProvider"]  # FORCE CPU
		)

		self.input_name = self.session.get_inputs()[0].name

	def preprocess(self, frame):
		img, r, pad_x, pad_y = letterbox(frame, INPUT_SIZE)
		img = img[:, :, ::-1].astype(np.float32) / 255.0
		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)
		return img, r, pad_x, pad_y

	def postprocess(self, output, r, pad_x, pad_y, orig_shape):
		output = output.squeeze(0).T  # (num_boxes, 84)

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

			x1 = max(0, min(x1, orig_shape[1]))
			y1 = max(0, min(y1, orig_shape[0]))
			x2 = max(0, min(x2, orig_shape[1]))
			y2 = max(0, min(y2, orig_shape[0]))

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


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
	classes = load_classes(CLASSES_PATH)
	colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(len(classes))}

	model = YOLOv8_CPU(MODEL_PATH)

	cap = cv2.VideoCapture("./temp/test.mp4")

	fps = 0
	frame_count = 0
	start = time.time()
	distanceDetector = SingleCamDistanceMeasure()
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		detections = model.detect(frame)
		distanceDetector.updateDistance(detections, classes)
		for x1, y1, x2, y2, conf, cls_id in detections:
			label = f"{classes[cls_id]} {conf:.2f}"
			color = colors[cls_id]

			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
			cv2.putText(frame, label, (x1, y1 - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



		distanceDetector.DrawDetectedOnFrame(frame)

		frame_count += 1
		if frame_count >= 30:
			end = time.time()
			fps = frame_count / (end - start)
			frame_count = 0
			start = time.time()

		cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow("YOLOv8 CPU", frame)
		if cv2.waitKey(1) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
