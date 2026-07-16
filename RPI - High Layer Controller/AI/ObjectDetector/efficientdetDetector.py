import os
import cv2
import random
import numpy as np
from typing import *

try:
	import sys
	from utils import ObjectModelType, hex_to_rgb, NMS, Scaler
	from core import ObjectDetectBase, RectInfo
	sys.path.append("..")
	from coreEngine import TensorRTEngine, OnnxEngine
except:
	from .utils import ObjectModelType, hex_to_rgb, NMS, Scaler
	from .core import ObjectDetectBase, RectInfo
	from coreEngine import TensorRTEngine, OnnxEngine


class YoloDetector(ObjectDetectBase):
	_defaults = {
		"model_path": "./models/yolov8n.onnx",
		"model_type": ObjectModelType.YOLOV8,
		"classes_path": "./models/coco_label.txt",
		"box_score": 0.25,
		"box_nms_iou": 0.45
	}

	def __init__(self, logger=None, **kwargs):
		ObjectDetectBase.__init__(self, logger)
		self.__dict__.update(kwargs)

		self._initialize_class(self.classes_path)
		self._initialize_model(self.model_path)

	def _initialize_model(self, model_path: str) -> None:
		model_path = os.path.expanduser(model_path)

		if model_path.endswith(".trt"):
			self.engine = TensorRTEngine(model_path)
		else:
			self.engine = OnnxEngine(model_path)

		if self.logger:
			self.logger.info(f"YOLOv8 Detector || Framework: {self.engine.framework_type} || Providers: {self.engine.providers}")

		self.set_input_details(self.engine)
		self.set_output_details(self.engine)

	def _initialize_class(self, classes_path: str) -> None:
		classes_path = os.path.expanduser(classes_path)
		assert os.path.isfile(classes_path), Exception(f"{classes_path} not found.")

		with open(classes_path) as f:
			self.class_names = [c.strip() for c in f.readlines()]

		get_colors = list(map(lambda _: hex_to_rgb("#" + "%06x" % random.randint(0, 0xFFFFFF)), range(len(self.class_names))))
		self.colors_dict = dict(zip(self.class_names, get_colors))

	def __prepare_input(self, srcimg: cv2.Mat) -> Tuple[np.ndarray, Scaler]:
		scaler = Scaler(self.input_shapes[-2:], True)
		image = scaler.process_image(srcimg)

		blob = cv2.dnn.blobFromImage(
			image, 1 / 255.0,
			(image.shape[1], image.shape[0]),
			swapRB=True, crop=False
		).astype(self.input_types)

		return blob, scaler

	def __process_output(self, output: np.ndarray):
		_raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = [], [], [], []

		# YOLOv8 ONNX output: (1, num_classes+4, num_boxes)
		if len(output.shape) == 3:
			output = output.squeeze(0)

		output = output.T  # (num_boxes, num_classes+4)

		ih, iw = self.input_shapes[-2:]

		boxes = output[:, :4]
		scores = output[:, 4:]

		for i in range(len(boxes)):
			class_id = int(np.argmax(scores[i]))
			conf = float(scores[i][class_id])

			if conf > self.box_score:
				x, y, w, h = boxes[i]

				# scale to input size
				x *= iw
				y *= ih
				w *= iw
				h *= ih

				_raw_boxes.append(np.array([
					x - w / 2,
					y - h / 2,
					x + w / 2,
					y + h / 2
				]))

				_raw_class_ids.append(class_id)
				_raw_class_confs.append(conf)

		return _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss

	def get_nms_results(self, boxes, class_confs, class_ids, kpss):
		results = []
		if len(boxes) == 0:
			return results

		boxes = np.array(boxes)
		nms_results = NMS.fast_soft_nms(boxes, class_confs, self.box_nms_iou, dets_type="xyxy")

		for i in nms_results:
			label = self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else "unknown"
			conf = class_confs[i]
			bbox = boxes[i]

			results.append(RectInfo(*bbox, conf=conf, label=label, kpss=[]))

		return results

	def DetectFrame(self, srcimg: cv2.Mat) -> None:
		input_tensor, scaler = self.__prepare_input(srcimg)

		output = self.engine.engine_inference(input_tensor)[0]

		_raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = self.__process_output(output)

		transform_boxes = scaler.convert_boxes_coordinate(_raw_boxes)
		self._object_info = self.get_nms_results(transform_boxes, _raw_class_confs, _raw_class_ids, np.array([]))

	def DrawDetectedOnFrame(self, frame_show: cv2.Mat) -> None:
		tl = max(2, round(0.002 * (frame_show.shape[0] + frame_show.shape[1]) / 2))

		for info in self._object_info:
			xmin, ymin, xmax, ymax = map(int, info.tolist())
			label = info.label

			color = self.colors_dict.get(label, (0, 0, 0))
			cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax), color, 2)

			text = f"{label} {info.conf:.2f}"
			(tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
			cv2.rectangle(frame_show, (xmin, ymin - th - 5), (xmin + tw, ymin), color, -1)
			cv2.putText(frame_show, text, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


if __name__ == "__main__":
	import time

	capture = cv2.VideoCapture("./temp/test.mp4")

	config = {
		"model_path": "models/yolov8n.onnx",
		"model_type": ObjectModelType.YOLOV8,
		"classes_path": "models/coco_label.txt",
		"box_score": 0.25,
		"box_nms_iou": 0.45,
	}

	YoloDetector.set_defaults(config)
	network = YoloDetector()

	fps = 0
	frame_count = 0
	start = time.time()

	while True:
		ret, frame = capture.read()
		if not ret:
			print("End of stream.")
			break

		network.DetectFrame(frame)
		network.DrawDetectedOnFrame(frame)

		frame_count += 1
		if frame_count >= 30:
			end = time.time()
			fps = frame_count / (end - start)
			frame_count = 0
			start = time.time()

		cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

		cv2.imshow("output", frame)
		if cv2.waitKey(1) == 27:
			break
