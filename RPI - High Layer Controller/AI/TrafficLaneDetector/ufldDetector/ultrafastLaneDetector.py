import cv2
import numpy as np
from typing import Tuple
from openvino import Core

from .utils import LaneModelType, OffsetType, lane_colors
from .core import LaneDetectBase

def _fast_softmax(x, axis=0):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class ModelConfig():
    def __init__(self, model_type):
        if model_type == LaneModelType.UFLD_TUSIMPLE:
            self.init_tusimple_config()
        else:
            self.init_culane_config()
        self.num_lanes = 4

    def init_tusimple_config(self):
        self.img_w = 800
        self.img_h = 288
        self.griding_num = 100
        self.cls_num_per_lane = 56
        self.row_anchor = np.linspace(64, 284, self.cls_num_per_lane)

    def init_culane_config(self):
        self.img_w = 1640
        self.img_h = 590
        self.griding_num = 200
        self.cls_num_per_lane = 18
        self.row_anchor = [round(v) for v in np.linspace(121, 287, self.cls_num_per_lane)]

class UltrafastLaneDetector(LaneDetectBase):
    _defaults = {
        "model_path": "models/ultra_falst_lane_detection_tusimple_288x800.xml",
        "model_type": LaneModelType.UFLD_TUSIMPLE,
    }

    def __init__(self, model_path: str = None, model_type: LaneModelType = None, logger=None):
        LaneDetectBase.__init__(self, logger)
        if None not in [model_path, model_type]:
            self.model_path, self.model_type = model_path, model_type
        else:
            self.model_path = self._defaults["model_path"]
            self.model_type = self._defaults["model_type"]

        if self.model_type not in [LaneModelType.UFLD_TUSIMPLE, LaneModelType.UFLD_CULANE]:
            raise Exception("UltrafastLaneDetector can't use %s type." % self.model_type.name)

        self.cfg = ModelConfig(self.model_type)
        self._initialize_model(self.model_path)

    def _initialize_model(self, model_path: str) -> None:
        import os
        base_dir = os.path.dirname(os.path.abspath(model_path))
        ov_xml = os.path.join(
            base_dir, "openvino",
            "ultra_falst_lane_detection_tusimple_288x800.xml"
        )

        print(f"  Loading OpenVINO model: {ov_xml}")
        ie = Core()
        model = ie.read_model(model=ov_xml)
        self.compiled_model = ie.compile_model(model=model, device_name="CPU")
        self.infer_request  = self.compiled_model.create_infer_request()
        self.input_layer    = self.compiled_model.input(0)
        self.output_layer   = self.compiled_model.output(0)

        print(f"  Input shape:  {self.input_layer.shape}")
        print(f"  Output shape: {self.output_layer.shape}")

        self.input_width  = self.cfg.img_w   # 800
        self.input_height = self.cfg.img_h   # 288
        print("  OpenVINO model loaded OK")

    def __prepare_input(self, image) -> np.ndarray:
        self.h_ratio = image.shape[0] / self.cfg.img_h
        self.w_ratio = image.shape[1] / self.cfg.img_w
        self.img_height, self.img_width, self.img_channels = image.shape

        img_input = cv2.resize(image, (self.input_width, self.input_height))
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_input = (img_input - mean) / std

        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        return np.ascontiguousarray(img_input, dtype=np.float32)

    def __process_output(self, output, cfg: ModelConfig) -> Tuple[np.ndarray, list]:
        processed_output = np.squeeze(output[0])
        if processed_output.ndim == 4:
            processed_output = np.squeeze(processed_output)

        if processed_output.shape == (4, 18, 201) or processed_output.shape == (4, 56, 101):
            processed_output = processed_output.transpose(2, 1, 0)

        processed_output = processed_output[:, ::-1, :]
        prob = _fast_softmax(processed_output[:-1, :, :], axis=0)
        idx  = np.arange(cfg.griding_num) + 1
        idx  = idx.reshape(-1, 1, 1)
        loc  = np.sum(prob * idx, axis=0)
        processed_output = np.argmax(processed_output, axis=0)
        loc[processed_output == cfg.griding_num] = 0
        processed_output = loc

        col_sample   = np.linspace(0, self.input_width - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        lanes_points   = []
        lanes_detected = []

        max_lanes = processed_output.shape[1]
        for lane_num in range(max_lanes):
            lane_points = []
            if np.sum(processed_output[:, lane_num] != 0) > 2:
                lanes_detected.append(True)
                for point_num in range(processed_output.shape[0]):
                    if processed_output[point_num, lane_num] > 0:
                        lane_point = [
                            processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / self.input_width - 1,
                            cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane - 1 - point_num] / self.input_height) - 1
                        ]
                        lane_points.append([
                            int(lane_point[0] * self.w_ratio),
                            int(lane_point[1] * self.h_ratio)
                        ])
            else:
                lanes_detected.append(False)
            lanes_points.append(lane_points)

        return np.array(lanes_points, dtype=object), np.array(lanes_detected, dtype=object)

    def DetectFrame(self, image, adjust_lanes: bool = True) -> None:
        img_input = self.__prepare_input(image)
        self.infer_request.infer({self.input_layer: img_input})
        output = [self.infer_request.get_output_tensor(0).data.copy()]

        self.lane_info.lanes_points, self.lane_info.lanes_status = \
            self.__process_output(output, self.cfg)
        self.adjust_lanes = adjust_lanes
        self._LaneDetectBase__update_lanes_status(self.lane_info.lanes_status)
        self._LaneDetectBase__update_lanes_area(
            self.lane_info.lanes_points, self.img_height
        )

    def DrawDetectedOnFrame(self, image, type: OffsetType = OffsetType.UNKNOWN, alpha: float = 0.3) -> None:
        overlay = image.copy()
        for lane_num, lane_points in enumerate(self.lane_info.lanes_points):
            if   lane_num == 1 and type == OffsetType.RIGHT: color = (0, 0, 255)
            elif lane_num == 2 and type == OffsetType.LEFT:  color = (0, 0, 255)
            else:                                            color = lane_colors[lane_num]
            for lane_point in lane_points:
                cv2.circle(overlay, (lane_point[0], lane_point[1]), 3, color, thickness=-1)
        image[:] = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def DrawAreaOnFrame(self, image, color: tuple = (255, 191, 0), alpha: float = 0.85) -> None:
        H, W, _ = image.shape
        if self.lane_info.area_status:
            lane_segment_img = image.copy()
            cv2.fillPoly(lane_segment_img, pts=[self.lane_info.area_points], color=color)
            image[:H, :W, :] = cv2.addWeighted(image, alpha, lane_segment_img, 1 - alpha, 0)
