import cv2
import numpy as np
import ncnn
from typing import Tuple

from .utils import LaneModelType, OffsetType, lane_colors
from .core import LaneDetectBase

def _softmax(x) :
    x = x - np.max(x, axis=-1, keepdims=True) 
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x, axis=-1, keepdims=True)

class ModelConfig():
    def __init__(self, model_type):
        if model_type == LaneModelType.UFLDV2_TUSIMPLE:
            self.init_tusimple_config()
        elif model_type == LaneModelType.UFLDV2_CURVELANES :
            self.init_curvelanes_config()
        else :
            self.init_culane_config()
        self.num_lanes = 4

    def init_tusimple_config(self):
        self.img_w = 288
        self.img_h = 800
        self.griding_num = 100
        self.crop_ratio = 0.8
        self.row_anchor = np.linspace(160,710, 56)/720
        self.col_anchor = np.linspace(0,1, 41)

    def init_curvelanes_config(self) :
        self.img_w = 1600
        self.img_h = 800
        self.griding_num = 200
        self.crop_ratio = 0.8
        self.row_anchor = np.linspace(0.4, 1, 72)
        self.col_anchor = np.linspace(0, 1, 81)
    
    def init_culane_config(self):
        self.img_w = 1600
        self.img_h = 320
        self.griding_num = 200
        self.crop_ratio = 0.6
        self.row_anchor = np.linspace(0.42, 1, 72)
        self.col_anchor = np.linspace(0,1, 81)

class UltrafastLaneDetectorV2(LaneDetectBase):
    _defaults = {
        "model_path": "models/culane_res18.onnx", 
        "model_type" : LaneModelType.UFLDV2_TUSIMPLE,
    }

    def __init__(self, model_path : str = None, model_type : LaneModelType = None, logger = None):
        LaneDetectBase.__init__(self, logger)
        if (None not in [model_path, model_type]) :
            self.model_path, self.model_type = model_path, model_type
        else:
            self.model_path = self._defaults["model_path"]
            self.model_type = self._defaults["model_type"]

        if ( self.model_type not in [LaneModelType.UFLDV2_TUSIMPLE, LaneModelType.UFLDV2_CULANE]) :
            raise Exception("UltrafastLaneDetectorV2 can't use %s type." % self.model_type.name)
        
        self.cfg = ModelConfig(self.model_type)
        self._initialize_model(self.model_path)
        
    def _initialize_model(self, model_path : str) -> None:
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        
        base_path = model_path.replace('.onnx', '').replace('.bin', '').replace('.param', '')
        self.net.load_param(base_path + ".param")
        self.net.load_model(base_path + ".bin")
        self.input_width = self.cfg.img_w
        self.input_height = self.cfg.img_h
        
    def __prepare_input(self, image : cv2) -> ncnn.Mat:
        self.h_ratio = image.shape[0] / self.cfg.img_h
        self.w_ratio = image.shape[1] / self.cfg.img_w
        self.img_height, self.img_width, self.img_channels = image.shape
        img_input = cv2.resize(image, (self.input_width, self.input_height))
        img_input = np.ascontiguousarray(img_input, dtype=np.uint8)
        
        mat_in = ncnn.Mat.from_pixels(img_input, ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.input_width, self.input_height)
        mean_vals = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        norm_vals = [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        return mat_in

    def __process_output(self, output, cfg : ModelConfig, local_width :int = 1) -> Tuple[np.ndarray, list]:
        original_image_width = self.img_width
        original_image_height = self.img_height
        output = {"loc_row" : output[0], 'loc_col' : output[1], "exist_row" : output[2], "exist_col" : output[3]}

        max_indices_row = output['loc_row'].argmax(1)
        valid_row = output['exist_row'].argmax(1)
        max_indices_col = output['loc_col'].argmax(1)
        valid_col = output['exist_col'].argmax(1)
        batch_size, num_grid_row, num_cls_row, num_lane_row = output['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = output['loc_col'].shape

        row_lane_idx = [1,2]
        col_lane_idx = [0,3]

        lanes_points = {"left-side" : [], "left-ego" : [] , "right-ego" : [], "right-side" : []}
        lanes_detected =  {"left-side" : False, "left-ego" : False , "right-ego" : False, "right-side" : False}
        
        for i in row_lane_idx:
            tmp = []
            if valid_row[0,:,i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0,k,i]:
                        all_ind = list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1))
                        out_tmp = ( _softmax(output['loc_row'][0,all_ind,k,i]) * list(map(float, all_ind))).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                        tmp.append((int(out_tmp), int(cfg.row_anchor[k] * original_image_height)))
                if (i == 1) :
                    lanes_points["left-ego"].extend(tmp)
                    if (len(tmp) > 2) : lanes_detected["left-ego"] = True
                else :
                    lanes_points["right-ego"].extend(tmp)
                    if (len(tmp) > 2) : lanes_detected["right-ego"] = True

        for i in col_lane_idx:
            tmp = []
            if valid_col[0,:,i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0,k,i]:
                        all_ind = list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1))
                        out_tmp = ( _softmax(output['loc_col'][0,all_ind,k,i]) * list(map(float, all_ind))).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                        tmp.append((int(cfg.col_anchor[k] * original_image_width), int(out_tmp)))
                if (i == 0) :
                    lanes_points["left-side" ].extend(tmp)
                    if (len(tmp) > 2) : lanes_detected["left-side"] = True
                else :
                    lanes_points["right-side"].extend(tmp)
                    if (len(tmp) > 2) : lanes_detected["right-side"] = True
        return np.array(list(lanes_points.values()), dtype="object"), list(lanes_detected.values())

    def DetectFrame(self, image : cv2, adjust_lanes : bool = True) -> None:
        mat_in = self.__prepare_input(image)
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        
        outputs = []
        blob_names = ["out0", "out1", "out2", "out3"]
        for name in blob_names:
            ret, mat_out = ex.extract(name)
            outputs.append(np.array(mat_out))

        self.lane_info.lanes_points, self.lane_info.lanes_status = self.__process_output(outputs, self.cfg)
        self.adjust_lanes = adjust_lanes
        self._LaneDetectBase__update_lanes_status(self.lane_info.lanes_status)
        self._LaneDetectBase__update_lanes_area(self.lane_info.lanes_points, self.img_height)

    def DrawDetectedOnFrame(self, image : cv2, type : OffsetType = OffsetType.UNKNOWN, alpha: float = 0.3) -> None:
        overlay = image.copy()
        for lane_num,lane_points in enumerate(self.lane_info.lanes_points):
            if ( lane_num==1 and type == OffsetType.RIGHT) : color = (0, 0, 255)
            elif (lane_num==2 and type == OffsetType.LEFT) : color = (0, 0, 255)
            else : color = lane_colors[lane_num]
            for lane_point in lane_points:
                cv2.circle(overlay, (lane_point[0],lane_point[1]), 3, color, thickness=-1)
        image[:] = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def DrawAreaOnFrame(self, image : cv2, color : tuple = (255,191,0), alpha: float = 0.85) -> None :
        H, W, _ = image.shape
        if(self.lane_info.area_status):
            lane_segment_img = image.copy()
            cv2.fillPoly(lane_segment_img, pts = [self.lane_info.area_points], color =color)
            image[:H,:W,:] = cv2.addWeighted(image, alpha, lane_segment_img, 1-alpha, 0)
