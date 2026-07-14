#!/usr/bin/env python3
"""
MetaDrive ADAS Bridge: YOLOv8n + YOLOPv2 (Shadow Mode)
Vehicle is driven by MetaDrive's built-in ExpertPolicy.
CV Pipeline calculates and displays Steering/Curvature but DOES NOT control the car.
"""

import sys, os, time, threading
import cv2
import numpy as np
import ncnn
from ultralytics import YOLO

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.expert_policy import ExpertPolicy

# ==========================
# CONFIGURATION
# ==========================
CAM_WIDTH, CAM_HEIGHT = 640, 480

YOLOV8_MODEL_PATH = "D:\\graduation-project-related\\Vehicle-CV-ADAS-master\\YOLOPv2-ncnn-main\\models\\yolov8n_ncnn_model"
YOLOPV2_MODEL_DIR = "D:\\graduation-project-related\\Vehicle-CV-ADAS-master\\YOLOPv2-ncnn-main\\models"
INFER_SIZE = 320
CONF_THRESH = 0.30
VEHICLE_CLASSES = [2, 3, 5, 7]
MAX_STRIDE = 64

ANCHORS = {
    8:  np.array([12, 16, 19, 36, 40, 28],       dtype=np.float32),
    16: np.array([36, 75, 76, 55, 72, 146],      dtype=np.float32),
    32: np.array([142, 110, 192, 243, 459, 401], dtype=np.float32),
}

CLASS_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

WRITE_SHARED_FILE = True
SHARED_FILE_PATH  = "D:\\graduation-project-related\\Vehicle-CV-ADAS-master\\YOLOPv2-ncnn-main\\shared\\adas_output.txt"

DISPLAY_WINDOW_NAME = "ADAS Inference View (Shadow Mode)"

# ==========================
# YOLOPv2 helpers
# ==========================

def letterbox(img, target_size):
    h, w = img.shape[:2]
    if w > h:
        scale = target_size / w
        nw, nh = target_size, int(h * scale)
    else:
        scale = target_size / h
        nh, nw = target_size, int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    wpad = (nw + MAX_STRIDE - 1) // MAX_STRIDE * MAX_STRIDE - nw
    hpad = (nh + MAX_STRIDE - 1) // MAX_STRIDE * MAX_STRIDE - nh
    padded = cv2.copyMakeBorder(
        resized, hpad // 2, hpad - hpad // 2, wpad // 2, wpad - wpad // 2,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, wpad, hpad


def generate_proposals_fast(anchors_flat, stride, pad_w, pad_h, feat, conf_thresh):
    num_anchors = len(anchors_flat) // 2
    num_grid    = feat.shape[1]
    if pad_w > pad_h:
        ngx = pad_w // stride
        ngy = num_grid // ngx
    else:
        ngy = pad_h // stride
        ngx = num_grid // ngy

    gy, gx = np.meshgrid(np.arange(ngy), np.arange(ngx), indexing='ij')
    gx = gx.flatten(); gy = gy.flatten()

    objects = []
    for q in range(num_anchors):
        aw = anchors_flat[q * 2]; ah = anchors_flat[q * 2 + 1]
        row = feat[q]
        box_conf   = 1 / (1 + np.exp(-row[:, 4]))
        cls_scores = 1 / (1 + np.exp(-row[:, 5:]))
        cls_idx    = np.argmax(cls_scores, axis=1)
        cls_conf   = cls_scores[np.arange(len(cls_scores)), cls_idx]
        confidence = box_conf * cls_conf
        mask = confidence >= conf_thresh
        if not np.any(mask):
            continue
        r  = row[mask]; cx = gx[mask].astype(np.float32); cy = gy[mask].astype(np.float32)
        dx = 1 / (1 + np.exp(-r[:, 0])); dy = 1 / (1 + np.exp(-r[:, 1]))
        dw = 1 / (1 + np.exp(-r[:, 2])); dh = 1 / (1 + np.exp(-r[:, 3]))
        pb_cx = (dx * 2 - 0.5 + cx) * stride
        pb_cy = (dy * 2 - 0.5 + cy) * stride
        pb_w  = (dw * 2) ** 2 * aw
        pb_h  = (dh * 2) ** 2 * ah
        x0 = pb_cx - pb_w * 0.5; y0 = pb_cy - pb_h * 0.5
        x1 = pb_cx + pb_w * 0.5; y1 = pb_cy + pb_h * 0.5
        conf_m = confidence[mask]; idx_m = cls_idx[mask]
        for i in range(len(x0)):
            objects.append({'x': float(x0[i]), 'y': float(y0[i]),
                             'w': float(x1[i] - x0[i]), 'h': float(y1[i] - y0[i]),
                             'label': int(idx_m[i]), 'prob': float(conf_m[i])})
    return objects


def nms(objects, nms_threshold=0.45):
    if not objects:
        return []
    objects = sorted(objects, key=lambda o: o['prob'], reverse=True)
    areas  = [o['w'] * o['h'] for o in objects]
    picked = []
    for i, a in enumerate(objects):
        keep = True
        for j in picked:
            b = objects[j]
            ix0 = max(a['x'], b['x']); iy0 = max(a['y'], b['y'])
            ix1 = min(a['x'] + a['w'], b['x'] + b['w'])
            iy1 = min(a['y'] + a['h'], b['y'] + b['h'])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter / (areas[i] + areas[j] - inter) > nms_threshold:
                keep = False; break
        if keep:
            picked.append(i)
    return [objects[i] for i in picked]


def yolopv2_infer(net, bgr, target_size, conf_thresh):
    img_h, img_w = bgr.shape[:2]
    padded, scale, wpad, hpad = letterbox(bgr, target_size)
    pad_w, pad_h = padded.shape[1], padded.shape[0]

    mat_in = ncnn.Mat.from_pixels(
        padded, ncnn.Mat.PixelType.PIXEL_BGR2RGB, pad_w, pad_h)
    mat_in.substract_mean_normalize([0, 0, 0], [1 / 255., 1 / 255., 1 / 255.])

    ex = net.create_extractor()
    ex.input("images", mat_in)

    proposals = []
    for stride, anch in ANCHORS.items():
        blob_name = {8: "det0", 16: "det1", 32: "det2"}[stride]
        _, out = ex.extract(blob_name)
        arr = np.array(out)
        proposals.extend(
            generate_proposals_fast(anch, stride, pad_w, pad_h, arr, conf_thresh))
    objects = nms(proposals)
    for obj in objects:
        x0 = (obj['x'] - wpad / 2) / scale; y0 = (obj['y'] - hpad / 2) / scale
        x1 = (obj['x'] + obj['w'] - wpad / 2) / scale
        y1 = (obj['y'] + obj['h'] - hpad / 2) / scale
        obj['x'] = float(np.clip(x0, 0, img_w - 1))
        obj['y'] = float(np.clip(y0, 0, img_h - 1))
        obj['w'] = float(np.clip(x1, 0, img_w - 1)) - obj['x']
        obj['h'] = float(np.clip(y1, 0, img_h - 1)) - obj['y']

    t = hpad // 2; l = wpad // 2

    _, ll_out = ex.extract("769")
    ll_arr = np.array(ll_out)
    ll_c   = ll_arr[:, t:ll_arr.shape[1] - (hpad - t), l:ll_arr.shape[2] - (wpad - l)]
    ll_r   = np.stack([cv2.resize(ll_c[c], (img_w, img_h),
                                   interpolation=cv2.INTER_LINEAR)
                        for c in range(ll_c.shape[0])])

    return objects, ll_r


# ==========================
# Lane Curvature Analysis (Calculates for Display Only)
# ==========================

class LaneCurvatureAnalyzer:
    WARN_RADIUS_M    = 30.0
    PIXELS_PER_METER = 10.0
    SMOOTH_ALPHA     = 0.20
    
    KP = 0.0035  
    KD = 0.0050  
    LOOKAHEAD  = 0.60 

    def __init__(self):
        self.smooth_curvature = 0.0
        self.smooth_steer_val = 0.0 
        self.direction        = 0
        self.last_error_x     = 0.0

    def update(self, lane_binary: np.ndarray) -> dict:
        img_h, img_w = lane_binary.shape
        cx = img_w // 2

        mid_xs, mid_ys = [], []
        for y in range(img_h - 1, int(img_h * 0.40), -10):
            row = lane_binary[y]
            xs  = np.where(row)[0]
            if len(xs) == 0:
                continue
            left  = xs[xs < cx]
            right = xs[xs > cx]
            if len(left) == 0 or len(right) == 0:
                continue
            lx = left[-1]
            rx = right[0]
            mid_xs.append((lx + rx) / 2.0)
            mid_ys.append(float(y))

        if len(mid_xs) < 6:
            return self._no_data()

        ys = np.array(mid_ys)
        xs = np.array(mid_xs)

        try:
            coeffs = np.polyfit(ys, xs, 2)
        except np.linalg.LinAlgError:
            return self._no_data()

        a, b, c = coeffs

        # Two-Point Error Calculation (For smooth visual output)
        bottom_y = float(img_h - 1)
        current_x = a * (bottom_y**2) + b * bottom_y + c
        cte = current_x - cx

        lookahead_y = img_h * self.LOOKAHEAD 
        target_x = a * (lookahead_y**2) + b * lookahead_y + c
        lookahead_error = target_x - cx

        error_x = (0.4 * cte) + (0.6 * lookahead_error)
        
        p_term = self.KP * error_x
        d_term = self.KD * (error_x - self.last_error_x)
        raw_steer = p_term + d_term
        self.last_error_x = error_x
        
        steer_val = np.clip(raw_steer, -1.0, 1.0)
        
        self.smooth_steer_val = (self.SMOOTH_ALPHA * steer_val + 
                                 (1 - self.SMOOTH_ALPHA) * self.smooth_steer_val)

        eval_y    = bottom_y       
        dx_dy     = 2 * a * eval_y + b
        d2x_dy2   = 2 * a
        curvature = abs(d2x_dy2) / ((1 + dx_dy ** 2) ** 1.5 + 1e-9)

        radius_m = (1.0 / (curvature * self.PIXELS_PER_METER + 1e-9))
        radius_m = min(radius_m, 9999.0)
        
        raw_dir = -1 if a > 0 else (1 if a < 0 else 0)
        self.smooth_curvature = (self.SMOOTH_ALPHA * curvature + 
                                  (1 - self.SMOOTH_ALPHA) * self.smooth_curvature)
        self.direction = raw_dir
        warn = radius_m < self.WARN_RADIUS_M

        return {
            'valid':     True,
            'radius_m':  radius_m,
            'curvature': self.smooth_curvature,
            'steer_val': self.smooth_steer_val,
            'direction': raw_dir,
            'warn':      warn,
            'coeffs':    coeffs,
            'mid_pts':   list(zip(mid_xs, mid_ys)),
        }

    def _no_data(self):
        self.smooth_steer_val *= 0.8
        return {'valid': False, 'radius_m': 9999, 'curvature': 0, 
                'steer_val': self.smooth_steer_val, 'direction': 0, 'warn': False, 
                'coeffs': None, 'mid_pts': []}


# ==========================
# Shared state with Synchronous Locks
# ==========================

class SharedState:
    def __init__(self):
        self._frame_lock = threading.Lock()
        self.frame_id = 0
        self.frame = None

        self._decision_lock = threading.Lock()
        self.decision = {
            'cut_in': 0, 'lane_curve': 0, 'steer_val': 0.0,
            'radius_m': 9999.0, 'detections': [], 'lane_mask': None,
            'infer_fps': 0.0, 'frame_id': -1,
        }

        self.stop_event = threading.Event()
        
        # Lockstep synchronization events
        self.frame_ready = threading.Event()
        self.decision_ready = threading.Event()

    def set_frame(self, frame):
        with self._frame_lock:
            self.frame = frame
            self.frame_id += 1

    def get_frame(self):
        with self._frame_lock:
            return self.frame, self.frame_id

    def set_decision(self, **kwargs):
        with self._decision_lock:
            self.decision.update(kwargs)

    def get_decision(self):
        with self._decision_lock:
            return dict(self.decision)


# ==========================
# Synchronous Inference thread
# ==========================

def inference_worker(shared: SharedState):
    print("[INFER] Loading YOLOv8n NCNN model...")
    yolo = YOLO(YOLOV8_MODEL_PATH, task='detect')

    print("[INFER] Loading YOLOPv2 NCNN model...")
    pv2_net = ncnn.Net()
    pv2_net.opt.num_threads        = 4
    pv2_net.opt.use_fp16_storage   = True
    pv2_net.opt.use_packing_layout = True
    pv2_net.load_param(os.path.join(YOLOPV2_MODEL_DIR, 'yolopv2.param'))
    pv2_net.load_model(os.path.join(YOLOPV2_MODEL_DIR, 'yolopv2.bin'))

    dummy = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
    _ = list(yolo.predict(dummy, imgsz=640, verbose=False, classes=VEHICLE_CLASSES))
    print("[INFER] Models warmed up. Waiting for simulation frames...\n")

    curvature_analyzer = LaneCurvatureAnalyzer()

    while not shared.stop_event.is_set():
        if not shared.frame_ready.wait(timeout=1.0):
            continue
        shared.frame_ready.clear()
        
        frame, frame_id = shared.get_frame()
        if frame is None:
            shared.decision_ready.set()
            continue

        t0 = time.time()
        try:
            yolo_results = list(yolo.predict(
                frame, imgsz=640, conf=0.25, verbose=False,
                stream=True, classes=VEHICLE_CLASSES))

            detections = []
            if yolo_results:
                result = yolo_results[0]
                boxes = result.boxes
                if len(boxes) > 0:
                    xyxy  = boxes.xyxy.numpy().astype(int)
                    confs = boxes.conf.numpy()
                    clss  = boxes.cls.numpy().astype(int)
                    detections = [
                        (xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], confs[i], clss[i])
                        for i in range(len(boxes))
                    ]

            _, ll = yolopv2_infer(pv2_net, frame, INFER_SIZE, CONF_THRESH)

            cut_in = 0
            lane_curve = 0
            steer_val = 0.0
            radius_m = 9999.0
            lane_binary = None

            if ll is not None:
                lane_binary = np.round(ll[0]) == 1
                img_h, img_w = lane_binary.shape
                center_x = img_w // 2

                curv_info = curvature_analyzer.update(lane_binary)
                steer_val = curv_info['steer_val']
                radius_m  = curv_info['radius_m']
                lane_curve = 1 if (curv_info['valid'] and radius_m < 200) else 0

                pts_left, pts_right = [], []
                last_l = int(center_x * 0.2)
                last_r = int(center_x * 1.8)

                for y in range(img_h - 1, int(img_h * 0.45), -15):
                    row = lane_binary[y]
                    lane_xs = np.where(row)[0]
                    curr_l, curr_r = last_l, last_r
                    if len(lane_xs) > 0:
                        left_c  = lane_xs[lane_xs < center_x]
                        right_c = lane_xs[lane_xs > center_x]
                        if len(left_c) > 0:
                            curr_l = left_c[-1]; last_l = curr_l
                        if len(right_c) > 0:
                            curr_r = right_c[0]; last_r = curr_r
                    pts_left.append([curr_l, y])
                    pts_right.append([curr_r, y])

                ego_poly = np.array(pts_left + pts_right[::-1], np.int32)

                if len(ego_poly) > 0:
                    for det in detections:
                        x1, y1, x2, y2, conf, cls = det
                        bx = int((x1 + x2) / 2); by = int(y2)
                        if cv2.pointPolygonTest(ego_poly, (bx, by), False) >= 0:
                            cut_in = 1

            infer_fps = 1.0 / (time.time() - t0 + 1e-9)

            shared.set_decision(
                cut_in=cut_in, lane_curve=lane_curve, steer_val=float(steer_val),
                radius_m=float(radius_m), detections=detections,
                lane_mask=lane_binary, infer_fps=infer_fps, frame_id=frame_id,
            )

            if WRITE_SHARED_FILE:
                try:
                    with open(SHARED_FILE_PATH, 'w') as f:
                        f.write(f"{cut_in},{lane_curve}")
                except OSError:
                    pass

        except Exception as e:
            print(f"[INFER][ERROR] {e}")
            
        finally:
            shared.decision_ready.set()


# ==========================
# Display thread
# ==========================

def display_worker(shared: SharedState):
    cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
    last_shown_frame_id = -1

    while not shared.stop_event.is_set():
        frame, frame_id = shared.get_frame()
        if frame is None or frame_id == last_shown_frame_id:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                shared.stop_event.set()
                break
            time.sleep(0.01)
            continue
        last_shown_frame_id = frame_id

        vis = frame.copy()
        decision = shared.get_decision()

        lane_mask = decision.get('lane_mask')
        if lane_mask is not None and lane_mask.shape[:2] == vis.shape[:2]:
            overlay = vis.copy()
            overlay[lane_mask] = (0, 255, 0)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

        for det in decision.get('detections', []):
            x1, y1, x2, y2, conf, cls = det
            label = CLASS_NAMES.get(int(cls), str(cls))
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
            cv2.putText(vis, f"{label} {conf:.2f}", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        cut_in_color = (0, 0, 255) if decision.get('cut_in') else (0, 200, 0)
        curve_color  = (0, 165, 255) if decision.get('lane_curve') else (0, 200, 0)

        cv2.putText(vis, f"cut_in={decision.get('cut_in', 0)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cut_in_color, 2)
        
        cv2.putText(vis, f"lane_curve={decision.get('lane_curve', 0)} (R={decision.get('radius_m', 9999):.1f}m)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, curve_color, 2)
        
        cv2.putText(vis, f"steer_val={decision.get('steer_val', 0.0):.3f} ", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(vis, f"infer_fps={decision.get('infer_fps', 0.0):.1f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(DISPLAY_WINDOW_NAME, vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared.stop_event.set()
            break

    cv2.destroyWindow(DISPLAY_WINDOW_NAME)


# ==========================
# MetaDrive setup / main loop
# ==========================

def make_env():
    config = dict(
        use_render=False,
        image_observation=True,
        sensors={"rgb_camera": (RGBCamera, CAM_WIDTH, CAM_HEIGHT)},
        vehicle_config={"image_source": "rgb_camera"},
        stack_size=1,
        traffic_density=0.1,
        map=4,
        manual_control=False,
        log_level=50,
        
        # --- AUTOPILOT ENABLED ---
        agent_policy=ExpertPolicy,  
        
        out_of_road_penalty=0.0,
        crash_vehicle_done=False,
        crash_object_done=False,
        out_of_route_done=False,
        out_of_road_done=False,
        on_continuous_line_done=False,
        on_broken_line_done=False,
    )
    return MetaDriveEnv(config)

def obs_image_to_bgr(obs):
    rgb_float = obs["image"][..., -1]
    rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    return bgr

def main():
    shared = SharedState()

    print("[METADRIVE] Creating environment...")
    env = make_env()
    obs, info = env.reset()
    print("[METADRIVE] Environment ready. Autopilot engaged.")

    infer_thread = threading.Thread(target=inference_worker, args=(shared,), daemon=True)
    infer_thread.start()

    disp_thread = threading.Thread(target=display_worker, args=(shared,), daemon=True)
    disp_thread.start()

    max_steps_per_episode = 6000
    steps_in_episode = 0

    print("=" * 60)
    print("ADAS + METADRIVE RUNNING (SHADOW MODE)")
    print("Autopilot drives the car. CV calculates & displays Steer/Curvature in the background.")
    print("Press 'q' in the display window (or Ctrl+C here) to stop.")
    print("=" * 60)

    shared.set_frame(obs_image_to_bgr(obs))
    shared.frame_ready.set()

    try:
        while not shared.stop_event.is_set():
            shared.decision_ready.wait(timeout=2.0)
            shared.decision_ready.clear()

            # Pass dummy values [0, 0] because ExpertPolicy ignores them anyway
            obs, reward, terminated, truncated, info = env.step([0, 0])
            steps_in_episode += 1

            if truncated or steps_in_episode >= max_steps_per_episode:
                print(f"[MAIN] Episode reset (trunc={truncated}, steps={steps_in_episode})")
                obs, info = env.reset()
                steps_in_episode = 0

            shared.set_frame(obs_image_to_bgr(obs))
            shared.frame_ready.set()

    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")

    finally:
        shared.stop_event.set()
        shared.frame_ready.set()
        shared.decision_ready.set()
        time.sleep(0.5)
        env.close()
        print("=" * 60)
        print("ADAS + MetaDrive Stopped")
        print("=" * 60)


if __name__ == "__main__":
    main()