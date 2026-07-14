#!/usr/bin/env python3
"""
CARLA ADAS Bridge: YOLOv8n + YOLOPv2 + Lane Curvature -> CARLA vehicle control

ARCHITECTURE (why it won't tank your CARLA framerate):
  - CARLA sim thread (main loop): ticks the world at a fixed rate, applies
    whatever driving decision is freshest, and moves on. It NEVER waits on
    the AI.
  - Camera callback (fired by CARLA per tick): does the absolute minimum -
    convert the raw buffer to a numpy array and drop it in a shared slot.
    No inference happens here.
  - Inference thread: runs completely independently, at whatever pace the
    models allow (probably slower than sim tick rate). It always grabs the
    LATEST frame available, never a queue of stale ones, and writes the
    latest driving decision (steer/throttle/brake + cut_in/lane_curve) to
    a shared slot.
  - Display thread: also independent, just reads the latest frame + latest
    inference results and draws an overlay window. Never blocks sim or AI.

This is the same pattern as your original ThreadedWebcamStream + infer_thread
split, just with a CARLA sensor instead of a USB webcam, and CARLA vehicle
control instead of UART/GPIO.
"""

import sys, os, time, threading
import cv2
import numpy as np
import ncnn
from ultralytics import YOLO

import carla

# ==========================
# CONFIGURATION
# ==========================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

SYNC_MODE = True
FIXED_DELTA_SECONDS = 0.05          # 20 Hz simulation tick
VEHICLE_BLUEPRINT = "vehicle.tesla.model3"

CAM_WIDTH, CAM_HEIGHT = 640, 480
CAM_FOV = 90
CAM_LOCATION = carla.Location(x=1.6, z=1.7)   # roughly windshield/dash mount

YOLOV8_MODEL_PATH = "D:\graduation-project-related\Vehicle-CV-ADAS-master\YOLOPv2-ncnn-main\models\yolov8n_ncnn_model"
YOLOPV2_MODEL_DIR = "..\models"
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

# Driving policy tuning
BASE_THROTTLE       = 0.45
CURVE_THROTTLE_SCALE = 0.55   # throttle multiplier when lane_curve == 1
CUTIN_BRAKE          = 0.7
STEER_DEG_MAX        = 30.0   # matches LaneCurvatureAnalyzer clip range
STEER_SMOOTH_ALPHA   = 0.3    # smoothing applied to the carla steer command

# Write cut_in/lane_curve to disk too, in case you still want to feed the
# ESP32/UART side (e.g. rc_car_controller.py) from this simulated run.
WRITE_SHARED_FILE = True
SHARED_FILE_PATH  = "/tmp/adas_output.txt"

DISPLAY_WINDOW_NAME = "ADAS Inference View (CARLA)"


# ==========================
# YOLOPv2 helpers (unchanged from your pipeline)
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

    _, da_out = ex.extract("677")
    da_arr = np.array(da_out)
    da_c   = da_arr[:, t:da_arr.shape[1] - (hpad - t), l:da_arr.shape[2] - (wpad - l)]
    da_r   = np.stack([cv2.resize(da_c[c], (img_w, img_h),
                                   interpolation=cv2.INTER_LINEAR)
                        for c in range(da_c.shape[0])])

    return objects, ll_r, da_r


# ==========================
# Lane Curvature Analysis (unchanged)
# ==========================

class LaneCurvatureAnalyzer:
    WARN_RADIUS_M    = 30.0
    PIXELS_PER_METER = 10.0
    SMOOTH_ALPHA     = 0.25

    def __init__(self):
        self.smooth_curvature = 0.0
        self.smooth_steer_deg = 0.0
        self.direction        = 0

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
            coeffs = np.polyfit(xs, ys, 2)
        except np.linalg.LinAlgError:
            return self._no_data()

        a, b, _ = coeffs
        eval_x    = float(cx)
        dy_dx     = 2 * a * eval_x + b
        d2y_dx2   = 2 * a
        curvature = abs(d2y_dx2) / ((1 + dy_dx ** 2) ** 1.5 + 1e-9)

        radius_m = (1.0 / (curvature * self.PIXELS_PER_METER + 1e-9))
        radius_m = min(radius_m, 9999.0)

        raw_dir   = 1 if a > 0 else (-1 if a < 0 else 0)
        steer_deg = np.clip(-raw_dir * (1.0 / (radius_m + 1e-9)) * 300.0,
                             -STEER_DEG_MAX, STEER_DEG_MAX)

        self.smooth_curvature = (self.SMOOTH_ALPHA * curvature +
                                  (1 - self.SMOOTH_ALPHA) * self.smooth_curvature)
        self.smooth_steer_deg = (self.SMOOTH_ALPHA * steer_deg +
                                  (1 - self.SMOOTH_ALPHA) * self.smooth_steer_deg)
        self.direction = raw_dir

        warn = radius_m < self.WARN_RADIUS_M

        return {
            'valid':     True,
            'radius_m':  radius_m,
            'curvature': self.smooth_curvature,
            'steer_deg': self.smooth_steer_deg,
            'direction': raw_dir,
            'warn':      warn,
            'coeffs':    coeffs,
            'mid_pts':   list(zip(mid_xs, mid_ys)),
        }

    def _no_data(self):
        return {'valid': False, 'radius_m': 9999, 'curvature': 0,
                'steer_deg': 0, 'direction': 0, 'warn': False,
                'coeffs': None, 'mid_pts': []}


# ==========================
# Shared state between threads
# ==========================

class SharedState:
    """Every field guarded by its own lightweight lock so the sim loop,
    inference thread, and display thread never fight over each other."""

    def __init__(self):
        self._frame_lock = threading.Lock()
        self.frame_id = 0
        self.frame = None            # latest BGR numpy frame from CARLA camera

        self._decision_lock = threading.Lock()
        self.decision = {
            'cut_in': 0, 'lane_curve': 0, 'steer_deg': 0.0,
            'radius_m': 9999.0, 'detections': [], 'lane_mask': None,
            'infer_fps': 0.0, 'frame_id': -1,
        }

        self.stop_event = threading.Event()

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
# CARLA camera callback -> shared frame (must stay CHEAP)
# ==========================

def make_camera_callback(shared: SharedState):
    def _callback(image: carla.Image):
        # carla.Image.raw_data is BGRA, width*height*4 bytes
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = arr[:, :, :3]  # drop alpha, already BGR order
        shared.set_frame(bgr.copy())
    return _callback


# ==========================
# Inference thread: runs at its own pace, never blocks CARLA
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
    print("[INFER] Models warmed up. Inference thread running.\n")

    curvature_analyzer = LaneCurvatureAnalyzer()
    last_seen_frame_id = -1

    while not shared.stop_event.is_set():
        frame, frame_id = shared.get_frame()
        if frame is None or frame_id == last_seen_frame_id:
            # No new frame yet -- don't spin hot, but don't block CARLA either.
            time.sleep(0.005)
            continue
        last_seen_frame_id = frame_id

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

            _, ll, da = yolopv2_infer(pv2_net, frame, INFER_SIZE, CONF_THRESH)

            cut_in = 0
            lane_curve = 0
            steer_deg = 0.0
            radius_m = 9999.0
            lane_binary = None

            if ll is not None:
                lane_binary = np.round(ll[0]) == 1
                img_h, img_w = lane_binary.shape
                center_x = img_w // 2

                curv_info = curvature_analyzer.update(lane_binary)
                steer_deg = curv_info['steer_deg']
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
                cut_in=cut_in, lane_curve=lane_curve, steer_deg=float(steer_deg),
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


# ==========================
# Driving policy: decision -> carla.VehicleControl
# ==========================

class SteerSmoother:
    def __init__(self):
        self.val = 0.0

    def update(self, target):
        self.val = STEER_SMOOTH_ALPHA * target + (1 - STEER_SMOOTH_ALPHA) * self.val
        return self.val


def decision_to_control(decision: dict, smoother: SteerSmoother) -> carla.VehicleControl:
    cut_in     = decision['cut_in']
    lane_curve = decision['lane_curve']
    steer_deg  = decision['steer_deg']

    target_steer = float(np.clip(steer_deg / STEER_DEG_MAX, -1.0, 1.0))
    steer = float(np.clip(smoother.update(target_steer), -1.0, 1.0))

    if cut_in:
        return carla.VehicleControl(throttle=0.0, steer=steer, brake=CUTIN_BRAKE)

    throttle = BASE_THROTTLE * (CURVE_THROTTLE_SCALE if lane_curve else 1.0)
    return carla.VehicleControl(throttle=float(throttle), steer=steer, brake=0.0)


# ==========================
# Display thread: draws the inference overlay, independent of sim/AI pace
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

        cut_in_color = (0, 0, 255) if decision['cut_in'] else (0, 200, 0)
        curve_color  = (0, 165, 255) if decision['lane_curve'] else (0, 200, 0)

        cv2.putText(vis, f"cut_in={decision['cut_in']}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cut_in_color, 2)
        cv2.putText(vis, f"lane_curve={decision['lane_curve']} (R={decision['radius_m']:.1f}m)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, curve_color, 2)
        cv2.putText(vis, f"steer_deg={decision['steer_deg']:.1f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"infer_fps={decision['infer_fps']:.1f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(DISPLAY_WINDOW_NAME, vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared.stop_event.set()
            break

    cv2.destroyWindow(DISPLAY_WINDOW_NAME)


# ==========================
# CARLA setup / main loop
# ==========================

def main():
    shared = SharedState()

    print("[CARLA] Connecting...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    world = client.get_world()
    original_settings = world.get_settings()

    if SYNC_MODE:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter(VEHICLE_BLUEPRINT)[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available on this map.")

    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(False)  # our AI drives, not CARLA's built-in autopilot
    print(f"[CARLA] Spawned {VEHICLE_BLUEPRINT} at {spawn_points[0].location}")

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(CAM_WIDTH))
    camera_bp.set_attribute('image_size_y', str(CAM_HEIGHT))
    camera_bp.set_attribute('fov', str(CAM_FOV))
    camera_transform = carla.Transform(CAM_LOCATION)
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera.listen(make_camera_callback(shared))

    infer_thread = threading.Thread(target=inference_worker, args=(shared,), daemon=True)
    infer_thread.start()

    disp_thread = threading.Thread(target=display_worker, args=(shared,), daemon=True)
    disp_thread.start()

    steer_smoother = SteerSmoother()

    print("=" * 60)
    print("ADAS + CARLA RUNNING")
    print("Sim ticks independently; AI applies decisions as they're ready.")
    print("Press 'q' in the display window (or Ctrl+C here) to stop.")
    print("=" * 60)

    try:
        while not shared.stop_event.is_set():
            if SYNC_MODE:
                world.tick()
            else:
                time.sleep(FIXED_DELTA_SECONDS)

            decision = shared.get_decision()
            control = decision_to_control(decision, steer_smoother)
            vehicle.apply_control(control)

    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")

    finally:
        shared.stop_event.set()
        time.sleep(0.2)
        try:
            camera.stop()
            camera.destroy()
            vehicle.destroy()
        except Exception:
            pass
        if SYNC_MODE:
            world.apply_settings(original_settings)
        print("=" * 60)
        print("ADAS + CARLA Stopped")
        print("=" * 60)


if __name__ == "__main__":
    main()