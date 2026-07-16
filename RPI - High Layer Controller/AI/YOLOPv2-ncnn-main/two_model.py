#!/usr/bin/env python3
"""
Merged ADAS: YOLOv8n (detection + bounding boxes) + YOLOPv2 (lane lines + drivable area)
Threading fix: display loop and inference thread each hold the shared lock
for ONE list assignment only — never during inference or imshow.
"""

import sys, os, time, threading
import cv2
import numpy as np
import ncnn
from ultralytics import YOLO
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# ==========================
# CONFIGURATION
# ==========================
YOLOV8_MODEL_PATH  = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n_ncnn_model"
YOLOPV2_MODEL_DIR  = "../models"   # must contain yolopv2.param + yolopv2.bin
WEBCAM_INDEX       = 0
INFER_SIZE         = 320           # YOLOPv2 inference size (multiple of 64)
CONF_THRESH        = 0.30
LANE_DOTS          = 6
LANE_DOT_COLOR     = (255, 180, 80)

# YOLOv8 vehicle class ids: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES    = [2, 3, 5, 7]

MAX_STRIDE = 64

ANCHORS = {
    8:  np.array([12,16, 19,36, 40,28],     dtype=np.float32),
    16: np.array([36,75, 76,55, 72,146],    dtype=np.float32),
    32: np.array([142,110,192,243,459,401], dtype=np.float32),
}

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
    """
    Single backbone pass: detection blobs + lane-line blob + drivable-area blob.
    Returns (objects, ll_resized, da_resized) — ll and da are full-res numpy arrays.
    """
    img_h, img_w = bgr.shape[:2]
    padded, scale, wpad, hpad = letterbox(bgr, target_size)
    pad_w, pad_h = padded.shape[1], padded.shape[0]

    mat_in = ncnn.Mat.from_pixels(
        padded, ncnn.Mat.PixelType.PIXEL_BGR2RGB, pad_w, pad_h)
    mat_in.substract_mean_normalize([0, 0, 0], [1 / 255., 1 / 255., 1 / 255.])

    ex = net.create_extractor()
    ex.input("images", mat_in)

    # ---- vehicle detection ----
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

    # ---- lane lines ----
    _, ll_out = ex.extract("769")
    ll_arr = np.array(ll_out)
    ll_c   = ll_arr[:, t:ll_arr.shape[1] - (hpad - t), l:ll_arr.shape[2] - (wpad - l)]
    ll_r   = np.stack([cv2.resize(ll_c[c], (img_w, img_h),
                                  interpolation=cv2.INTER_LINEAR)
                       for c in range(ll_c.shape[0])])

    # ---- drivable area ----
    _, da_out = ex.extract("677")
    da_arr = np.array(da_out)
    da_c   = da_arr[:, t:da_arr.shape[1] - (hpad - t), l:da_arr.shape[2] - (wpad - l)]
    da_r   = np.stack([cv2.resize(da_c[c], (img_w, img_h),
                                  interpolation=cv2.INTER_LINEAR)
                       for c in range(da_c.shape[0])])

    return objects, ll_r, da_r


# ==========================
# Threaded webcam
# ==========================

class ThreadedWebcamStream:
    """Decouples frame grabbing from the main thread to eliminate USB sync stalls."""
    def __init__(self, src=WEBCAM_INDEX):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop(); return
            self.ret  = ret
            self.frame = frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# ==========================
# Main
# ==========================

def main():
    # ------------------------------------------------------------------
    # 1. Setup Shared State & Dependencies
    # ------------------------------------------------------------------
    lock           = threading.Lock()
    latest_frame   = [None]
    latest_result  = [None]
    stop_event     = threading.Event()
    infer_fps_val  = [0.0]

    # Initialize webcam first so the stream starts buffering
    print("Starting webcam stream...")
    vs = ThreadedWebcamStream(src=WEBCAM_INDEX).start()
    time.sleep(1.0)  # sensor warm-up

    distanceDetector = SingleCamDistanceMeasure()
    
    # We define class names here so the main thread has access to them 
    # for the distance detector before YOLO finishes loading.
    class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    # ------------------------------------------------------------------
    # 2. Inference Thread (Models loaded HERE to prevent C++ thread crash)
    # ------------------------------------------------------------------
    def infer_thread():
        print("[Thread] Loading YOLOv8n NCNN model...")
        yolo = YOLO(YOLOV8_MODEL_PATH, task='detect')
        
        print("[Thread] Loading YOLOPv2 NCNN model...")
        pv2_net = ncnn.Net()
        pv2_net.opt.num_threads        = 4
        pv2_net.opt.use_fp16_storage   = True
        pv2_net.opt.use_packing_layout = True
        pv2_net.load_param(os.path.join(YOLOPV2_MODEL_DIR, 'yolopv2.param'))
        pv2_net.load_model(os.path.join(YOLOPV2_MODEL_DIR, 'yolopv2.bin'))

        # Warm-up pass to prevent initialization stutter
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = list(yolo.predict(dummy_frame, imgsz=640, verbose=False, classes=VEHICLE_CLASSES))
        print("[Thread] Models loaded and warmed up successfully.")

        while not stop_event.is_set():
            with lock:
                frame = latest_frame[0]
            
            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.time()
            try:
                # ── YOLOv8n Detection ──
                yolo_results = list(yolo.predict(
                    frame, imgsz=640, conf=0.25, verbose=False,
                    stream=True, classes=VEHICLE_CLASSES))

                detections = []
                annotated  = frame.copy()

                if yolo_results:
                    result = yolo_results[0]
                    annotated = result.plot()
                    boxes = result.boxes
                    if len(boxes) > 0:
                        xyxy  = boxes.xyxy.numpy().astype(int)
                        confs = boxes.conf.numpy()
                        clss  = boxes.cls.numpy().astype(int)
                        detections = [
                            (xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], confs[i], clss[i])
                            for i in range(len(boxes))
                        ]

                # ── YOLOPv2 Lane & Drivable Area ──
                _, ll, da = yolopv2_infer(pv2_net, frame, INFER_SIZE, CONF_THRESH)

                # ── Publish ──
                with lock:
                    latest_result[0] = {
                        'annotated':  annotated,
                        'detections': detections,
                        'll':         ll,
                        'da':         da,
                    }

                infer_fps_val[0] = 1.0 / (time.time() - t0 + 1e-9)

            except Exception as e:
                print(f"[Inference error] {e}")

    # Start the inference thread
    t = threading.Thread(target=infer_thread, daemon=True)
    t.start()

    # ------------------------------------------------------------------
    # 3. Main Display Loop
    # ------------------------------------------------------------------
# ------------------------------------------------------------------
    # 3. Main Display Loop
    # ------------------------------------------------------------------
    disp_fps  = 0.0
    t0_disp   = time.time()
    snap_count = 0

    window_name = "ADAS — YOLOv8n + YOLOPv2  (q / ESC = quit  |  s = snapshot)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    while True:
        ret, frame = vs.read()
        if not ret or frame is None or np.sum(frame) == 0:
            time.sleep(0.01)
            continue

        # ---> BULLETPROOF FALLBACK: Guaranteed assignment before logic <---
        display = frame.copy()

        with lock:
            latest_frame[0] = frame.copy()
            result_payload = latest_result[0]

        # Overwrite 'display' only if the inference thread has given us data
        if result_payload is not None:
            display    = result_payload['annotated'].copy()
            ll         = result_payload['ll']
            da         = result_payload['da']
            detections = result_payload['detections']

            # Create an overlay for alpha blending (transparency)
            overlay = display.copy()

            # --- Drivable-Area Overlay (Green tint) ---
            if da is not None and da.shape[0] >= 2:
                drivable = da[0] < da[1]
                overlay[drivable] = (0, 180, 0)

            # Initialize cut_in variable to default 0
            cut_in = 0

            # --- Ego Lane ("My Lane") & Cut-In Logic ---
            if ll is not None:
                lane_binary = np.round(ll[0]) == 1
                img_h, img_w = lane_binary.shape
                center_x = img_w // 2

                pts_left = []
                pts_right = []
                
                # Fallback boundaries if lane lines momentarily disappear
                last_l = int(center_x * 0.2)  
                last_r = int(center_x * 1.8)  

                # 1. Scan lower half of image to build Ego Lane polygon
                for y in range(img_h - 1, int(img_h * 0.45), -15):
                    row = lane_binary[y]
                    lane_xs = np.where(row)[0]
                    
                    curr_l, curr_r = last_l, last_r
                    if len(lane_xs) > 0:
                        left_candidates = lane_xs[lane_xs < center_x]
                        right_candidates = lane_xs[lane_xs > center_x]
                        
                        if len(left_candidates) > 0:
                            curr_l = left_candidates[-1]
                            last_l = curr_l
                        if len(right_candidates) > 0:
                            curr_r = right_candidates[0]
                            last_r = curr_r
                            
                    pts_left.append([curr_l, y])
                    pts_right.append([curr_r, y])

                ego_poly = np.array(pts_left + pts_right[::-1], np.int32)
                
                if len(ego_poly) > 0:
                    # Paint a distinct BLUE overlay for "My Lane"
                    cv2.fillPoly(overlay, [ego_poly], (255, 0, 0))

                # Blend the green drivable area and blue ego lane into the display
                display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

                # 2. Check for Cut-Ins
                if len(ego_poly) > 0:
                    for det in detections:
                        x1, y1, x2, y2, conf, cls = det
                        bx = int((x1 + x2) / 2)
                        by = int(y2)  # Bottom-center of the vehicle's bounding box
                        
                        # Check if vehicle's footprint is inside the Blue Ego Polygon
                        is_inside = cv2.pointPolygonTest(ego_poly, (bx, by), False)
                        if is_inside >= 0:
                            cut_in = 1
                            # Highlight the invading vehicle in heavy Red
                            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                            cv2.putText(display, "IN MY LANE!", (int(x1), int(y1)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                # Output to Terminal
                if cut_in == 1:
                    print(f"[{time.strftime('%H:%M:%S')}] ALERT: cut_in = {cut_in} | Vehicle in Ego Lane!")

                # 3. Draw Lane-line dots
                ys, xs = np.where(lane_binary[::LANE_DOTS, ::LANE_DOTS])
                ys = ys * LANE_DOTS; xs = xs * LANE_DOTS
                for x, y in zip(xs, ys):
                    cv2.circle(display, (int(x), int(y)), 2, LANE_DOT_COLOR, -1, lineType=cv2.LINE_AA)
            else:
                # If no lane lines detected, just blend the drivable area
                display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

            # Distance / ADAS overlays
            distanceDetector.updateDistance(detections, class_names)
            distanceDetector.DrawDetectedOnFrame(display)

        # FPS counters
        disp_fps = 1.0 / (time.time() - t0_disp + 1e-9)
        t0_disp  = time.time()

        cv2.putText(display, f"Display FPS: {disp_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Infer FPS:   {infer_fps_val[0]:.1f}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        # Visual indicator on screen of cut_in variable state
        cv2.putText(display, f"cut_in: {cut_in if 'cut_in' in locals() else 0}", (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if ('cut_in' in locals() and cut_in==1) else (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            fname = f"snapshot_{snap_count:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")
            snap_count += 1

    stop_event.set()
    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


