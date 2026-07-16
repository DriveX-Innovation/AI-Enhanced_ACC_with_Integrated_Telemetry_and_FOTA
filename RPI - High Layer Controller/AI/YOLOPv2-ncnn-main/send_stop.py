#!/usr/bin/env python3

import sys, os, time, threading
import cv2
import numpy as np
import ncnn

MAX_STRIDE = 64

ANCHORS = {
    8:  np.array([12,16, 19,36, 40,28],    dtype=np.float32),
    16: np.array([36,75, 76,55, 72,146],   dtype=np.float32),
    32: np.array([142,110,192,243,459,401],dtype=np.float32),
}

net = None


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(model_dir):
    global net
    param = os.path.join(model_dir, 'yolopv2.param')
    binf  = os.path.join(model_dir, 'yolopv2.bin')
    if not os.path.exists(param): print(f"ERROR: {param} not found"); sys.exit(1)
    if not os.path.exists(binf):  print(f"ERROR: {binf} not found");  sys.exit(1)

    net = ncnn.Net()
    net.opt.num_threads        = 4
    net.opt.use_fp16_storage   = True
    net.opt.use_packing_layout = True
    net.load_param(param)
    net.load_model(binf)
    print(f"Model loaded from {model_dir}")


# ── Letterbox ─────────────────────────────────────────────────────────────────
def letterbox(img, target_size):
    h, w = img.shape[:2]
    if w > h:
        scale = target_size / w
        nw, nh = target_size, int(h * scale)
    else:
        scale = target_size / h
        nh, nw = target_size, int(w * scale)

    resized = cv2.resize(img, (nw, nh))
    wpad = (nw + MAX_STRIDE-1)//MAX_STRIDE*MAX_STRIDE - nw
    hpad = (nh + MAX_STRIDE-1)//MAX_STRIDE*MAX_STRIDE - nh
    padded = cv2.copyMakeBorder(resized,
        hpad//2, hpad-hpad//2, wpad//2, wpad-wpad//2,
        cv2.BORDER_CONSTANT, value=(114,114,114))
    return padded, scale, wpad, hpad


# ── Vectorized proposal decode ────────────────────────────────────────────────
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
        aw = anchors_flat[q*2]; ah = anchors_flat[q*2+1]
        row = feat[q]

        box_conf   = 1/(1+np.exp(-row[:,4]))
        cls_scores = 1/(1+np.exp(-row[:,5:]))
        cls_idx    = np.argmax(cls_scores, axis=1)
        cls_conf   = cls_scores[np.arange(len(cls_scores)), cls_idx]
        confidence = box_conf * cls_conf

        mask = confidence >= conf_thresh
        if not np.any(mask):
            continue

        r  = row[mask]; cx = gx[mask].astype(np.float32); cy = gy[mask].astype(np.float32)
        dx = 1/(1+np.exp(-r[:,0])); dy = 1/(1+np.exp(-r[:,1]))
        dw = 1/(1+np.exp(-r[:,2])); dh = 1/(1+np.exp(-r[:,3]))

        pb_cx = (dx*2-0.5+cx)*stride
        pb_cy = (dy*2-0.5+cy)*stride
        pb_w  = (dw*2)**2 * aw
        pb_h  = (dh*2)**2 * ah

        x0 = pb_cx-pb_w*0.5; y0 = pb_cy-pb_h*0.5
        x1 = pb_cx+pb_w*0.5; y1 = pb_cy+pb_h*0.5

        conf_m = confidence[mask]; idx_m = cls_idx[mask]
        for i in range(len(x0)):
            objects.append({
                'x': float(x0[i]), 'y': float(y0[i]),
                'w': float(x1[i]-x0[i]), 'h': float(y1[i]-y0[i]),
                'label': int(idx_m[i]), 'prob': float(conf_m[i])
            })
    return objects


def nms(objects, nms_threshold=0.45):
    if not objects: return []
    objects = sorted(objects, key=lambda o: o['prob'], reverse=True)
    areas = [o['w']*o['h'] for o in objects]
    picked = []
    for i, a in enumerate(objects):
        keep = True
        for j in picked:
            b = objects[j]
            ix0=max(a['x'],b['x']); iy0=max(a['y'],b['y'])
            ix1=min(a['x']+a['w'],b['x']+b['w']); iy1=min(a['y']+a['h'],b['y']+b['h'])
            inter=max(0,ix1-ix0)*max(0,iy1-iy0)
            if inter/(areas[i]+areas[j]-inter) > nms_threshold:
                keep=False; break
        if keep: picked.append(i)
    return [objects[i] for i in picked]


# ── Combined inference ────────────────────────────────────────────────────────
def detect_combined(bgr, target_size, conf_thresh, want_vehicles=True, want_drivable=False):
    img_h, img_w = bgr.shape[:2]
    padded, scale, wpad, hpad = letterbox(bgr, target_size)
    pad_w, pad_h = padded.shape[1], padded.shape[0]

    mat_in = ncnn.Mat.from_pixels(
        padded, ncnn.Mat.PixelType.PIXEL_BGR2RGB, pad_w, pad_h)
    mat_in.substract_mean_normalize([0,0,0], [1/255.,1/255.,1/255.])

    ex = net.create_extractor()
    ex.input("images", mat_in)

    objects = []
    if want_vehicles:
        proposals = []
        for stride, anch in ANCHORS.items():
            blob_name = {8:"det0",16:"det1",32:"det2"}[stride]
            _, out = ex.extract(blob_name)
            arr = np.array(out)
            proposals.extend(generate_proposals_fast(anch, stride, pad_w, pad_h, arr, conf_thresh))
        objects = nms(proposals)

        for obj in objects:
            x0=(obj['x']-wpad/2)/scale; y0=(obj['y']-hpad/2)/scale
            x1=(obj['x']+obj['w']-wpad/2)/scale; y1=(obj['y']+obj['h']-hpad/2)/scale
            obj['x']=float(np.clip(x0,0,img_w-1)); obj['y']=float(np.clip(y0,0,img_h-1))
            obj['w']=float(np.clip(x1,0,img_w-1))-obj['x']
            obj['h']=float(np.clip(y1,0,img_h-1))-obj['y']

    ll_r = None
    _, ll_out = ex.extract("769")
    ll_arr = np.array(ll_out)
    t=hpad//2; l=wpad//2
    ll_c = ll_arr[:, t:ll_arr.shape[1]-(hpad-t), l:ll_arr.shape[2]-(wpad-l)]
    ll_r = np.stack([cv2.resize(ll_c[c],(img_w,img_h),interpolation=cv2.INTER_LINEAR)
                     for c in range(ll_c.shape[0])])

    da_r = None
    if want_drivable:
        _, da_out = ex.extract("677")
        da_arr = np.array(da_out)
        da_c = da_arr[:, t:da_arr.shape[1]-(hpad-t), l:da_arr.shape[2]-(wpad-l)]
        da_r = np.stack([cv2.resize(da_c[c],(img_w,img_h),interpolation=cv2.INTER_LINEAR)
                         for c in range(da_c.shape[0])])

    return objects, ll_r, da_r


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_results(frame, objects, ll_mask, da_mask, dots, dot_color,
                 disp_fps=0, infer_fps=0):
    image = frame.copy()
    h, w  = image.shape[:2]

    if ll_mask is not None:
        lane_binary = np.round(ll_mask[0]) == 1
        h_lane, w_lane = lane_binary.shape

        # Scan near the bottom to find left/right ego lane boundaries
        scan_y = int(h_lane * 0.90)
        lane_pixels = np.where(lane_binary[scan_y])[0]

        ego_lane = np.zeros_like(lane_binary)
        if len(lane_pixels) >= 2:
            center_x = w_lane // 2
            left_candidates  = lane_pixels[lane_pixels < center_x]
            right_candidates = lane_pixels[lane_pixels >= center_x]

            if len(left_candidates) > 0 and len(right_candidates) > 0:
                left_lane  = left_candidates[-1]
                right_lane = right_candidates[0]
                ego_lane[:, left_lane:right_lane] = lane_binary[:, left_lane:right_lane]

        # Draw ego lane as dots
        ys, xs = np.where(ego_lane[::dots, ::dots])
        ys = ys * dots
        xs = xs * dots
        for x, y in zip(xs, ys):
            cv2.circle(image, (int(x), int(y)), 2, dot_color, -1, lineType=cv2.LINE_AA)

        # Debug: print lane pixel count
        lane_px = int(ego_lane.sum())
        print(f"\rEgo lane pixels: {lane_px:<6}", end="", flush=True)

    cv2.putText(image, f"Display FPS: {disp_fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, f"Infer FPS:   {infer_fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return image


# ── Modes ─────────────────────────────────────────────────────────────────────
def run_image(image_path, model_dir, size, dots, color, conf, want_v, want_d):
    load_model(model_dir)
    frame = cv2.imread(image_path)
    if frame is None: print(f"Cannot read: {image_path}"); sys.exit(1)
    objects, ll, da = detect_combined(frame, size, conf, want_v, want_d)
    result = draw_results(frame, objects, ll, da, dots, color)
    cv2.imwrite("result_ultimate.jpg", result)
    print(f"\nSaved result_ultimate.jpg  ({len(objects)} vehicles)")
    cv2.imshow("YOLOPv2 Ego Lane", result)
    cv2.waitKey(0)


def run_camera(cam_id, model_dir, size, dots, color, conf, want_v, want_d):
    load_model(model_dir)

    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if not cap.isOpened(): cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened(): print(f"Cannot open camera {cam_id}"); sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"Camera: {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f}fps")
    print(f"Inference size: {size}x{size}  vehicles={want_v}  drivable={want_d}")
    print("q/ESC=quit  s=snapshot\n")

    latest_frame  = [None]
    latest_result = [[], None, None]
    lock = threading.Lock()
    stop_event = threading.Event()
    infer_fps_val = [0.0]

    def infer_thread():
        while not stop_event.is_set():
            with lock:
                frame = latest_frame[0]
            if frame is None:
                time.sleep(0.005); continue
            t0 = time.time()
            try:
                objects, ll, da = detect_combined(frame, size, conf, want_v, want_d)
                with lock:
                    latest_result[0] = objects
                    latest_result[1] = ll
                    latest_result[2] = da
                infer_fps_val[0] = 1.0/(time.time()-t0)
            except Exception as e:
                print(f"\nInference error: {e}")

    t = threading.Thread(target=infer_thread, daemon=True)
    t.start()

    snap_count = 0
    disp_fps = 0.0
    t0_disp  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03); continue

        with lock:
            latest_frame[0] = frame.copy()
            objects = latest_result[0]
            ll      = latest_result[1]
            da      = latest_result[2]

        display = draw_results(frame, objects, ll, da, dots, color,
                               disp_fps=disp_fps, infer_fps=infer_fps_val[0])

        cv2.imshow("YOLOPv2 Ego Lane  (q=quit  s=snapshot)", display)

        disp_fps = 1.0/(time.time()-t0_disp+1e-9)
        t0_disp  = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            stop_event.set()
            break
        if key == ord('s'):
            fname = f"snapshot_{snap_count:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"\nSaved {fname}")
            snap_count += 1

    stop_event.set()
    print()
    cap.release()
    cv2.destroyAllWindows()


# ── Entry ─────────────────────────────────────────────────────────────────────
def usage():
    print("Usage: python3 yolopv2_ego_lane.py [image|camera] [path/cam_id] [model_dir]")
    print("Options: --size N  --dots N  --color B,G,R  --conf F  --no-vehicles  --no-drivable")

if __name__ == "__main__":
    if len(sys.argv) < 2: usage(); sys.exit(1)
    mode = sys.argv[1]
    args = sys.argv[2:]

    size = 320
    dots = 6
    color = (255, 180, 80)
    conf = 0.30
    want_vehicles = True
    want_drivable = False   # drivable area off by default
    positional = []

    i = 0
    while i < len(args):
        if args[i] == "--size":
            size = int(args[i+1]); i += 2
        elif args[i] == "--dots":
            dots = int(args[i+1]); i += 2
        elif args[i] == "--color":
            b,g,r = map(int, args[i+1].split(",")); color=(b,g,r); i += 2
        elif args[i] == "--no-vehicles":
            want_vehicles = False; i += 1
        elif args[i] == "--no-drivable":
            want_drivable = False; i += 1
        elif args[i] == "--conf":
            conf = float(args[i+1]); i += 2
        else:
            positional.append(args[i]); i += 1

    if size % MAX_STRIDE != 0:
        print(f"WARNING: --size {size} is not a multiple of {MAX_STRIDE}, rounding up")
        size = ((size + MAX_STRIDE - 1)//MAX_STRIDE) * MAX_STRIDE

    if mode == "image":
        if len(positional) < 1: usage(); sys.exit(1)
        run_image(positional[0],
                  positional[1] if len(positional) >= 2 else "../models",
                  size, dots, color, conf, want_vehicles, want_drivable)

    elif mode == "camera":
        run_camera(int(positional[0]) if len(positional) >= 1 else 0,
                   positional[1] if len(positional) >= 2 else "../models",
                   size, dots, color, conf, want_vehicles, want_drivable)
    else:
        print(f"Unknown mode: {mode}"); usage(); sys.exit(1)
