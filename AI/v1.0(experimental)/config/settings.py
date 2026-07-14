"""
ADAS System Configuration
All tunable parameters in one place.
"""

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n_ncnn_model"
VIDEO_PATH   = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/temp/test.mp4"
IMGSZ        = 320          # reduced from 640 → faster on RPi 5
CONF_THRESH  = 0.25

# ── Distance ─────────────────────────────────────────────────────────────────
FOCAL_LENGTH_PX   = 800     # calibrate for your camera
REAL_CAR_WIDTH_M  = 1.8     # average vehicle width in metres
DANGER_DIST_M     = 8.0     # distance below which risk starts climbing
CRITICAL_DIST_M   = 3.5

# ── Brake-light detector ─────────────────────────────────────────────────────
BRAKE_ROI_FRAC         = 0.35   # bottom fraction of bbox to sample for lights
BRAKE_RED_SAT_MIN      = 100    # HSV saturation minimum for red
BRAKE_RED_VAL_MIN      = 100    # HSV value minimum
BRAKE_PIXEL_RATIO_ON   = 0.04   # fraction of ROI pixels that must be red → ON
BRAKE_PIXEL_RATIO_OFF  = 0.02   # hysteresis off threshold
BRAKE_CONFIRM_FRAMES   = 2      # frames the signal must persist before locking ON
BRAKE_RELEASE_FRAMES   = 4      # frames signal must be absent before locking OFF

# ── Risk engine ───────────────────────────────────────────────────────────────
# Weight for each risk factor (must sum ≈ 1.0 for readability)
RISK_WEIGHT_DISTANCE   = 0.50
RISK_WEIGHT_BRAKE      = 0.35
RISK_WEIGHT_TTC        = 0.15   # time-to-collision contribution

TTC_DANGER_SECONDS     = 3.0    # TTC below this → max TTC risk

# smoothing factor for displayed risk score (0=no smooth, 1=no update)
RISK_EMA_ALPHA         = 0.35

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_WINDOW   = True
WINDOW_NAME   = "ADAS — Predictive Collision Risk"
FPS_HISTORY   = 30            # rolling-average window length

# Risk level thresholds (0–100 score → label + colour BGR)
RISK_LEVELS = [
    (75, "CRITICAL",  (0,   0,   220)),
    (45, "HIGH RISK", (0,   80,  255)),
    (20, "CAUTION",   (0,  165,  255)),
    ( 0, "SAFE",      (80, 200,   80)),
]
