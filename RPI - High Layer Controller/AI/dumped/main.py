"""
ADAS System — Main Entry Point
================================
Predictive Collision-Risk Estimator
  • YOLOv8 NCNN object detection
  • Monocular distance estimation
  • Visual brake-light detection
  • Time-to-Collision computation
  • Fused risk score with EMA smoothing

Optimised for Raspberry Pi 5:
  • NCNN backend (ARM-optimised)
  • imgsz=320 for ≥15 fps on RPi 5
  • Minimal memory allocations in the hot loop
  • Single-thread, sequential pipeline

Academic reference:
  "Predictive collision-risk estimation using sensor fusion between
   distance sensing and visual brake-light analysis."
"""

import sys
import cv2
from ultralytics import YOLO

from config.settings  import (MODEL_PATH, VIDEO_PATH,
                               IMGSZ, CONF_THRESH, SHOW_WINDOW, WINDOW_NAME)
from ObjectDetector.distance  import MonocularDistance, VEHICLE_CLASSES
from BrakeLightDetector.detector import BrakeLightDetector
from RiskEngine.engine          import RiskEngine
from utils.hud                  import HUDRenderer
from utils.fps                  import FPSTracker


def main() -> None:
    # ── Initialise subsystems ─────────────────────────────────────────────────
    print("[ADAS] Loading YOLOv8-NCNN model …")
    model = YOLO(MODEL_PATH, task="detect")

    distance_estimator = MonocularDistance()
    brake_detector     = BrakeLightDetector()
    risk_engine        = RiskEngine()
    hud                = HUDRenderer()
    fps_tracker        = FPSTracker()

    # ── Video source ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"[ADAS] Cannot open video source: {VIDEO_PATH}")

    print(f"[ADAS] Running on: {VIDEO_PATH}")
    print("[ADAS] Press ESC to quit.\n")

    # Synthetic track-id counter (replace with a real tracker if desired)
    _next_id    = 0
    _active_ids: set[int] = set()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ADAS] End of video stream.")
            break

        fps = fps_tracker.tick()

        # ── 1. Object detection ───────────────────────────────────────────────
        results  = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRESH,
                                 verbose=False)
        boxes    = results[0].boxes
        names    = model.names

        # annotated_frame = results[0].plot()   # use if you want YOLO's default boxes
        annotated_frame = frame.copy()

        # ── 2. Per-detection pipeline ─────────────────────────────────────────
        risks        = []
        _active_ids  = set()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id          = int(box.cls[0])
            class_name      = names[cls_id]

            # Skip non-vehicles (pedestrians, signs, etc.)
            if class_name.lower() not in VEHICLE_CLASSES:
                continue

            # Stable pseudo track-id: map detection index → id this session
            # For production, replace with ByteTrack / BotSORT via
            #   model.track(frame, persist=True, tracker="bytetrack.yaml")
            track_id = i          # simple; works per-frame without tracker

            _active_ids.add(track_id)

            # 2a. Distance ─────────────────────────────────────────────────────
            dist_m = distance_estimator.estimate(x1, y1, x2, y2, class_name)

            # 2b. Brake lights ─────────────────────────────────────────────────
            braking = brake_detector.update(track_id, frame, x1, y1, x2, y2)

            # 2c. Risk fusion ──────────────────────────────────────────────────
            risk = risk_engine.evaluate(track_id, dist_m, braking)
            risks.append(risk)

            # 2d. HUD overlay ──────────────────────────────────────────────────
            hud.draw_vehicle(annotated_frame, risk, x1, y1, x2, y2)

        # Prune stale state from subsystems
        brake_detector.prune(_active_ids)
        risk_engine.prune(_active_ids)

        # ── 3. Status bar + legend ────────────────────────────────────────────
        hud.draw_status_bar(annotated_frame, fps, risks)
        hud.draw_legend(annotated_frame)

        # ── 4. Display ────────────────────────────────────────────────────────
        if SHOW_WINDOW:
            cv2.imshow(WINDOW_NAME, annotated_frame)
            if cv2.waitKey(1) == 27:   # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[ADAS] Shutdown complete.")


if __name__ == "__main__":
    main()
