"""
MonocularDistance
─────────────────
Single-camera distance estimation via known-width geometry.

  distance = (focal_length × real_width) / pixel_width

Designed to be a drop-in replacement for SingleCamDistanceMeasure
while integrating cleanly with the RiskEngine pipeline.
"""

import cv2
import numpy as np
from config.settings import FOCAL_LENGTH_PX, REAL_CAR_WIDTH_M

# Object classes considered as "vehicles" for distance estimation
VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorbike", "motorcycle",
    "bicycle", "van", "vehicle",
}


class MonocularDistance:
    """
    Per-detection distance estimator.

    distance_m = (FOCAL_LENGTH_PX * REAL_CAR_WIDTH_M) / bbox_width_px
    """

    def estimate(self, x1: int, y1: int, x2: int, y2: int,
                 class_name: str = "car") -> float:
        """
        Return metric distance estimate for a single bounding box.
        Returns 0.0 if class is not vehicle-like.
        """
        if class_name.lower() not in VEHICLE_CLASSES:
            return 0.0

        bbox_w = max(1, x2 - x1)
        return (FOCAL_LENGTH_PX * REAL_CAR_WIDTH_M) / bbox_w

    def calibrate(self, known_distance_m: float,
                  measured_px_width: int,
                  real_width_m: float = REAL_CAR_WIDTH_M) -> float:
        """
        Utility: compute focal length from a calibration frame.
        Call once with a vehicle at a known distance and measure its pixel width.
        """
        fl = (measured_px_width * known_distance_m) / real_width_m
        print(f"[Calibration] Focal length = {fl:.1f} px  "
              f"(set FOCAL_LENGTH_PX = {fl:.0f} in config/settings.py)")
        return fl
