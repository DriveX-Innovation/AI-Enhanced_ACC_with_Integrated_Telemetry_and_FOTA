"""
BrakeLightDetector
──────────────────
Detects whether a vehicle's brake lights are active by analysing the
lower portion of its bounding box in HSV colour space.

Key design choices for RPi 5:
  • Works entirely on a small ROI crop  → negligible CPU cost
  • Hysteresis via confirm/release frame counters → no flicker
  • No deep-learning dependency → runs in <1 ms per tracked vehicle
"""

import cv2
import numpy as np
from config.settings import (
    BRAKE_ROI_FRAC,
    BRAKE_RED_SAT_MIN, BRAKE_RED_VAL_MIN,
    BRAKE_PIXEL_RATIO_ON, BRAKE_PIXEL_RATIO_OFF,
    BRAKE_CONFIRM_FRAMES, BRAKE_RELEASE_FRAMES,
)


# Red wraps around H=0/180 in OpenCV HSV
_RED_RANGES = [
    (np.array([0,   BRAKE_RED_SAT_MIN, BRAKE_RED_VAL_MIN]),
     np.array([10,  255, 255])),
    (np.array([165, BRAKE_RED_SAT_MIN, BRAKE_RED_VAL_MIN]),
     np.array([180, 255, 255])),
]


class _VehicleState:
    """Per-vehicle hysteresis state."""
    __slots__ = ("active", "_on_streak", "_off_streak")

    def __init__(self):
        self.active     = False
        self._on_streak  = 0
        self._off_streak = 0

    def update(self, signal: bool) -> bool:
        if signal:
            self._on_streak  += 1
            self._off_streak  = 0
            if self._on_streak >= BRAKE_CONFIRM_FRAMES:
                self.active = True
        else:
            self._off_streak += 1
            self._on_streak   = 0
            if self._off_streak >= BRAKE_RELEASE_FRAMES:
                self.active = False
        return self.active


class BrakeLightDetector:
    """
    Usage
    -----
    detector = BrakeLightDetector()

    # every frame, for each tracked vehicle:
    is_braking = detector.update(track_id, frame, x1, y1, x2, y2)
    """

    def __init__(self):
        self._states: dict[int, _VehicleState] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, track_id: int, frame: np.ndarray,
               x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Analyse the brake-light ROI of *one* vehicle for this frame.

        Returns True when brakes are confirmed active.
        """
        if track_id not in self._states:
            self._states[track_id] = _VehicleState()

        signal = self._measure_red_ratio(frame, x1, y1, x2, y2)
        return self._states[track_id].update(signal)

    def is_braking(self, track_id: int) -> bool:
        """Query current confirmed brake state without updating."""
        state = self._states.get(track_id)
        return state.active if state else False

    def prune(self, active_ids: set[int]) -> None:
        """Remove state for vehicles that are no longer tracked."""
        stale = set(self._states) - active_ids
        for tid in stale:
            del self._states[tid]

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _measure_red_ratio(frame: np.ndarray,
                           x1: int, y1: int,
                           x2: int, y2: int) -> bool:
        """
        Crop the lower BRAKE_ROI_FRAC of the bounding box, convert to HSV,
        threshold for red pixels, and decide if ratio exceeds the on-threshold.
        Uses hysteresis: returns True only above ON threshold; caller holds
        state and applies release threshold.
        """
        h = y2 - y1
        roi_y1 = max(0, y2 - int(h * BRAKE_ROI_FRAC))
        roi    = frame[roi_y1:y2, x1:x2]

        if roi.size == 0:
            return False

        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, *_RED_RANGES[0]) | cv2.inRange(hsv, *_RED_RANGES[1])
        ratio = float(cv2.countNonZero(mask)) / mask.size

        # Use the ON threshold here; the _VehicleState handles hysteresis
        return ratio >= BRAKE_PIXEL_RATIO_ON

    # ── debug visualisation ───────────────────────────────────────────────────

    def draw_roi(self, frame: np.ndarray,
                 x1: int, y1: int, x2: int, y2: int,
                 braking: bool) -> None:
        """Draw a thin ROI rectangle — useful during calibration."""
        h    = y2 - y1
        ry1  = max(0, y2 - int(h * BRAKE_ROI_FRAC))
        col  = (0, 0, 255) if braking else (0, 255, 0)
        cv2.rectangle(frame, (x1, ry1), (x2, y2), col, 1)
