"""
HUDRenderer
───────────
Draws all ADAS overlays onto a frame in a clean, automotive-style HUD.
Keeps all cv2 drawing calls in one place, away from business logic.
"""

import cv2
import numpy as np
from typing import Sequence
from RiskEngine.engine import VehicleRisk
from config.settings import RISK_LEVELS


# ── colour palette ────────────────────────────────────────────────────────────
_C_WHITE   = (255, 255, 255)
_C_BLACK   = (0,   0,   0)
_C_DARK    = (20,  20,  20)
_C_BRAKE   = (0,   0,   200)
_C_GREY    = (160, 160, 160)
_ALPHA_BOX = 0.45


def _blend_rect(frame: np.ndarray, pt1, pt2, colour, alpha: float) -> None:
    """Draw a semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class HUDRenderer:
    """
    Usage
    -----
    hud = HUDRenderer()
    hud.draw_vehicle(frame, risk)          # one vehicle
    hud.draw_status_bar(frame, fps, risks) # top-of-frame summary
    """

    # ── per-vehicle overlay ───────────────────────────────────────────────────

    def draw_vehicle(self, frame: np.ndarray,
                     risk: VehicleRisk,
                     x1: int, y1: int, x2: int, y2: int,
                     show_brake_roi: bool = False) -> None:
        col   = risk.risk_colour
        score = risk.risk_score

        # Bounding box — thickness scales with danger
        thickness = 1 + int(score / 35)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness)

        # Label pill above bbox
        label_parts = [f"{risk.risk_label}  {score:.0f}"]
        label_parts.append(f"d={risk.distance_m:.1f}m")
        if risk.braking:
            label_parts.append("BRAKE!")
        if risk.ttc_seconds < 5.0:
            label_parts.append(f"TTC={risk.ttc_seconds:.1f}s")
        label = "  |  ".join(label_parts)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
        pill_y1 = max(0, y1 - th - 10)
        pill_y2 = y1
        _blend_rect(frame, (x1, pill_y1), (x1 + tw + 8, pill_y2), col, 0.7)
        cv2.putText(frame, label, (x1 + 4, pill_y2 - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, _C_WHITE, 1, cv2.LINE_AA)

        # Mini risk bar on the right edge of bbox
        self._draw_risk_bar(frame, x2 + 4, y1, y2, score, col)

        # Brake indicator
        if risk.braking:
            self._draw_brake_indicator(frame, x1, y1, x2, y2)

    # ── top-of-frame status strip ─────────────────────────────────────────────

    def draw_status_bar(self, frame: np.ndarray,
                        fps: float,
                        risks: Sequence[VehicleRisk]) -> None:
        h, w = frame.shape[:2]
        bar_h = 42
        _blend_rect(frame, (0, 0), (w, bar_h), _C_DARK, 0.75)

        # FPS
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, _C_GREY, 1, cv2.LINE_AA)

        # Highest risk vehicle summary
        if risks:
            top = max(risks, key=lambda r: r.risk_score)
            summary = (f"LEAD VEHICLE  {top.risk_label}  "
                       f"d={top.distance_m:.1f}m  score={top.risk_score:.0f}")
            if top.braking:
                summary += "   ⚠ BRAKE DETECTED"
            cv2.putText(frame, summary, (110, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62,
                        top.risk_colour, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No vehicles detected", (110, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, _C_GREY, 1, cv2.LINE_AA)

    # ── legend (draw once at startup) ────────────────────────────────────────

    def draw_legend(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        x0 = w - 160
        y0 = 55
        _blend_rect(frame, (x0 - 6, y0 - 18), (w - 4, y0 + len(RISK_LEVELS)*22 + 4),
                    _C_DARK, 0.6)
        cv2.putText(frame, "RISK SCALE", (x0, y0),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, _C_GREY, 1)
        for i, (_, label, col) in enumerate(RISK_LEVELS):
            y = y0 + 20 + i * 22
            cv2.rectangle(frame, (x0, y - 10), (x0 + 12, y + 2), col, -1)
            cv2.putText(frame, label, (x0 + 18, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.40, col, 1)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_risk_bar(frame, x, y1, y2, score, col):
        bar_w, bar_h = 6, y2 - y1
        fill = int(bar_h * score / 100)
        cv2.rectangle(frame, (x, y1), (x + bar_w, y2), _C_DARK, -1)
        cv2.rectangle(frame, (x, y2 - fill), (x + bar_w, y2), col, -1)

    @staticmethod
    def _draw_brake_indicator(frame, x1, y1, x2, y2):
        """Pulsing red halo around the bbox when braking."""
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                      _C_BRAKE, 2, cv2.LINE_AA)
        # "BRAKE" stamp in bottom-right corner of bbox
        cv2.putText(frame, "BRAKE", (x2 - 58, y2 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, _C_BRAKE, 2, cv2.LINE_AA)
