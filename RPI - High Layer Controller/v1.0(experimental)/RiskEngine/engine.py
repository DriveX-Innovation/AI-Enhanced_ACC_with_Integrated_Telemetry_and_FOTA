"""
RiskEngine
──────────
Predictive Collision-Risk Estimator using sensor fusion between:
  1. Monocular distance estimation  (camera geometry)
  2. Brake-light state              (visual analysis)
  3. Time-to-Collision (TTC)        (distance derivative)

Academic framing:
  "The system performs predictive collision-risk estimation using
   sensor fusion between distance sensing and visual brake-light
   analysis, producing a continuous risk score that reacts to
   brake events before distance reduction alone would trigger an alert."

Output: risk score in [0, 100] for each tracked vehicle.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from config.settings import (
    DANGER_DIST_M, CRITICAL_DIST_M,
    RISK_WEIGHT_DISTANCE, RISK_WEIGHT_BRAKE, RISK_WEIGHT_TTC,
    TTC_DANGER_SECONDS, RISK_EMA_ALPHA, RISK_LEVELS,
)


@dataclass
class VehicleRisk:
    """Complete risk snapshot for one vehicle."""
    track_id:       int
    distance_m:     float
    braking:        bool
    ttc_seconds:    float          # positive = approaching; inf = stationary/moving away
    risk_score:     float          # 0–100, EMA-smoothed
    risk_label:     str
    risk_colour:    tuple[int,int,int]  # BGR


class _VehicleHistory:
    """Maintains per-vehicle distance history for TTC estimation."""
    _WINDOW = 8   # frames kept for derivative

    def __init__(self):
        self._times:     deque[float] = deque(maxlen=self._WINDOW)
        self._distances: deque[float] = deque(maxlen=self._WINDOW)
        self._smooth_risk = 50.0     # start at neutral

    def push(self, dist_m: float) -> None:
        self._times.append(time.monotonic())
        self._distances.append(dist_m)

    def ttc(self) -> float:
        """
        Time-to-Collision via linear regression on recent (t, d) pairs.
        Returns +inf when moving away or insufficient data.
        """
        if len(self._distances) < 3:
            return float("inf")

        t = np.array(self._times,     dtype=np.float64)
        d = np.array(self._distances, dtype=np.float64)
        t -= t[0]   # normalise to avoid float precision issues

        # least-squares slope = dd/dt  (negative → approaching)
        slope = float(np.polyfit(t, d, 1)[0])

        if slope >= 0:
            return float("inf")   # moving away — no collision risk from TTC

        current_d = d[-1]
        return current_d / (-slope)   # seconds until d → 0

    def smooth(self, raw: float) -> float:
        self._smooth_risk = (RISK_EMA_ALPHA * raw
                             + (1.0 - RISK_EMA_ALPHA) * self._smooth_risk)
        return self._smooth_risk


class RiskEngine:
    """
    Usage
    -----
    engine = RiskEngine()

    # each frame, per vehicle:
    risk = engine.evaluate(track_id, distance_m, braking=True/False)
    """

    def __init__(self):
        self._histories: dict[int, _VehicleHistory] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self,
                 track_id:   int,
                 distance_m: float,
                 braking:    bool) -> VehicleRisk:
        """Fuse all signals → VehicleRisk."""
        hist = self._get_or_create(track_id)
        hist.push(distance_m)
        ttc = hist.ttc()

        raw   = self._fuse(distance_m, braking, ttc)
        score = hist.smooth(raw)
        label, colour = self._classify(score)

        return VehicleRisk(
            track_id    = track_id,
            distance_m  = distance_m,
            braking     = braking,
            ttc_seconds = ttc,
            risk_score  = score,
            risk_label  = label,
            risk_colour = colour,
        )

    def prune(self, active_ids: set[int]) -> None:
        stale = set(self._histories) - active_ids
        for tid in stale:
            del self._histories[tid]

    # ── fusion logic ──────────────────────────────────────────────────────────

    @staticmethod
    def _fuse(dist_m: float, braking: bool, ttc: float) -> float:
        """
        Combine three independent risk components into [0, 100].

        Distance component  — monotonically increasing as gap closes.
        Brake component     — step function: braking adds immediate risk.
        TTC component       — rises sharply below TTC_DANGER_SECONDS.

        The brake component fires BEFORE distance collapses, which is the
        key academic contribution: early warning via visual cue fusion.
        """
        # 1. Distance risk ─────────────────────────────────────────────────
        if dist_m <= CRITICAL_DIST_M:
            d_risk = 1.0
        elif dist_m >= DANGER_DIST_M:
            d_risk = 0.0
        else:
            # smooth sigmoid-ish curve in [CRITICAL, DANGER]
            span   = DANGER_DIST_M - CRITICAL_DIST_M
            d_risk = 1.0 - ((dist_m - CRITICAL_DIST_M) / span) ** 1.5

        # 2. Brake-light risk ──────────────────────────────────────────────
        # Full weight when braking, zero when not.
        # This component contributes even at safe distance → early warning.
        b_risk = 1.0 if braking else 0.0

        # 3. TTC risk ──────────────────────────────────────────────────────
        if ttc == float("inf") or ttc > TTC_DANGER_SECONDS * 4:
            t_risk = 0.0
        elif ttc <= 1.0:
            t_risk = 1.0
        else:
            t_risk = 1.0 - ((ttc - 1.0) / (TTC_DANGER_SECONDS * 4 - 1.0)) ** 0.7

        # 4. Weighted fusion ───────────────────────────────────────────────
        raw = (RISK_WEIGHT_DISTANCE * d_risk
               + RISK_WEIGHT_BRAKE  * b_risk
               + RISK_WEIGHT_TTC    * t_risk)

        return min(100.0, raw * 100.0)

    # ── classification ────────────────────────────────────────────────────────

    @staticmethod
    def _classify(score: float) -> tuple[str, tuple[int,int,int]]:
        for threshold, label, colour in RISK_LEVELS:
            if score >= threshold:
                return label, colour
        return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_or_create(self, track_id: int) -> _VehicleHistory:
        if track_id not in self._histories:
            self._histories[track_id] = _VehicleHistory()
        return self._histories[track_id]
