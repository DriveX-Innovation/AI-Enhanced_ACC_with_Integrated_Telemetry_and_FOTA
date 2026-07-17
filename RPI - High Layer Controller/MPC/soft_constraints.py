# =============================================================================
#  MODULE: soft_constraints.py
#
#  DESCRIPTION:
#  Implements the analytic gradient contribution of the soft-constraint
#  penalties (headway and velocity) that are added to the base MPC
#  gradient vector f before the QP is solved each step. Soft constraints
#  are used instead of hard constraints for these state-dependent bounds
#  because rigid enforcement could otherwise render the QP infeasible
#  under extreme disturbance scenarios (e.g. the lead vehicle braking
#  suddenly to zero).
#
#  soft_gradient() evaluates, at every point along the free-response
#  prediction (X_free = Fx @ x), whether the predicted gap distance
#  violates the constant-time-gap safe headway, or whether the predicted
#  velocity exceeds the reference speed or dips below zero — and if so,
#  accumulates the corresponding penalty gradient contribution.
# =============================================================================

import numpy as np
from config import d_safe, tg, v_ref, rho_headway, rho_velocity


# =============================================================================
#  SECTION 9 — SOFT CONSTRAINT GRADIENT
# =============================================================================
def soft_gradient(Fx, Gu, x, Np, nx):
    X_free = Fx @ x
    f_s = np.zeros(Gu.shape[1])

    for i in range(Np):
        d_p = X_free[i*nx + 0]
        v_p = X_free[i*nx + 1]

        eps1 = max(0.0, (d_safe + tg*v_p) - d_p)
        if eps1 > 0:
            f_s -= 2.0 * rho_headway * eps1 * Gu[i*nx+0, :]

        eps2 = max(0.0, v_p - v_ref)
        if eps2 > 0:
            f_s += 2.0 * rho_velocity * eps2 * Gu[i*nx+1, :]

        eps3 = max(0.0, -v_p)
        if eps3 > 0:
            f_s -= 2.0 * rho_velocity * eps3 * Gu[i*nx+1, :]

    return f_s
