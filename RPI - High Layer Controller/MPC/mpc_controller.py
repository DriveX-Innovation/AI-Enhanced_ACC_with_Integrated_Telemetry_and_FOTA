# =============================================================================
#  MODULE: mpc_controller.py
#
#  DESCRIPTION:
#  Defines the top-level MPCController class, which ties together the
#  offline-computed prediction matrices (prediction.py), the soft
#  constraint gradient (soft_constraints.py), and the OSQP QP solver
#  (qp_solver.py) into a single per-cycle step() interface used by the
#  main control loop.
#
#  On construction, MPCController:
#    - Logs the open-loop plant eigenvalues.
#    - Computes the DARE terminal cost P.
#    - Builds the condensed prediction matrices Fx, Gu, Q_bar, R_bar, H.
#    - Instantiates the OSQPSolver with the fixed Hessian and constraint
#      bounds.
#    - Initializes u_prev = 0 for the first control cycle.
#
#  step(x, x_ref, v_lead) performs one MPC control cycle:
#    1. Applies the lead-vehicle disturbance to the effective state.
#    2. Builds the tracking-error gradient against the tiled reference.
#    3. Adds the soft-constraint gradient contribution.
#    4. Solves the QP for the optimal control sequence.
#    5. Propagates the plant model one step forward using the first
#       optimal control action, clamping the predicted velocity to
#       [v_min, v_ref].
#    6. Returns the optimal control sequence, the predicted next-step
#       velocity, and the applied acceleration command.
# =============================================================================

import numpy as np
import logging

from config import A, B, Bd, nx, nu, Np, Nc, Q, R, a_min, a_max, da_max, v_min, v_ref, Ts
from prediction import compute_terminal_cost, build_prediction_matrices
from soft_constraints import soft_gradient
from qp_solver import OSQPSolver

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("MPC_v3")


# =============================================================================
#  SECTION 12 — MPC CONTROLLER CLASS
# =============================================================================
class MPCController:
    def __init__(self):
        eigs = np.linalg.eigvals(A)
        log.info(f"Open-loop eigenvalues: {np.round(np.abs(eigs), 6)}")

        self.P = compute_terminal_cost(A, B, Q, R)
        self.Fx, self.Gu, self.Q_bar, self.R_bar, self.H = \
            build_prediction_matrices(A, B, Np, Nc, Q, R, self.P)
        self.qp     = OSQPSolver(self.H, Nc, nu, a_min, a_max, da_max)
        self.u_prev = np.zeros(nu)

        log.info(f"MPC v3 (OSQP) ready. Np={Np}, Nc={Nc}, Ts={Ts}s")

    def step(self, x, x_ref, v_lead):
        x_eff    = x.copy()
        x_eff[0] += Ts * v_lead

        X_ref   = np.tile(x_ref, Np)
        e_free  = self.Fx @ x_eff - X_ref
        f_base  = self.Gu.T @ self.Q_bar @ e_free
        f_soft  = soft_gradient(self.Fx, self.Gu, x_eff, Np, nx)
        f_total = f_base + f_soft

        U_opt = self.qp.solve(f_total, self.u_prev)

        u_opt       = U_opt[:nu]
        self.u_prev = u_opt.copy()

        x_next    = A @ x + B @ u_opt + Bd.flatten() * v_lead
        x_next[1] = np.clip(x_next[1], v_min, v_ref)

        return u_opt, float(x_next[1]), float(u_opt[0])
