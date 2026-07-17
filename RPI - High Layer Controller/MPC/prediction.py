# =============================================================================
#  MODULE: prediction.py
#
#  DESCRIPTION:
#  Offline (one-time, at-startup) matrix construction for the condensed
#  MPC formulation. Contains two functions:
#
#    1. compute_terminal_cost(A, B, Q, R) — verifies plant controllability
#       and solves the Discrete Algebraic Riccati Equation (DARE) to
#       obtain the terminal cost matrix P, which anchors the finite-
#       horizon MPC cost to guarantee closed-loop stability.
#
#    2. build_prediction_matrices(A, B, Np, Nc, Q, R, P) — builds the
#       condensed free-response matrix Fx, forced-response matrix Gu, the
#       block-diagonal stacked weighting matrices Q_bar/R_bar, and the
#       QP Hessian H = Gu^T Q_bar Gu + R_bar (symmetrized). These are
#       computed once at initialization since they depend only on the
#       fixed system matrices, horizons, and weights — only the gradient
#       vector f needs to be recomputed at every MPC step.
# =============================================================================

import numpy as np
from scipy.linalg import solve_discrete_are
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("MPC_v3")


# =============================================================================
#  SECTION 7 — TERMINAL COST  (DARE, solved once)
# =============================================================================
def compute_terminal_cost(A, B, Q, R):
    nx = A.shape[0]
    Mc = np.hstack([B, A @ B])
    rank = np.linalg.matrix_rank(Mc)
    if rank < nx:
        log.warning(f"Controllability rank {rank} < {nx}. Check matrices.")
    else:
        log.info(f"Controllability OK: rank = {rank}")

    try:
        P = solve_discrete_are(A, B, Q, R)
        log.info("DARE solved. Terminal cost P computed.")
    except Exception as e:
        log.warning(f"DARE failed ({e}). Using P = Q.")
        P = Q.copy()
    return P


# =============================================================================
#  SECTION 8 — PREDICTION MATRICES  (built once)
# =============================================================================
def build_prediction_matrices(A, B, Np, Nc, Q, R, P):
    nx, nu = A.shape[0], B.shape[1]

    Fx = np.zeros((Np*nx, nx))
    Ap = np.eye(nx)
    for i in range(Np):
        Ap = Ap @ A
        Fx[i*nx:(i+1)*nx, :] = Ap

    Gu = np.zeros((Np*nx, Nc*nu))
    for i in range(Np):
        for j in range(min(i+1, Nc)):
            Aij = np.eye(nx)
            for _ in range(i - j):
                Aij = Aij @ A
            Gu[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Aij @ B

    Q_bar = np.zeros((Np*nx, Np*nx))
    for i in range(Np - 1):
        Q_bar[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = Q
    Q_bar[(Np-1)*nx:, (Np-1)*nx:] = P

    R_bar = np.kron(np.eye(Nc), R)

    H = Gu.T @ Q_bar @ Gu + R_bar
    H = (H + H.T) / 2.0

    log.info(f"Prediction matrices: Fx{Fx.shape} Gu{Gu.shape} H{H.shape}")
    return Fx, Gu, Q_bar, R_bar, H
