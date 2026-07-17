# =============================================================================
#  MODULE: qp_solver.py
#
#  DESCRIPTION:
#  Wraps the OSQP (Operator Splitting Quadratic Program) solver used to
#  solve the condensed MPC QP problem in real time on the Raspberry Pi 5.
#
#  The OSQPSolver class:
#    - Builds the sparse Hessian P_sp from the (fixed) prediction Hessian H.
#    - Builds the constraint matrix A_sp, stacking a box-constraint
#      identity block (bounding each control move to [a_min, a_max]) on
#      top of a first-difference matrix D_diff (bounding the rate of
#      change between consecutive control moves to [-da_max, +da_max]).
#    - Performs a one-time OSQP setup() call at initialization, enabling
#      warm starting so that subsequent solves reuse the prior solution
#      as an initial guess for fast convergence.
#    - solve(f_total, u_prev) — updates the linear cost term q and the
#      rate-constraint bounds (which shift with u_prev each step), then
#      resolves the QP. Falls back to holding u_prev if OSQP fails to
#      reach a valid solution.
# =============================================================================

import numpy as np
import scipy.sparse as sp
import osqp
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("MPC_v3")


# =============================================================================
#  SECTION 10 — OSQP QP SOLVER
# =============================================================================
class OSQPSolver:
    def __init__(self, H, Nc, nu, a_min, a_max, da_max):
        self.Nc     = Nc
        self.nu     = nu
        self.n      = Nc * nu
        self.da_max = da_max
        self.n_box  = Nc * nu
        self.n_rate = Nc * nu

        P_sp = sp.csc_matrix(np.triu(H))

        I_block = sp.eye(self.n_box, self.n, format='csc')

        rows, cols, vals = [], [], []
        for k in range(Nc * nu):
            rows.append(k); cols.append(k); vals.append(1.0)
            if k > 0:
                rows.append(k); cols.append(k-1); vals.append(-1.0)
        D_diff = sp.csc_matrix(
            (vals, (rows, cols)), shape=(self.n_rate, self.n))

        self.A_sp  = sp.vstack([I_block, D_diff], format='csc')
        self.n_con = self.A_sp.shape[0]

        l_box  = np.full(self.n_box,  a_min)
        u_box  = np.full(self.n_box,  a_max)
        l_rate = np.full(self.n_rate, -da_max)
        u_rate = np.full(self.n_rate, +da_max)

        self.l = np.concatenate([l_box,  l_rate])
        self.u = np.concatenate([u_box,  u_rate])

        self.solver = osqp.OSQP()
        self.solver.setup(
            P            = P_sp,
            q            = np.zeros(self.n),
            A            = self.A_sp,
            l            = self.l,
            u            = self.u,
            warm_starting= True,
            verbose      = False,
            eps_abs      = 1e-4,
            eps_rel      = 1e-4,
            max_iter     = 4000,
            polish       = True,
        )
        log.info(f"OSQP setup: {self.n} variables, {self.n_con} constraints.")

    def solve(self, f_total, u_prev):
        l_new = self.l.copy()
        u_new = self.u.copy()
        for i in range(self.nu):
            idx        = self.n_box + i
            l_new[idx] = -self.da_max + u_prev[i]
            u_new[idx] = +self.da_max + u_prev[i]

        self.solver.update(q=f_total, l=l_new, u=u_new)
        result = self.solver.solve()

        if result.info.status not in ("solved", "solved_inaccurate"):
            log.warning(f"OSQP status: {result.info.status}. Using u_prev.")
            return np.tile(u_prev, self.Nc)

        return result.x
