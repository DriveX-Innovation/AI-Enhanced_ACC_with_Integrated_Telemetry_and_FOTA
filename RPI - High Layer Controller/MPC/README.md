# AI-Enhanced Adaptive Cruise Control — MPC Module (OSQP)

Real-time longitudinal Model Predictive Controller for the AI-Enhanced Adaptive
Cruise Control system. This module runs on the **Raspberry Pi 5** upper layer,
solves a condensed-form constrained QP at each 50 ms control step using
**OSQP**, and communicates the resulting speed command to the **ESP32**
lower-layer PID controller over UART.

This directory contains the modularized version of the original monolithic
`mpc_osqp.py` script. All logic, tuning values, and control flow are
unchanged — the code has only been reorganized into single-responsibility
modules for readability and maintainability.

---

## Architecture

```
                         ┌─────────────────────┐
                         │      config.py       │
                         │  (all constants)     │
                         └──────────┬───────────┘
                                    │
        ┌───────────────┬──────────┼──────────┬───────────────┐
        ▼               ▼          ▼          ▼               ▼
  sensors.py      uart_comm.py  prediction.py  soft_constraints.py
  (HC-SR04)       (ESP32 link)  (DARE + Fx/Gu)  (headway/velocity)
        │               │          │          │
        └───────────────┴────┬─────┴──────────┘
                              ▼
                        qp_solver.py
                        (OSQPSolver)
                              ▼
                      mpc_controller.py
                      (MPCController.step)
                              ▼
                          main.py
                    (50 ms control loop)
```

## Module Reference

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for sampling time, plant state-space matrices (A, B, Bd), MPC horizons (Np, Nc), cost weights (Q, R), soft-constraint weights, actuator/safety bounds, and UART settings. |
| `sensors.py` | Wraps the HC-SR04 ultrasonic sensor (GPIO 23/24). Returns a validated gap distance or `None` on an out-of-range/failed reading, so the control loop can fall back to its kinematic model estimate. |
| `uart_comm.py` | Defines the thread-safe `shared` state dict and `esp_lock`, opens/writes the UART link to the ESP32, and runs the background `esp_listener` thread that continuously drains and validates incoming ESP32 telemetry frames without blocking the control loop. |
| `prediction.py` | One-time offline setup: verifies plant controllability, solves the Discrete Algebraic Riccati Equation (DARE) for the terminal cost `P`, and builds the condensed prediction matrices `Fx`, `Gu`, `Q_bar`, `R_bar`, and the QP Hessian `H`. |
| `soft_constraints.py` | Computes the analytic gradient contribution of the soft headway and velocity constraints, added to the base tracking gradient before each QP solve. |
| `qp_solver.py` | `OSQPSolver` — builds the sparse QP (box + rate-of-change constraints), performs a one-time `setup()` with warm-starting enabled, and exposes `solve()` for per-cycle updates. |
| `mpc_controller.py` | `MPCController` — top-level class combining `prediction.py`, `soft_constraints.py`, and `qp_solver.py` into a single `step(x, x_ref, v_lead)` interface used by the control loop. |
| `main.py` | Application entry point. Initializes the controller and UART link, starts the listener thread, and runs the real-time 50 ms loop: read ESP32 feedback → read ultrasonic gap → solve MPC step → send command frame → log → sleep to maintain timing. |

---

## Control System Summary

- **Plant model:** second-order discrete LTI state-space system
  - States: `x1` = inter-vehicle gap [m], `x2` = ego velocity [m/s]
  - Input: `u` = commanded longitudinal acceleration [m/s²]
  - Disturbance: `v_lead` = lead vehicle velocity [m/s]
- **Prediction horizon (Np):** 20 steps
- **Control horizon (Nc):** 5 steps
- **Sampling time (Ts):** 0.05 s (20 Hz)
- **Solver:** OSQP (ADMM-based), warm-started every cycle
- **Hard constraints:** acceleration bounds, jerk (rate-of-change) bounds
- **Soft constraints:** minimum safe headway (constant time-gap model),
  velocity upper/lower bounds
- **Terminal cost:** DARE solution, guarantees closed-loop stability

---

## Mathematical Formulation

### 1. Plant Model (discrete-time, `config.py`)

State vector `x = [d, v]ᵀ` (gap distance, ego velocity), input `u = a`
(commanded acceleration), disturbance `v_lead` (lead vehicle velocity):

```
x_{k+1} = A·x_k + B·u_k + Bd·v_lead,k
```

With sampling time `Ts = 0.05 s`:

```
A  = [ 1   -Ts ]        B  = [ -0.5·Ts² ]        Bd = [ Ts ]
     [ 0    1  ]             [    Ts    ]             [ 0  ]
```

### 2. MPC Cost Function

At each step `k`, the controller solves:

```
min_U  Σ_{i=0}^{Np-1} [ xᵢᵀQxᵢ + uᵢᵀRuᵢ ]  +  x_{Np}ᵀ P x_{Np}

s.t.   x_{i+1} = A xᵢ + B uᵢ
       a_min ≤ uᵢ ≤ a_max ,  i = 0 … Nc-1
       |Δuᵢ| ≤ Δu_max
```

Implemented weights (`config.py`):

```
Q = diag(250.0, 100.0)      # gap-distance weight, velocity weight
R = diag(0.1)                # acceleration effort weight
Np = 20   (1.0 s look-ahead)
Nc = 5    (0.25 s of free control moves)
```

### 3. Terminal Cost — Discrete Algebraic Riccati Equation (`prediction.py`)

The terminal weight `P` is the unique stabilizing solution of the DARE,
solved once via `scipy.linalg.solve_discrete_are`:

```
P = AᵀPA − AᵀPB(BᵀPB + R)⁻¹BᵀPA + Q
```

`P` replaces `Q` at the final prediction step, providing an infinite-horizon
stability guarantee for the finite-horizon MPC cost.

### 4. Condensed Prediction Form (`prediction.py`)

The state trajectory over the horizon is expressed as an affine function of
the decision variables `U`:

```
X = Fx·x_k + Gu·U
```

- `Fx` (free response): stacked powers of `A`
- `Gu` (forced response): block convolution of `A` and `B`

The resulting QP Hessian and gradient:

```
H = GuᵀQ̄Gu + R̄            (built once, since it depends only on A, B, Q, R)
f = GuᵀQ̄(Fx·x_k − X_ref)  (recomputed every step from the current state)
```

`Q̄` is block-diagonal with `Q` on the diagonal and `P` in the final block;
`R̄ = I_Nc ⊗ R`. `H` is explicitly symmetrized: `H ← (H + Hᵀ)/2`.

### 5. Hard Constraints (`config.py`, enforced in `qp_solver.py`)

Actuator saturation:

```
a_min = −0.05 m/s²  ≤  u_k  ≤  a_max = 0.10 m/s²
```

Jerk (rate-of-change) limiting:

```
|Δu_k| = |u_k − u_{k−1}|  ≤  Δu_max = 0.15 m/s²
```

The rate constraint is implemented via a first-difference matrix `D_diff`,
with the previous control action `u_prev` folded into the bounds at the
first step of every solve.

### 6. Soft Constraints (`soft_constraints.py`)

**Headway** — constant time-gap (CTG) safety model:

```
d_safe(v) = D_default + t_gap · v_ego         (D_default = 0.2 m, t_gap = 0.15 s)

ε1(i) = max(0, d_safe + t_gap·v(i) − d(i))
Penalty_headway = ρ_headway · ε1(i)²           (ρ_headway = 1500)
```

**Velocity bounds:**

```
ε2(i) = max(0, v(i) − v_ref)        Penalty_upper = ρ_velocity · ε2(i)²
ε3(i) = max(0, −v(i))               Penalty_lower = ρ_velocity · ε3(i)²
                                      (ρ_velocity = 500)
```

Soft-constraint gradient contributions are added analytically to the base
gradient before every solve:

```
f_total = f_base + f_soft
```

This preserves a constant Hessian `H` across all cycles, allowing OSQP to
warm-start from the previous solution.

### 7. Lead-Vehicle Velocity Estimation (`main.py`)

From successive ultrasonic gap readings, exponentially smoothed:

```
v_rel            = (d_actual,k − d_actual,k−1) / Ts
v_lead_calc      = max(0, v_ego + v_rel)
v_lead_now       = α·v_lead_calc + (1 − α)·v_lead_now      (α = 0.25)
```

### 8. QP Solve (`qp_solver.py`)

OSQP solves the standard-form QP each cycle:

```
min_U   ½ Uᵀ H U + f_totalᵀ U
s.t.    l ≤ A_sp U ≤ u
```

where `A_sp` stacks the box-constraint identity block and the rate-of-change
difference block, and `l`/`u` are updated every cycle around the shifting
`u_prev` bound for the rate constraint. Only the first control action `u_0`
from the optimal sequence `U*` is applied to the plant (receding-horizon
principle); the process repeats at the next 50 ms step.

---

## Data Flow (per control cycle)

1. **UART RX (non-blocking):** the background listener thread has already
   parsed any new ESP32 telemetry frame into `shared["v_actual"]`; the main
   loop snapshots it under `esp_lock`.
2. **Gap sensing:** `get_real_distance()` reads the HC-SR04; if valid, gap and
   lead-vehicle velocity estimates are updated (exponential smoothing).
3. **MPC solve:** `MPCController.step()` builds the QP gradient (tracking +
   soft-constraint terms) and calls `OSQPSolver.solve()` for the optimal
   control sequence.
4. **UART TX:** the resulting ego-speed command, combined with the cut-in /
   lane-curvature flag byte (read from `/tmp/adas_output.txt`), is packed into
   a `[SOF | speed f32 | flags | EOF]` frame and sent to the ESP32.
5. **Timing:** the loop sleeps to hold a fixed 50 ms period using absolute
   scheduling (`next_loop += Ts`), so solver/logging time does not cause
   period drift.

---

## Requirements

```
numpy
scipy
osqp
pyserial
gpiozero
```

Hardware-specific dependencies (`gpiozero`, `pyserial`) are only required at
runtime on the Raspberry Pi; the QP/prediction modules have no hardware
dependency and can be unit-tested independently.

## Running

All modules must remain in the same directory (they import each other by
module name):

```bash
python3 main.py
```

The loop runs until interrupted (`Ctrl+C`), at which point it sends a final
zero-speed stop command to the ESP32, signals the listener thread to exit,
and closes the UART port cleanly.

---

## Notes

- **Threading model:** UART reception runs on a dedicated background thread
  (`esp_listener`) so that a slow or delayed ESP32 reply never stalls the
  50 ms MPC cycle. All shared state is protected by `esp_lock`.
- **Fallback behavior:** if the ultrasonic sensor reading is invalid, or no
  fresh ESP32 telemetry has arrived, the controller falls back to its own
  kinematic model estimate rather than stalling.
- **Tuning:** all cost weights, horizons, and constraint bounds live in
  `config.py` — this is the only file that should need editing for
  controller re-tuning.
