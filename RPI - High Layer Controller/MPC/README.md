# 🚗 MPC Module (OSQP)

A real-time longitudinal **Model Predictive Controller** for the AI-Enhanced
Adaptive Cruise Control system.

- 🧠 Runs on the **Raspberry Pi 5** (upper layer)
- ⚡ Solves a constrained QP every **100 ms** using **OSQP**
- 🔌 Sends the resulting speed command to the **ESP32** (lower-layer PID) over UART

This folder is the modularized version of the original monolithic
`mpc_osqp.py` script. Nothing about the logic, tuning, or control flow has
changed — it's just been split into clean, single-responsibility files so
it's easier to read, test, and maintain. 🧩

---

## 🗺️ Architecture

```
                                config.py
                              (all constants)
                                    │
        ┌────────────────┬──────────┴──────────┬────────────────────┐
        ▼                ▼                     ▼                    ▼
   sensors.py       uart_comm.py          prediction.py     soft_constraints.py
   (HC-SR04)         (ESP32 link)          (DARE + Fx/Gu)     (headway / velocity)
        │                │                     │                    │
        └────────────────┴───────────┬─────────┴────────────────────┘
                                     ▼
                                qp_solver.py
                                (OSQPSolver)
                                     │
                                     ▼
                             mpc_controller.py
                             (MPCController.step)
                                     │
                                     ▼
                                  main.py
                             (100 ms control loop)
```
---

## 📦 Module Reference

Quick cheat sheet — what each file is for, in one line:

- ⚙️ **`config.py`** — every tuning number and constant lives here (matrices, weights, bounds, UART settings). One file to edit if you want to re-tune anything.
- 📡 **`sensors.py`** — reads the HC-SR04 ultrasonic sensor for the gap distance. Returns `None` if the reading is bad so the loop can fall back safely.
- 🔗 **`uart_comm.py`** — talks to the ESP32. Sends speed commands, and runs a background thread that listens for telemetry without blocking the control loop.
- 📐 **`prediction.py`** — one-time setup math: solves the DARE for the terminal cost, and builds the prediction matrices used every cycle.
- 🛟 **`soft_constraints.py`** — adds "soft" safety nudges for headway and velocity, so the QP stays solvable even in extreme situations.
- 🧮 **`qp_solver.py`** — wraps OSQP itself: builds the QP, applies acceleration/jerk limits, and solves it every cycle.
- 🎛️ **`mpc_controller.py`** — the brain. Combines prediction + constraints + solver into one simple `step()` call.
- ▶️ **`main.py`** — the entry point. Runs the live 100 ms loop: read sensors → solve MPC → send command → repeat.

---

## 🎯 Control System Summary

- **Plant model:** simple 2-state system
  - `x1` = gap to lead vehicle [m] 📏
  - `x2` = ego vehicle speed [m/s] 🏎️
  - `u` = commanded acceleration [m/s²] 🎚️
  - `v_lead` = lead vehicle speed (disturbance) [m/s] 🚙
- **Look-ahead (Np):** 20 steps
- **Free control moves (Nc):** 5 steps
- **Loop rate (Ts):** 0.1 s → **100 ms**, ~10 Hz
- **Solver:** OSQP, warm-started every cycle for speed ⚡
- **Hard limits:** acceleration range + jerk (smoothness) limit 🚧
- **Soft limits:** safe following distance + speed range 🛡️
- **Stability guarantee:** DARE-based terminal cost ✅

---

## 🧮 The Math, Simplified

### 1️⃣ Plant model (`config.py`)

State update each step:

```
x_{k+1} = A·x_k + B·u_k + Bd·v_lead,k
```

```
A  = [ 1   -Ts ]        B  = [ -0.5·Ts² ]        Bd = [ Ts ]
     [ 0    1  ]             [    Ts    ]             [ 0  ]
```

### 2️⃣ What the controller is trying to minimize

```
min_U  Σ [ xᵢᵀQxᵢ + uᵢᵀRuᵢ ]  +  x_Np ᵀ P x_Np

s.t.   x_{i+1} = A xᵢ + B uᵢ
       a_min ≤ uᵢ ≤ a_max
       |Δuᵢ| ≤ Δu_max
```

```
Q = diag(250.0, 100.0)   # care more about gap, then speed
R = diag(0.1)             # don't overreact with acceleration
Np = 20  |  Nc = 5
```

### 3️⃣ Terminal cost — keeps it stable (`prediction.py`)

```
P = AᵀPA − AᵀPB(BᵀPB + R)⁻¹BᵀPA + Q
```

Solved once via the **Discrete Algebraic Riccati Equation (DARE)** —
guarantees the controller won't go unstable over a long horizon.

### 4️⃣ Condensed prediction (`prediction.py`)

```
X = Fx·x_k + Gu·U
H = GuᵀQ̄Gu + R̄        (built once)
f = GuᵀQ̄(Fx·x_k − X_ref)   (rebuilt every cycle)
```

### 5️⃣ Hard limits (`config.py` → enforced in `qp_solver.py`)

```
a_min = −0.05 m/s²  ≤  u_k  ≤  a_max = 0.10 m/s²
|u_k − u_{k−1}|  ≤  Δu_max = 0.15 m/s²
```

### 6️⃣ Soft limits (`soft_constraints.py`)

**🛑 Safe headway** (constant time-gap model):

```
d_safe(v) = 0.2 + 0.15·v_ego
ε1 = max(0, d_safe − d)          penalty = 1500 · ε1²
```

**🚦 Velocity bounds:**

```
ε2 = max(0, v − v_ref)   →  penalty = 500 · ε2²   (too fast)
ε3 = max(0, −v)          →  penalty = 500 · ε3²   (going backward)
```

### 7️⃣ Estimating lead vehicle speed (`main.py`)

```
v_rel        = (d_k − d_{k−1}) / Ts
v_lead_calc  = max(0, v_ego + v_rel)
v_lead_now   = 0.25·v_lead_calc + 0.75·v_lead_now   # smoothed
```

### 8️⃣ Solving it (`qp_solver.py`)

```
min_U   ½ Uᵀ H U + f_totalᵀ U
s.t.    l ≤ A_sp U ≤ u
```

Only the **first** control action gets applied — then the whole thing
repeats next cycle (that's the "receding horizon" in MPC). 🔁

---

## 🔄 What Happens Every 100 ms

1. 📥 **Read ESP32 feedback** — grab the latest speed reading (non-blocking).
2. 📏 **Read the ultrasonic sensor** — update gap distance and estimate lead-car speed.
3. 🧮 **Solve the MPC step** — get the optimal next speed command.
4. 📤 **Send it to the ESP32** — packed with cut-in / lane-curvature flags.
5. 📝 **Log the cycle.**
6. ⏱️ **Sleep** just enough to hold a steady 100 ms rhythm.

---

## 🛠️ Requirements

```
numpy
scipy
osqp
pyserial
gpiozero
```

> 🖥️ Hardware libraries (`gpiozero`, `pyserial`) are only needed on the
> actual Raspberry Pi. The math modules (`prediction.py`, `qp_solver.py`,
> `soft_constraints.py`) have zero hardware dependencies and can be tested
> on any machine.

## ▶️ Running It

Keep all files in the same folder — they import each other by name:

```bash
python3 main.py
```

Press `Ctrl+C` to stop safely — it sends a final zero-speed command,
shuts down the listener thread, and closes the UART port cleanly. ✅

---

## 📝 Notes

- 🧵 **Threading:** UART reads happen on a background thread, so a slow
  ESP32 reply never stalls the 100 ms control loop.
- 🛟 **Fallbacks:** bad sensor reading or missing ESP32 reply? The
  controller just trusts its own model instead of freezing up.
- 🎚️ **Tuning:** everything you'd want to retune lives in `config.py` —
  that's the only file you should normally need to touch.
