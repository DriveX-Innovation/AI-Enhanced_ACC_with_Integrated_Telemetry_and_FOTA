# 🚗 MPC Module (OSQP)

A real-time longitudinal **Model Predictive Controller** for the AI-Enhanced
Adaptive Cruise Control system.

- 🧠 Runs on the **Raspberry Pi 5** (upper layer)
- ⚡ Solves a constrained QP every **100 ms** using **OSQP**
- 🔌 Sends the resulting speed command to the **ESP32** (lower-layer PID) over UART
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
### 🔍 Deep dive — function by function

| Constant | Value | What it controls |
|---|---|---|
| `Ts` | 0.1 s | Control loop period (100 ms) |
| `A`, `B`, `Bd` | matrices | Discrete-time plant dynamics |
| `Np`, `Nc` | 20, 5 | Prediction / control horizon lengths |
| `Q`, `R` | diag(250, 100), diag(0.1) | Tracking-error vs. effort weighting |
| `rho_headway`, `rho_velocity` | 1500, 500 | Soft-constraint penalty strength |
| `a_min`, `a_max` | −0.05, 0.10 m/s² | Hard acceleration bounds |
| `da_max` | 0.15 m/s² | Max change in acceleration per step (jerk limit) |
| `d_safe`, `tg` | 0.2 m, 0.15 s | Base safe distance + time-gap for the CTG model |
| `SOF`, `EOF` | 0xAA, 0x55 | UART frame start/end delimiters |
| `BAUDRATE`, `TIMEOUT` | 115200, 0.1 s | UART link speed and read timeout |


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
 
## 🔌 UART Wire Protocol
 
Both directions use the same simple framed format:
 
**Command frame (Raspberry Pi → ESP32):**
 
| Byte(s) | Field | Type | Meaning |
|---|---|---|---|
| 0 | `SOF` | `0xAA` | Start of frame |
| 1–4 | speed | `float32` (little-endian) | Target ego speed [m/s] |
| 5 | flags | `uint8` | Bit 0 = cut-in flag, Bit 1 = lane-curvature flag |
| 6 | `EOF` | `0x55` | End of frame |
 
**Telemetry frame (ESP32 → Raspberry Pi):**
 
| Byte(s) | Field | Type | Meaning |
|---|---|---|---|
| 0 | `SOF` | `0xAA` | Start of frame |
| 1–4 | speed | `float32` (little-endian) | Measured wheel speed [m/s] |
| 5 | `EOF` | `0x55` | End of frame |
 
- 📶 Baud rate: **115,200 bps**
- ⏱️ Read timeout: **100 ms** (lets the listener thread poll the shutdown flag regularly)
- 🧪 Every incoming frame is validated: correct `SOF`/`EOF`, correct length, and speed within `[0.0, v_max]` — anything else is silently discarded and logged.
---
 
## 🗂️ File Structure
 
### 📁 Module folder layout
 
```
mpc_modules/
├── config.py                 
├── sensors.py                 
├── uart_comm.py                 
├── prediction.py                  
├── soft_constraints.py              
├── qp_solver.py                       
├── mpc_controller.py                   
├── main.py                                                                              
└── README.md                                
```
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
## 🩺 Troubleshooting
 
- ❌ **"Cannot open ESP32 UART"** — check the ESP32 is wired to `/dev/ttyAMA0`
  (in `config.py`), and that the Pi's serial console is disabled (it competes
  for the same UART pins).
- 🕳️ **Gap always shows "MODEL" instead of "SENSOR" in the logs** — the
  HC-SR04 is returning `None`. Check wiring on GPIO 23 (trigger) / GPIO 24
  (echo), and make sure the target is within 0.05–4.0 m.
- 🐌 **"Loop overrun" warnings** — the 100 ms budget was missed, usually
  because OSQP took too long to converge or a blocking call snuck into the
  main loop. Check `max_iter` in `qp_solver.py` and confirm nothing besides
  `esp_listener` is reading from the UART port.
- 🚫 **ESP32 never receiving frames** — verify the flag byte encoding
  (`cut_in` in bit 0, `lane_curve` in bit 1) matches what the ESP32 firmware
  expects, and double check the baud rate matches on both ends.
- 🧊 **Speed stuck at 0** — the software watchdog on the ESP32 side zeroes
  the target if it hasn't seen a valid frame in a while; confirm
  `send_uart_ESP()` is actually being called every cycle (check the log line
  for the cut-in/lane-curve print statement).
---



## 📝 Notes

- 🧵 **Threading:** UART reads happen on a background thread, so a slow
  ESP32 reply never stalls the 100 ms control loop.
- 🛟 **Fallbacks:** bad sensor reading or missing ESP32 reply? The
  controller just trusts its own model instead of freezing up.
- 🎚️ **Tuning:** everything you'd want to retune lives in `config.py` —
  that's the only file you should normally need to touch.
