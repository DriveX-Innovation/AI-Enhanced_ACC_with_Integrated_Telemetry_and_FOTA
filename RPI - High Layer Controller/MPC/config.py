# =============================================================================
#  MODULE: config.py
#
#  DESCRIPTION:
#  Central configuration module for the OSQP-based MPC Adaptive Cruise
#  Control system. Contains all fixed constants used across the other
#  modules: the sampling time, the discrete-time plant state-space
#  matrices, the MPC horizon and weighting parameters, the soft-constraint
#  penalty weights, the physical actuator/safety bounds, and the UART
#  communication settings (port, baud rate, timeout, frame delimiters).
#
#  This module performs no computation — it only defines constants that
#  are imported by plant_model.py, mpc_solver.py, mpc_controller.py,
#  uart_comm.py, and main.py. Keeping these values in one place makes
#  tuning (Q, R, horizons, constraint bounds) a single-file edit.
# =============================================================================

import numpy as np

# =============================================================================
#  SECTION 1 — SAMPLING TIME
# =============================================================================
Ts = 0.05  # [s]


# =============================================================================
#  SECTION 2 — PLANT MODEL MATRICES
# =============================================================================
A = np.array([
    [1.0,  -Ts         ],
    [0.0,   1.0        ]
])
B = np.array([
    [-0.5 * Ts**2],
    [ Ts          ]
])
Bd = np.array([
    [Ts  ],
    [0.0 ]
])
C = np.array([[0.0, 0.0]])
D = np.array([[1.0]])

nx = A.shape[0]
nu = B.shape[1]


# =============================================================================
#  SECTION 3 — MPC TUNING
# =============================================================================
Np = 20
Nc = 5

Q = np.diag([250.0, 100.0])  # left: 50 to 80 - right 25 to 40
R = np.diag([0.1])           # 0.05 to 0.1


# =============================================================================
#  SECTION 4 — SOFT CONSTRAINT WEIGHTS
# =============================================================================
rho_headway  = 1500.0
rho_velocity =  500.0


# =============================================================================
#  SECTION 5 — PHYSICAL CONSTRAINT BOUNDS
# =============================================================================
a_min  = -0.05
a_max  =  0.10
da_max =  0.15

d_safe =  0.2
tg     =  0.15
v_ref  =  0.5   # will be overwritten by startup prompt
v_min  =  0.0


# =============================================================================
#  SECTION 6 — UART CONFIGURATION
# =============================================================================
SOF      = 0xAA
EOF      = 0x55

ESP_PORT = "/dev/ttyAMA0"   # RPi UART connected to ESP32
BAUDRATE = 115200
TIMEOUT  = 0.1              # [CHANGE 4] Short timeout — listener thread loops fast
v_max    = 1.0              # sanity-check ceiling for received speeds
