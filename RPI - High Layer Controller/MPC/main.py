# =============================================================================
#  MODULE: main.py
#
#  DESCRIPTION:
#  Entry point of the OSQP-based MPC Adaptive Cruise Control application.
#  Wires together all other modules (config, uart_comm, sensors,
#  mpc_controller) into the real-time 50 ms control loop that runs on the
#  Raspberry Pi 5.
#
#  Responsibilities:
#    - Instantiates the MPCController (one-time DARE + prediction-matrix
#      setup) and opens the UART link to the ESP32.
#    - Launches the background esp_listener thread ([CHANGE 4]) so UART
#      reads never block the control loop.
#    - Runs the main loop, each cycle:
#        1. Snapshots the latest ESP32 speed from shared state
#           (non-blocking).
#        2. Reads the HC-SR04 ultrasonic sensor for gap distance and
#           estimates lead-vehicle speed from its rate of change
#           ([CHANGE 3]).
#        3. Calls mpc.step() to solve the QP and obtain the next control
#           command.
#        4. Reads cut-in / lane-curvature flags from a shared output
#           file and sends the combined speed + flags command frame to
#           the ESP32.
#        5. Logs the cycle state.
#        6. Sleeps to maintain a precise Ts-second loop period.
#    - On shutdown (KeyboardInterrupt or exception), signals the listener
#      thread to stop, sends a final zero-speed stop command, and closes
#      the UART port cleanly.
#
#  Note: the operator-startup-prompt behavior described in [CHANGE 1] and
#  [CHANGE 2] of the original script (entering v_ref / v_lead once at
#  startup, then receiving only v_lead updates from a PC listener) is
#  referenced in the original header comments; this module preserves the
#  same v_ref/v_lead handling as implemented in the original main loop.
# =============================================================================

import time
import logging
import threading
import numpy as np
import sys

sys.path.append('/home/drivx/adas-sys/Vehicle-CV-ADAS-master/YOLOPv2-ncnn-main/')

from config import Ts, d_safe, tg, v_ref
from uart_comm import (
    shared, esp_lock, open_all_uarts, send_uart_ESP,
    esp_listener, brake_flag_toggler
)
from sensors import get_real_distance
from mpc_controller import MPCController

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("MPC_v3")


# =============================================================================
#  SECTION 13 — MAIN LOOP
# =============================================================================
def main():
    mpc = MPCController()
    start_time = time.time()   # <-- Add this here
    ser_esp = open_all_uarts()
    if ser_esp is None:
        log.error("FATAL: Cannot open ESP32 UART. Aborting.")
        return

    ser_esp.reset_input_buffer()
    time.sleep(0.1)

    # ── [CHANGE 4] Start background UART listener thread ──────────────────
    #   daemon=True → thread dies automatically when main() exits
    listener_thread = threading.Thread(
        target=esp_listener,
        args=(ser_esp,),
        daemon=True,
        name="ESP32-Listener"
    )
    listener_thread.start()
    log.info("[MAIN] UART listener thread launched.")
    # ──────────────────────────────────────────────────────────────────────

    x          = np.array([d_safe, 0.0])
    sim_steps  = 1000
    ALPHA      = 0.25
    last_d     = None
    v_lead_now = v_ref + d_safe

    log.info("MPC v3 control loop starting...")
    next_loop = time.time()
    # last_toggle = time.time()
    send_flag = 0
    # --------------------------------------
    lane_curve_toggle = brake_flag_toggler()

    try:
        while True:

            t_start = time.time()

            # ── Step 1: Read latest ESP32 speed from shared state (NON-BLOCKING) ──
            #
            #   [CHANGE 4] Instead of calling receive_uart_ESP() which would
            #   block the loop for up to TIMEOUT seconds, we just snapshot the
            #   shared variable written by the listener thread.
            #   The lock is held for microseconds — never for the full UART wait.
            # ─────────────────────────────────────────────────────────────────────
            with esp_lock:
                v_actual = shared["v_actual"]
                is_fresh = shared["fresh"]
                shared["fresh"] = False     # consume the flag for this cycle
            speed = v_actual if is_fresh else x[1]  # fallback to model speed if no new data
            if is_fresh:
                x[1] = v_actual
                log.debug(f"[RX] Fresh ESP32 speed: {v_actual:.4f} m/s")

            # ── Step 2: Ultrasonic gap + v_lead estimation ────────────────────
            d_actual = get_real_distance()
            if d_actual is not None:
                if last_d is not None:
                    v_rel             = (d_actual - last_d) / Ts
                    v_lead_calculated = max(0.0, x[1] + v_rel)
                    v_lead_now        = ALPHA * v_lead_calculated + (1 - ALPHA) * v_lead_now
                x[0]   = d_actual
                last_d = d_actual
            # else: keep last v_lead_now and x[0]

            # ── Step 3: MPC compute ───────────────────────────────────────────
            x_meas = np.array([x[0], x[1]])
            d_ref  = d_safe + tg * x[1]
            x_ref  = np.array([d_ref, v_ref])

            u_opt, speed_out, accel_out = mpc.step(x_meas, x_ref, v_lead_now)

            if not is_fresh:
                x[1] = speed_out    # fallback: trust the model when ESP32 silent

            # ── Step 4: Send speed command x[1] to ESP32 ─────────────────────
            #
            #   [CHANGE 4] We send x[1] (the MPC ego-speed state) as the
            #   target command.  ESP32 receives this, drives its motor toward
            #   that speed, then replies with actual encoder speed → listener
            #   thread catches the reply asynchronously next cycle.
            # ─────────────────────────────────────────────────────────────────
            try:
                with open('/tmp/adas_output.txt', 'r') as f:
                    vals = f.read().strip().split(',')
                    cut_in     = int(vals[0])
                    lane_curve = int(vals[1])
            except:
                cut_in     = 0
                lane_curve = 0
            # -------------------toggle------------------
            # try:
                # with open('/tmp/adas_output.txt', 'r') as f:
                    # vals = f.read().strip().split(',')
                    # cut_in = int(vals[0])
            # except:
                # cut_in = 0

            # Toggle independently of the file lane_curve
            # lane_curve = next(lane_curve_toggle)
            # ---------------------------------------------------
            # ← ADD THIS LINE
            print(f"[MPC] cut_in={cut_in} | lane_curve={lane_curve}")

            flag_byte = (int(cut_in) & 0x01) | ((int(lane_curve) & 0x01) << 1)

            send_uart_ESP(ser_esp, float(x[1]), flag_byte)

            # ── Step 5: Log ───────────────────────────────────────────────────
            gap_ok     = x[0] >= d_safe
            gap_source = "SENSOR" if d_actual is not None else "MODEL"
            t_elapsed  = (time.time() - t_start) * 1000

            log.info(
                f"gap={x[0]:.2f}m [{gap_source}] | "
                f"v_ego={x[1]:.3f} m/s | v_lead={v_lead_now:.3f} m/s | "
                f"v_actual={speed:.3f} m/s | "
                f"gap={'OK' if gap_ok else 'VIOLATION'} | "
            )

            # ── Step 6: Precise 50ms timing ───────────────────────────────────
            next_loop += Ts
            sleep_time = next_loop - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                log.warning(f"Loop overrun: {-sleep_time*1000:.1f}ms late")
                next_loop = time.time()     # resync clock

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    except Exception as e:
        import traceback
        log.error(f"Crash: {e}\n{traceback.format_exc()}")
    finally:
        # ── [CHANGE 4] Signal listener thread to exit cleanly ─────────────
        with esp_lock:
            shared["alive"] = False

        log.info("Emergency stop — sending 0.0 m/s")
        if ser_esp and ser_esp.is_open:
            send_uart_ESP(ser_esp, 0.0, 0)
            time.sleep(0.1)
            ser_esp.close()

        listener_thread.join(timeout=1.0)
        log.info("UART closed. Listener stopped. Safe.")


if __name__ == "__main__":
    main()
