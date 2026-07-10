"""
send_rpm.py — Stage 2  (final)
- COM8 : sends current RPM every 10ms (continuous refresh), steps value every 3s
- COM5 : reads "target,actual\n" CSV, skips ESP_LOGI lines
- Live rolling plot
"""

import struct, time, threading, serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ── Serial config ──────────────────────────────────────────────────────────
DATA_PORT   = "COM8"
LOG_PORT    = "COM5"
BAUD_RATE   = 115200

# ── RPM step settings ──────────────────────────────────────────────────────
MIN_RPM      = 0
MAX_RPM      = 140
STEP_SIZE    = 20
STEP_PERIOD  = 3.0    # seconds between value changes
SEND_PERIOD  = 0.01   # 10 ms — continuous refresh to keep ESP queue alive

# ── Plot ───────────────────────────────────────────────────────────────────
PLOT_WINDOW  = 600
t_buf        = deque(maxlen=PLOT_WINDOW)
target_buf   = deque(maxlen=PLOT_WINDOW)
actual_buf   = deque(maxlen=PLOT_WINDOW)

current_rpm  = 0
frame_count  = 0
lock         = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
#  THREAD 1 — RPM sender  (COM8)
#  Sends current_rpm every 10 ms.
#  Steps the value every STEP_PERIOD seconds independently.
# ══════════════════════════════════════════════════════════════════════════
def rpm_sender(ser_data: serial.Serial):
    global current_rpm
    direction        = 1
    last_step_time   = time.perf_counter()
    send_count       = 0

    print(f"[SENDER] Started — refresh every {SEND_PERIOD*1000:.0f}ms, "
          f"step every {STEP_PERIOD}s, step={STEP_SIZE} RPM")

    while True:
        t0 = time.perf_counter()

        with lock:
            rpm = current_rpm

        frame   = struct.pack("<h", rpm)
        ser_data.write(frame)
        ser_data.flush()
        send_count += 1

        # ── step logic: change value every STEP_PERIOD seconds ────────────
        now = time.perf_counter()
        if now - last_step_time >= STEP_PERIOD:
            last_step_time = now

            next_rpm = current_rpm + direction * STEP_SIZE
            if next_rpm > MAX_RPM:
                next_rpm  = MAX_RPM
                direction = -1
            elif next_rpm < MIN_RPM:
                next_rpm  = MIN_RPM
                direction = 1

            with lock:
                current_rpm = next_rpm

            print(f"[SENDER] step → {next_rpm:3d} RPM  "
                  f"(frame #{send_count}  raw={list(frame)})")

        # ── keep 10 ms period ─────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        sleep   = SEND_PERIOD - elapsed
        if sleep > 0:
            time.sleep(sleep)


# ══════════════════════════════════════════════════════════════════════════
#  THREAD 2 — CSV reader  (COM5)
#  Skips ESP_LOGI lines (start with a letter), parses "int,int\n"
# ══════════════════════════════════════════════════════════════════════════
def csv_reader(ser_log: serial.Serial):
    global frame_count
    t_start   = time.perf_counter()
    bad_lines = 0
    log_lines = 0

    print(f"[READER] Started on {LOG_PORT} — filtering ESP_LOGI lines")

    while True:
        try:
            raw = ser_log.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()

            # Skip any line that doesn't start with a digit (ESP log lines)
            if not line or not line[0].isdigit():
                log_lines += 1
                continue

            parts = line.split(",")
            if len(parts) != 2:
                bad_lines += 1
                continue

            target = int(parts[0])
            actual = int(parts[1])
            t_now  = time.perf_counter() - t_start

            t_buf.append(t_now)
            target_buf.append(target)
            actual_buf.append(actual)
            frame_count += 1

            if frame_count % 200 == 0:
                print(f"[READER] samples={frame_count:6d} | "
                      f"target={target:3d} | actual={actual:3d} | "
                      f"skipped_logs={log_lines} | bad_csv={bad_lines}")

        except (ValueError, UnicodeDecodeError):
            bad_lines += 1


# ══════════════════════════════════════════════════════════════════════════
#  LIVE PLOT
# ══════════════════════════════════════════════════════════════════════════
def animate(_frame, ax, line_target, line_actual, txt_status):
    if len(t_buf) < 2:
        return
    ts  = list(t_buf)
    tgt = list(target_buf)
    act = list(actual_buf)
    line_target.set_data(ts, tgt)
    line_actual.set_data(ts, act)
    x_end = ts[-1]
    ax.set_xlim(max(0.0, x_end - 6.0), x_end + 0.1)
    ax.set_ylim(-5, MAX_RPM + 20)
    with lock:
        rpm = current_rpm
    txt_status.set_text(f"Setpoint: {rpm} RPM\nSamples : {frame_count}")
    ax.set_title("Motor Speed — PID Tracking", fontsize=12)


def main():
    try:
        ser_data = serial.Serial(DATA_PORT, BAUD_RATE,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE, timeout=1)
        print(f"[OK] {DATA_PORT} — RPM command port (UART2)")
    except serial.SerialException as e:
        print(f"[ERROR] {DATA_PORT}: {e}"); return

    try:
        ser_log = serial.Serial(LOG_PORT, BAUD_RATE,
                                bytesize=serial.EIGHTBITS,
                                parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE, timeout=2)
        print(f"[OK] {LOG_PORT} — CSV+log port (UART0)")
        print(f"[!]  Keep serial monitor CLOSED on {LOG_PORT} while this runs")
    except serial.SerialException as e:
        print(f"[ERROR] {LOG_PORT}: {e}\n"
              f"       Close the serial monitor on {LOG_PORT} first.")
        ser_data.close(); return

    threading.Thread(target=rpm_sender, args=(ser_data,), daemon=True).start()
    threading.Thread(target=csv_reader, args=(ser_log,),  daemon=True).start()

    fig, ax = plt.subplots(figsize=(11, 5))
    line_target, = ax.plot([], [], "b-",  linewidth=2.0, label="Target RPM")
    line_actual,  = ax.plot([], [], "r-", linewidth=1.5, label="Actual RPM")
    txt_status = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                         fontsize=10, va="top", ha="right",
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (RPM)")
    ax.set_ylim(-5, MAX_RPM + 20)
    ax.set_xlim(0, 6)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(fig, animate,
                                  fargs=(ax, line_target, line_actual, txt_status),
                                  interval=100, cache_frame_data=False)
    _ = ani  # keep reference alive — prevents garbage collection
    plt.tight_layout()
    print("\n[PLOT] Window open — Ctrl+C or close window to stop\n")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[STOP] Total samples: {frame_count}")
        ser_data.close()
        ser_log.close()
        print(f"[OK]  Ports closed.")


if __name__ == "__main__":
    main()