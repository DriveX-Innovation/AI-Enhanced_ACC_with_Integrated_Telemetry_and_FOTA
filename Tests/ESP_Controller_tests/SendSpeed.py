"""
send_ms.py — Motor Controller Test
- COM8 : sends speed in m/s as 6-byte frame [0xAA][float32 LE][0x55]  (matches task_UART_RX)
- COM5 : reads ESP UART0 debug logs (printf + ESP_LOGI)
- Steps RPM 0→140 in increments of 20, then back down, every 3s
"""

import math, struct, time, threading, serial
from collections import deque

# ── Physical constant ──────────────────────────────────────────────────────
R_METERS   = 0.034
RPM_TO_MS  = (2 * math.pi * R_METERS) / 60   # 0.0035604717

# ── Serial ports ───────────────────────────────────────────────────────────
DATA_PORT  = "COM8"    # TTL → ESP UART2 RX  (speed commands)
LOG_PORT   = "COM5"    # USB → ESP UART0     (printf / ESP_LOGI)
BAUD_RATE  = 115200

# ── Frame format (must match task_UART_RX exactly) ─────────────────────────
HEADER = 0xAA
FOOTER = 0x55

def build_frame(speed_ms: float) -> bytes:
    """Pack one 6-byte frame: [0xAA][float32 LE][0x55]"""
    return struct.pack("<Bf B", HEADER, speed_ms, FOOTER)   # 1+4+1 = 6 bytes

# ── Step settings ──────────────────────────────────────────────────────────
MIN_RPM     = 0
MAX_RPM     = 140
STEP_SIZE   = 20
STEP_PERIOD = 3.0    # seconds between RPM steps
SEND_PERIOD = 0.01   # 10 ms send rate (keeps ESP watchdog happy, watchdog=2000ms)

# ── Shared state ───────────────────────────────────────────────────────────
current_rpm = 0
lock        = threading.Lock()
running     = True


# ══════════════════════════════════════════════════════════════════════════
#  THREAD 1 — Speed sender (COM8)
#  Sends [0xAA][float32 speed_ms LE][0x55] every 10ms
#  Steps RPM every STEP_PERIOD seconds
# ══════════════════════════════════════════════════════════════════════════
def speed_sender(ser: serial.Serial):
    global current_rpm, running

    direction      = 1
    last_step_time = time.perf_counter()
    send_count     = 0

    print(f"\n[SENDER] Started on {DATA_PORT}")
    print(f"[SENDER] Frame format: [0xAA][float32 LE speed_ms][0x55] = 6 bytes")
    print(f"[SENDER] Refresh every {SEND_PERIOD*1000:.0f}ms | Step every {STEP_PERIOD}s\n")

    while running:
        t0 = time.perf_counter()

        with lock:
            rpm = current_rpm

        speed_ms = rpm * RPM_TO_MS
        frame    = build_frame(speed_ms)

        try:
            ser.write(frame)
            ser.flush()
        except serial.SerialException as e:
            print(f"[SENDER] Write error: {e}")
            running = False
            break

        send_count += 1

        # ── Debug: print every 50 sends (~0.5s) ───────────────────────────
        if send_count % 50 == 1:
            print(f"[SENDER] #{send_count:5d} | {rpm:3d} RPM | "
                  f"{speed_ms:.4f} m/s | "
                  f"frame={list(frame)}")

        # ── Step logic ─────────────────────────────────────────────────────
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

            print(f"\n{'='*55}")
            print(f"[STEP]  {rpm} RPM  →  {next_rpm} RPM  "
                  f"({next_rpm * RPM_TO_MS:.4f} m/s)")
            print(f"{'='*55}\n")

        # ── Keep 10ms period ──────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        sleep   = SEND_PERIOD - elapsed
        if sleep > 0:
            time.sleep(sleep)


# ══════════════════════════════════════════════════════════════════════════
#  THREAD 2 — Log reader (COM5)
#  Prints everything from ESP UART0 (printf + ESP_LOGI)
#  Highlights key debug tags for easy reading
# ══════════════════════════════════════════════════════════════════════════
def log_reader(ser: serial.Serial):
    global running

    # Tags we want to highlight
    HIGHLIGHT = ["SENSOR_TASK", "LCD_TASK", "UART_RX", "UART_TX",
                 "MOTOR_TASK", "RTOS_INIT", "INIT", "ERROR", "WARN",
                 "Actual:", "V:", "T:"]

    print(f"[READER] Started on {LOG_PORT} — printing all ESP logs\n")

    while running:
        try:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Highlight important lines
            prefix = "       "
            for tag in HIGHLIGHT:
                if tag in line:
                    prefix = "  >>   "
                    break

            print(f"{prefix}{line}")

        except serial.SerialException as e:
            print(f"[READER] Read error: {e}")
            running = False
            break


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    global running

    # ── Open COM8 (speed commands) ─────────────────────────────────────────
    try:
        ser_data = serial.Serial(
            DATA_PORT, BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"[OK] {DATA_PORT} opened — speed command port")
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {DATA_PORT}: {e}")
        return

    # ── Open COM5 (ESP logs) ───────────────────────────────────────────────
    try:
        ser_log = serial.Serial(
            LOG_PORT, BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"[OK] {LOG_PORT} opened  — ESP log port")
        print(f"[!]  Close PlatformIO serial monitor on {LOG_PORT} before running!\n")
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {LOG_PORT}: {e}")
        print(f"        Close the serial monitor on {LOG_PORT} first.")
        ser_data.close()
        return

    # ── Start threads ──────────────────────────────────────────────────────
    t1 = threading.Thread(target=speed_sender, args=(ser_data,), daemon=True)
    t2 = threading.Thread(target=log_reader,   args=(ser_log,),  daemon=True)
    t1.start()
    t2.start()

    print("[MAIN]  Running — press Ctrl+C to stop\n")

    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[STOP]  Ctrl+C received")
    finally:
        running = False
        time.sleep(0.2)
        ser_data.close()
        ser_log.close()
        print("[OK]    Ports closed. Done.")


if __name__ == "__main__":
    main()