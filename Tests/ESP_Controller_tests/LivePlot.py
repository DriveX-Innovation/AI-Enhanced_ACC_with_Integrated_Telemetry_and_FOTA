"""
Live plot: Actual speed (encoder feedback) vs Received speed (setpoint from RPi/MPC)

Reads ESP32 debug lines of the form:
    Actual:  140 rpm --> 0.4985 m/s     Received: 0.5982 m/s -->  168 rpm

Requirements:
    pip install pyserial matplotlib

Usage:
    Just run the script while the ESP32 debug UART is streaming on the configured port.
"""

import re
import time
from collections import deque

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- CONFIG ----------------
COM_PORT = "COM5"
BAUD_RATE = 115200          # must match the ESP32 debug UART baud rate
UPDATE_INTERVAL_MS = 100     # how often the plot redraws
LOG_TO_CSV = True
CSV_PATH = "speed_log.csv"
# -----------------------------------------

LINE_PATTERN = re.compile(
    r"Actual:\s*(-?\d+)\s*rpm\s*-->\s*(-?[\d.]+)\s*m/s\s*"
    r"Received:\s*(-?[\d.]+)\s*m/s\s*-->\s*(-?\d+)\s*rpm"
)

# Full-history buffers (nothing is ever dropped)
timestamps = deque()
actual_speed = deque()
received_speed = deque()

ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.05)
start_time = time.time()

csv_file = None
if LOG_TO_CSV:
    csv_file = open(CSV_PATH, "w", encoding="utf-8")
    csv_file.write("time_s,actual_mps,received_mps\n")


def read_available_lines():
    """Drain everything currently sitting in the serial buffer."""
    new_points = []
    while ser.in_waiting:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue
        match = LINE_PATTERN.search(raw)
        if match:
            _actual_rpm, actual_mps, received_mps, _received_rpm = match.groups()
            t = time.time() - start_time
            new_points.append((t, float(actual_mps), float(received_mps)))
    return new_points


fig, ax = plt.subplots(figsize=(10, 5))
line_actual, = ax.plot([], [], label="Actual speed (m/s)", color="tab:blue")
line_received, = ax.plot([], [], label="Received speed / setpoint (m/s)", color="tab:orange")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Speed (m/s)")
ax.set_title("Actual vs Received Speed - Live")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)


def update(_frame):
    new_points = read_available_lines()
    if new_points:
        for t, a, r in new_points:
            timestamps.append(t)
            actual_speed.append(a)
            received_speed.append(r)
            if csv_file:
                csv_file.write(f"{t:.3f},{a},{r}\n")
        if csv_file:
            csv_file.flush()

    if timestamps:
        line_actual.set_data(timestamps, actual_speed)
        line_received.set_data(timestamps, received_speed)

        t_max = timestamps[-1]
        ax.set_xlim(0, t_max + 0.5)

        y_all = list(actual_speed) + list(received_speed)
        if y_all:
            y_margin = 0.1 * (max(y_all) - min(y_all) + 0.01)
            ax.set_ylim(min(y_all) - y_margin, max(y_all) + y_margin)

    return line_actual, line_received


ani = animation.FuncAnimation(fig, update, interval=UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False)

try:
    plt.tight_layout()
    plt.show()
finally:
    ser.close()
    if csv_file:
        csv_file.close()