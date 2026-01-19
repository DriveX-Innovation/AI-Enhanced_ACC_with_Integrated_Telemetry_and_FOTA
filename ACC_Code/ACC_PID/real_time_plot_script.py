import serial
import matplotlib.pyplot as plt
from collections import deque

PORT = "COM5"        # change if needed
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0.1)

MAX_POINTS = 200
target_buf = deque(maxlen=MAX_POINTS)
pid_buf = deque(maxlen=MAX_POINTS)

plt.ion()
fig, ax = plt.subplots()
line_target, = ax.plot([], [], label="Target Speed")
line_pid, = ax.plot([], [], label="PID Output")

ax.set_xlabel("Samples")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True)

plt.show(block=False)

try:
    while plt.fignum_exists(fig.number):   # <<< EXIT WHEN WINDOW CLOSED
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            plt.pause(0.01)
            continue

        values = line.split(",")

        if len(values) != 2:
            continue

        try:
            target = int(values[0])
            pid = int(values[1])
        except ValueError:
            continue

        target_buf.append(target)
        pid_buf.append(pid)

        line_target.set_data(range(len(target_buf)), target_buf)
        line_pid.set_data(range(len(pid_buf)), pid_buf)

        ax.relim()
        ax.autoscale_view()

        plt.pause(0.01)

finally:
    print("Closing program...")
    ser.close()
    plt.close("all")
