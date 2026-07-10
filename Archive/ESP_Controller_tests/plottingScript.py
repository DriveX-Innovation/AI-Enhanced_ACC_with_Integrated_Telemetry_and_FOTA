import collections
import threading
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import serial

# ──────────────────────────────────────────────
#  ★  CHANGE THESE AS NEEDED
# ──────────────────────────────────────────────
COM_PORT    = "COM5"
BAUD_RATE   = 115200
WINDOW_SIZE = 300
# ──────────────────────────────────────────────

UPDATE_INTERVAL = 50
RECONNECT_DELAY = 2

lock       = threading.Lock()
t_data     = collections.deque(maxlen=WINDOW_SIZE)
meas_data  = collections.deque(maxlen=WINDOW_SIZE)
out_data   = collections.deque(maxlen=WINDOW_SIZE)
status_msg = ["Connecting..."]
sample_idx = [0]
paused     = [False]


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def serial_reader():
    while True:
        try:
            with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
                with lock:
                    status_msg[0] = f"Connected  {COM_PORT} @ {BAUD_RATE}"
                print(f"[serial] connected to {COM_PORT}")
                while True:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace")
                    parsed = parse_line(line)
                    if parsed is None:
                        continue
                    meas, out = parsed
                    if meas > 60000:
                        continue
                    with lock:
                        if not paused[0]:   # only store when not paused
                            t_data.append(sample_idx[0])
                            meas_data.append(meas)
                            out_data.append(out)
                            sample_idx[0] += 1

        except serial.SerialException as e:
            with lock:
                status_msg[0] = f"Disconnected - retrying in {RECONNECT_DELAY}s"
            print(f"[serial] {e} - reconnecting...")
            time.sleep(RECONNECT_DELAY)


def calc_stats(data):
    if not data:
        return 0, 0, 0
    return min(data), max(data), sum(data) / len(data)


def main():
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    # ── Figure layout ─────────────────────────
    # Leave space at bottom for buttons
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.18)

    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")
    ax.tick_params(colors="#cdd6f4")
    ax.yaxis.label.set_color("#cdd6f4")
    ax.xaxis.label.set_color("#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")

    line_meas, = ax.plot([], [], color="#a6e3a1", lw=0.9, label="Throttle Speed (RPM)")
    line_out,  = ax.plot([], [], color="#f38ba8", lw=0.9, label="Current Speed (RPM)")

    ax.set_ylabel("Value")
    ax.set_xlabel("Sample")
    ax.set_ylim(-5, 230)
    ax.grid(True, color="#313244", linewidth=0.4)
    ax.legend(loc="upper left", facecolor="#313244",
              labelcolor="#cdd6f4", framealpha=0.8)

    title_txt = ax.set_title("PID Response", color="#cdd6f4", fontsize=13, pad=8)

    stats_box = ax.text(
        0.99, 0.97, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        color="#cdd6f4", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#313244", alpha=0.8),
    )
    status_txt = fig.text(
        0.5, 0.01, "", ha="center",
        fontsize=9, color="#6c7086", family="monospace"
    )

    # ── Pause / Resume buttons ────────────────
    ax_pause  = fig.add_axes([0.38, 0.05, 0.1, 0.06])
    ax_resume = fig.add_axes([0.52, 0.05, 0.1, 0.06])

    btn_pause = Button(ax_pause,  "Pause",
                       color="#313244", hovercolor="#45475a")
    btn_resume = Button(ax_resume, "Resume",
                        color="#313244", hovercolor="#45475a")

    for btn in (btn_pause, btn_resume):
        btn.label.set_color("#cdd6f4")
        btn.label.set_fontsize(10)

    def on_pause(event):
        with lock:
            paused[0] = True
        status_txt.set_text(f"PAUSED  —  {COM_PORT} @ {BAUD_RATE}")
        status_txt.set_color("#f38ba8")

    def on_resume(event):
        with lock:
            paused[0] = False
        status_txt.set_color("#6c7086")

    btn_pause.on_clicked(on_pause)
    btn_resume.on_clicked(on_resume)

    # ── Animation ─────────────────────────────
    def update(_frame):
        with lock:
            t    = list(t_data)
            meas = list(meas_data)
            out  = list(out_data)
            msg  = status_msg[0]
            is_paused = paused[0]

        if not t:
            return line_meas, line_out

        line_meas.set_data(t, meas)
        line_out.set_data(t, out)

        x_min = t[0]
        x_max = max(t[-1], t[0] + WINDOW_SIZE)
        ax.set_xlim(x_min, x_max)

        mn_m, mx_m, avg_m = calc_stats(meas)
        mn_o, mx_o, avg_o = calc_stats(out)

        stats_box.set_text(
            f"measured  min={mn_m:.0f}  max={mx_m:.0f}  avg={avg_m:.1f}\n"
            f"output    min={mn_o:.1f}  max={mx_o:.1f}  avg={avg_o:.1f}"
        )
        title_txt.set_text(f"PID Response  -  {len(t)} samples")

        if not is_paused:
            status_txt.set_text(msg)
            status_txt.set_color("#6c7086")

        return line_meas, line_out

    ani = animation.FuncAnimation(
        fig, update, interval=UPDATE_INTERVAL,
        blit=False, cache_frame_data=False,
    )
    _ = ani
    plt.show()


main()