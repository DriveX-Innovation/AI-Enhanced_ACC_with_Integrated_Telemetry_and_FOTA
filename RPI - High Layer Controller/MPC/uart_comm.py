# =============================================================================
#  MODULE: uart_comm.py
#
#  DESCRIPTION:
#  Handles all UART communication between the Raspberry Pi 5 (upper layer,
#  running this MPC controller) and the ESP32 (lower layer, running the
#  PID motor controller). Responsibilities:
#
#    1. Defining the shared, thread-safe state dictionary ("shared") that
#       holds the latest ESP32-reported speed and a "fresh" flag, plus the
#       "alive" flag used to signal shutdown to the background thread.
#
#    2. open_all_uarts() — opens and flushes the serial port to the ESP32.
#
#    3. send_uart_ESP() — packs and transmits a [SOF | float32 speed |
#       flags byte | EOF] command frame to the ESP32 (non-blocking write).
#
#    4. brake_flag_toggler() — a generator utility that toggles a flag
#       every 6 seconds (used for testing/demo purposes).
#
#    5. esp_listener() — the background thread target function. Runs in
#       its own thread (see [CHANGE 4] in the original script) so that
#       UART reads never block the 50 ms MPC control loop. It continuously
#       drains the UART RX buffer, validates incoming telemetry frames
#       from the ESP32, and stores the latest valid speed reading plus a
#       "fresh" flag into the shared state dict under esp_lock.
#
#  All functions here operate on the config-module constants (SOF, EOF,
#  BAUDRATE, TIMEOUT, v_max) so the wire protocol stays centrally defined.
# =============================================================================

import struct
import time
import logging
import threading
import serial

from config import SOF, EOF, BAUDRATE, TIMEOUT, v_max, ESP_PORT

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("MPC_v3")


# =============================================================================
#  [CHANGE 4] SHARED STATE — ESP32 speed, protected by a Lock
#
#  All fields written by the listener thread, read by the MPC loop.
#  Never access these without holding esp_lock.
# =============================================================================
shared = {
    "v_actual": 0.0,    # latest valid speed received from ESP32  [m/s]
    "fresh":    False,  # True  → new packet arrived since last MPC read
                         # False → no new data; MPC should use model fallback
    "alive":    True,   # set to False by main() to stop the listener thread
}
esp_lock = threading.Lock()


def open_all_uarts():
    try:
        ser_esp = serial.Serial(ESP_PORT, BAUDRATE, timeout=TIMEOUT)
        time.sleep(0.1)
        ser_esp.reset_input_buffer()
        log.info("UART initialized and flushed.")
        return ser_esp
    except Exception as e:
        log.error(f"Failed to open UART port: {e}")
        return None


def send_uart_ESP(ser, speed, send_flag):
    """Send a 6-byte packet [SOF | float32_LE | EOF] to ESP32."""
    if ser is None or not ser.is_open:
        return False
    try:
        payload = struct.pack('<fB', speed, send_flag)
        packet  = bytes([SOF]) + payload + bytes([EOF])
        ser.write(packet)
        return True
    except Exception as e:
        log.error(f"UART send error: {e}")
        return False


# -------TOGGLE-----------
def brake_flag_toggler():
    """Generator-style toggler: call this each loop iteration."""
    last_toggle = time.time()
    flag = 0
    while True:
        now = time.time()
        if now - last_toggle >= 6:
            flag = 1 - flag  # toggles between 0 and 1
            last_toggle = now
        yield flag


# =============================================================================
#  [CHANGE 4] BACKGROUND UART LISTENER THREAD
#
#  Replaces the old receive_uart_ESP() polling call inside the MPC loop.
#  This thread runs independently — it blocks on ser.read() without ever
#  stalling the 50 ms MPC cycle.
#
#  Packet expected from ESP32:
#      Byte 0   : 0xAA  (SOF)
#      Bytes 1-4: float32 little-endian  (actual wheel speed in m/s)
#      Byte 5   : 0x55  (EOF)
# =============================================================================
def esp_listener(ser):
    """
    Background thread — continuously reads ESP32 speed feedback.

    Writes to shared["v_actual"] and shared["fresh"] under esp_lock.
    Exits cleanly when shared["alive"] becomes False.
    """
    log.info("[LISTENER] UART listener thread started.")

    while True:
        # ── Check shutdown signal ──────────────────────────────────────
        with esp_lock:
            if not shared["alive"]:
                break

        # ── Wait for SOF byte (blocks up to TIMEOUT=0.1 s, then retries) ──
        try:
            byte = ser.read(1)
        except Exception as e:
            log.warning(f"[LISTENER] Read error: {e}")
            time.sleep(0.01)
            continue

        if not byte:
            continue                    # timeout — loop and try again
        if byte[0] != SOF:
            continue                    # not a packet start — discard

        # ── Read remaining 5 bytes (4 payload + 1 EOF) ────────────────
        try:
            payload  = ser.read(4)
            eof_byte = ser.read(1)
        except Exception as e:
            log.warning(f"[LISTENER] Partial read error: {e}")
            continue

        if len(payload) != 4 or not eof_byte or eof_byte[0] != EOF:
            log.warning("[LISTENER] Malformed packet — discarding.")
            continue

        # ── Unpack and validate speed ──────────────────────────────────
        speed = struct.unpack('<f', payload)[0]
        if not (0.0 <= speed <= v_max):
            log.warning(f"[LISTENER] Out-of-range speed: {speed:.3f} m/s — discarding.")
            continue

        # ── Store in shared state ──────────────────────────────────────
        with esp_lock:
            shared["v_actual"] = speed
            shared["fresh"]    = True

        log.debug(f"[LISTENER] Received speed: {speed:.4f} m/s")
        log.info(f"[LISTENER] Received speed = {speed:.3f}")

    log.info("[LISTENER] UART listener thread exiting.")
