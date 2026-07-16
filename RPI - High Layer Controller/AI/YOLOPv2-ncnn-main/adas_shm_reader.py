"""
adas_shm_reader.py
Run this in the SECOND terminal while the ADAS script is running.

Usage:
    python3 adas_shm_reader.py

Reads cut_in and lane_curve in real-time from shared memory.
Import read_adas_output() into any other script to get live values.
"""

import struct
import time
from multiprocessing import shared_memory

SHM_NAME = "adas_output"
SHM_SIZE = 8               # 2 × int32


def read_adas_output() -> dict:
    """
    Returns {'cut_in': int, 'lane_curve': int} from shared memory.
    Returns {'cut_in': 0, 'lane_curve': 0} if ADAS script is not yet running.
    """
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False, size=SHM_SIZE)
        cut_in, lane_curve = struct.unpack_from("ii", shm.buf, 0)
        shm.close()
        return {"cut_in": cut_in, "lane_curve": lane_curve}
    except FileNotFoundError:
        return {"cut_in": 0, "lane_curve": 0}


# ── If run directly: live monitor loop ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("ADAS Shared Memory Reader")
    print("Waiting for ADAS script to start...")
    print("=" * 50)

    shm = None
    while shm is None:
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=False, size=SHM_SIZE)
            print("[OK] Connected to ADAS shared memory.\n")
        except FileNotFoundError:
            time.sleep(0.2)

    try:
        while True:
            cut_in, lane_curve = struct.unpack_from("ii", shm.buf, 0)
            print(f"[{time.strftime('%H:%M:%S')}] cut_in={cut_in} | lane_curve={lane_curve}")
            time.sleep(0.05)          # ~20 Hz polling — adjust freely

    except KeyboardInterrupt:
        print("\n[STOP] Reader stopped.")
    finally:
        shm.close()
