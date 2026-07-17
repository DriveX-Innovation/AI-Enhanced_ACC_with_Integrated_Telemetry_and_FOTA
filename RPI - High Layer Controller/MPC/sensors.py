# =============================================================================
#  MODULE: sensors.py
#
#  DESCRIPTION:
#  Wraps the physical HC-SR04 ultrasonic distance sensor connected to the
#  Raspberry Pi 5 GPIO pins, which supplies the real inter-vehicle gap
#  distance used by the MPC controller ([CHANGE 3] in the original
#  script). Exposes a single function, get_real_distance(), that returns
#  a validated distance reading in metres, or None if the reading is out
#  of the sensor's reliable range or an error occurs — in which case the
#  MPC loop falls back to its internal kinematic model estimate.
#
#  Trigger pin : GPIO 23
#  Echo   pin  : GPIO 24
#  max_distance: 4.0 m
# =============================================================================

import logging
from gpiozero import DistanceSensor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("MPC_v3")


# =============================================================================
#  [CHANGE 3] ULTRASONIC SENSOR — HC-SR04 on RPi GPIO
#
#  Trigger pin : GPIO 23
#  Echo   pin  : GPIO 24
#  max_distance: 4.0 m
# =============================================================================
sensor = DistanceSensor(echo=24, trigger=23, max_distance=4.0)


def get_real_distance():
    """
    Read gap distance from the HC-SR04 ultrasonic sensor.

    Returns
    -------
    float : gap in metres (0.05 … 4.0), or
    None  : if out of reliable range or sensor error.
    """
    try:
        dist = sensor.distance
        if 0.05 <= dist <= 4.0:
            return dist
        else:
            log.warning(f"Ultrasonic out of range: {dist:.3f} m — using model estimate.")
            return None
    except Exception as e:
        log.warning(f"Ultrasonic sensor error: {e} — using model estimate.")
        return None
