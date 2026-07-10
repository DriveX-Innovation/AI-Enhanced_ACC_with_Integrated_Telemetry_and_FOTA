\# ESP32 Embedded Control Subsystem



This folder contains the firmware for the ESP32-based embedded control layer of the AI-Enhanced Adaptive Cruise Control system. It handles motor control, sensor reading, UART communication with the Raspberry Pi, and safety mechanisms.



\## Overview



The ESP32 receives target speed commands over UART from the Raspberry Pi (which runs the AI perception pipeline and MPC controller), executes closed-loop motor control via PID, and reports back encoder-derived RPM along with system status flags.



\## Architecture



Built on FreeRTOS with six tasks coordinated through shared synchronization primitives:



| Task | Responsibility |

|---|---|

| `task\_MotorControl` | PID control loop, drives the L298N H-bridge |

| `task\_UART\_RX` | Receives target speed + flags from the Raspberry Pi |

| `task\_UART\_TX` | Sends encoder RPM feedback to the Raspberry Pi |

| `task\_Read\_Sensors` | Polls current, temperature, and voltage sensors |

| `task\_LCD` | Updates the 2x16 HD44780 display |

| `task\_Buzzer` | Drives audible alerts (lane departure, emergency stop) |



Synchronization: `system\_eventgroup`, `speed\_queue`, `enc\_queue`, `sensor\_mutex`.



\## Motor Control



\- L298N H-bridge driving a 25GA370 motor (12V, 250 RPM, 374 PPR encoder)

\- PID gains: Kp=1.23, Ki=0.23, Kd=0.01, feedforward Kff=0.8, τ=0.02s, T=0.01s

\- Derivative-on-measurement with low-pass filtering

\- Trapezoidal integration with dynamic anti-windup

\- Setpoint ramping and active braking (both H-bridge pins high)



\## UART Protocol



7-byte framed protocol between ESP32 and Raspberry Pi:

