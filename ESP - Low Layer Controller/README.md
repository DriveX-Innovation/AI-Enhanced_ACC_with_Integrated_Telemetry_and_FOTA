# ESP32 Embedded Control Subsystem

This folder contains the firmware for the ESP32-based embedded control layer of the AI-Enhanced Adaptive Cruise Control system. It handles motor control, sensor reading, UART communication with the Raspberry Pi, and safety mechanisms.

## Overview

The ESP32 receives target speed commands over UART from the Raspberry Pi (which runs the AI perception pipeline and MPC controller), executes closed-loop motor control via PID, and reports back encoder-derived RPM along with system status flags.

## Architecture

Built on FreeRTOS with six tasks coordinated through shared synchronization primitives:

| Task | Responsibility |
|---|---|
| task_MotorControl | PID control loop, drives the L298N H-bridge |
| task_UART_RX | Receives target speed + flags from the Raspberry Pi |
| task_UART_TX | Sends encoder RPM feedback to the Raspberry Pi |
| task_Read_Sensors | Polls current, temperature, and voltage sensors |
| task_LCD | Updates the 2x16 HD44780 display |
| task_Buzzer | Drives audible alerts (lane departure, emergency stop) |

Synchronization: system_eventgroup, speed_queue, enc_queue, sensor_mutex.

## Motor Control

- L298N H-bridge driving a 25GA370 motor (12V, 250 RPM, 374 PPR encoder)
- PID gains: Kp=1.23, Ki=0.23, Kd=0.01, feedforward Kff=0.8, tau=0.02s, T=0.01s
- Derivative-on-measurement with low-pass filtering
- Trapezoidal integration with dynamic anti-windup
- Setpoint ramping and active braking (both H-bridge pins high)

## UART Protocol

7-byte framed protocol between ESP32 and Raspberry Pi: [0xAA][float32 LE speed][uint8 flags][0x55]

Flags byte: bit 0 = emergency stop, bit 1 = lane departure warning.
A 2-second software watchdog resets control if no valid frame is received in time.

## Sensors

- ACS712 5A current sensor
- LM35 temperature sensor
- 0-25V voltage divider
- All share a single ADC handle via getADCHandle() to avoid ESP-IDF single-owner ADC unit restriction

## Safety Mechanisms

- Software watchdog (2s timeout)
- Active braking on emergency stop
- Lane departure warning via buzzer
- Edge-detected was_braking flag ensures PID/ramp/encoder state resets only once per brake event, not every tick

## Build and Flash

This is a PlatformIO project.

pio run              (build)
pio run -t upload    (flash)
pio device monitor   (serial monitor)

## Folder Structure

ESP_Controller/
  include/     headers
  src/         implementation (.cpp)
  lib/         local libraries (if any)
  test/        PlatformIO test framework (currently unused/placeholder)
  platformio.ini
  README.md

Python test/debug scripts used during development are in ../Archive/ESP_Controller_tests/ (real-time plotting, manual UART frame senders, etc.) - kept separate from the firmware since they are host-side tools, not part of the embedded build.

## Contributing

This README documents the ESP32 subsystem specifically. If you are adding work here:
- Keep subsystem-specific docs in this file
- If your addition touches multiple subsystems (e.g. shared protocol changes), also update the root README.md
- Please do not remove or overwrite existing sections without discussing with the team, add a new section instead
