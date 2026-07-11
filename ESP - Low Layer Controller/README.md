# ESP32 Low-Layer Controller

The **ESP32 Low-Layer Controller** is the real-time embedded subsystem of the **AI-Enhanced Adaptive Cruise Control (ACC) with Integrated Telemetry and FOTA** platform.

It is responsible for executing deterministic vehicle control tasks, including motor actuation, sensor acquisition, UART communication with the Raspberry Pi, and safety-critical functions. While the Raspberry Pi performs perception and high-level decision making, the ESP32 guarantees reliable low-latency execution of all actuator and feedback loops.

---

# System Responsibilities

The ESP32 firmware provides the following core functionalities:

* Closed-loop DC motor speed control using a PID controller
* Bidirectional UART communication with the Raspberry Pi
* Real-time encoder speed measurement
* Monitoring of current, voltage, and temperature sensors
* Emergency braking and lane-departure safety handling
* LCD-based system status display
* Buzzer-based warning notifications

---

# Software Architecture

The firmware is implemented using **FreeRTOS**, where independent tasks execute concurrently while sharing data through queues, mutexes, and event groups.

| FreeRTOS Task         | Function                                                      |
| --------------------- | ------------------------------------------------------------- |
| **task_MotorControl** | Executes the PID control loop and drives the L298N H-bridge   |
| **task_UART_RX**      | Receives target speed and control flags from the Raspberry Pi |
| **task_UART_TX**      | Sends encoder RPM feedback to the Raspberry Pi                |
| **task_Read_Sensors** | Acquires current, temperature, and voltage measurements       |
| **task_LCD**          | Updates the 16×2 HD44780 LCD                                  |
| **task_Buzzer**       | Generates audible alerts for safety events                    |

### Synchronization Primitives

The firmware uses the following FreeRTOS synchronization mechanisms:

* `system_eventgroup`
* `speed_queue`
* `enc_queue`
* `sensor_mutex`

These primitives ensure deterministic communication and thread-safe access to shared resources.

---

# Control System

The ESP32 performs closed-loop speed regulation using a PID controller with several enhancements to improve stability and responsiveness.

### Hardware

* ESP32
* L298N H-Bridge Motor Driver
* 25GA370 DC Motor
* 12 V supply
* 374 PPR quadrature encoder

### Controller Parameters

| Parameter                |  Value |
| ------------------------ | -----: |
| Kp                       |   1.23 |
| Ki                       |   0.23 |
| Kd                       |   0.01 |
| Feedforward Gain (Kff)   |   0.80 |
| Filter Time Constant (τ) | 0.02 s |
| Control Period (T)       | 0.01 s |

### Controller Features

* Derivative-on-measurement implementation
* Low-pass filtered derivative term
* Trapezoidal integral calculation
* Dynamic anti-windup
* Smooth setpoint ramping
* Active braking by driving both H-bridge inputs HIGH

---

# UART Communication

The ESP32 exchanges data with the Raspberry Pi through a lightweight framed UART protocol.

### Command Frame

| Byte | Description                             |
| ---- | --------------------------------------- |
| 0    | Start Byte (`0xAA`)                     |
| 1–4  | Target Speed (`float32`, Little Endian) |
| 5    | Control Flags                           |
| 6    |  CRC XOR Checksum                       |
| 7    | End Byte (`0x55`)                       |

### Control Flags

| Bit | Function               |
| --: | ---------------------- |
|   0 | Emergency Stop         |
|   1 | Lane Departure Warning |

A software watchdog continuously monitors UART activity. If no valid frame is received within **2 seconds**, the controller automatically enters a safe state.

---

# Sensor Interface

The embedded controller monitors multiple vehicle parameters:

* ACS712 5 A Current Sensor
* LM35 Temperature Sensor
* 0–25 V Voltage Sensor

All sensors share a single ADC instance through `getADCHandle()`, preventing conflicts with the ESP-IDF single-owner ADC architecture.

---

# Safety Features

The firmware incorporates several mechanisms to ensure reliable and fail-safe operation.

* UART communication watchdog
* Automatic emergency braking
* Lane departure warning buzzer
* Controlled motor shutdown during communication failure
* Edge-detected `was_braking` logic to prevent repeated controller resets during continuous braking events

---

# Project Structure

```text
ESP_Controller/
├── include/                    # Header files
├── src/                        # Source files
├── lib/                        # Local libraries
├── test/                       # PlatformIO test framework
├── platformio.ini              # PlatformIO configuration
├── README.md                   # Documentation
└── STM32F103C6TX_FLASH.ld      # Linker script
```

Development utilities such as UART testing tools and real-time visualization scripts are stored separately in:

```text
Archive/
└── ESP_Controller_tests/
```

These utilities are intended exclusively for development and debugging and are not part of the embedded firmware build.

---

# Build Instructions

This project uses **PlatformIO**.

## Build

```bash
pio run
```

## Upload Firmware

```bash
pio run -t upload
```

## Serial Monitor

```bash
pio device monitor
```

---

# Integration within the ACC System

The ESP32 serves as the **real-time control layer** of the dual-layer Adaptive Cruise Control architecture.

* **Raspberry Pi 5**

  * AI perception
  * Lane detection
  * Vehicle detection
  * Distance estimation
  * Model Predictive Control (MPC)
  * Target speed generation

⬇ UART Communication

* **ESP32**

  * PID motor control
  * Encoder processing
  * Sensor acquisition
  * Safety management
  * Motor actuation

This separation enables computationally intensive perception algorithms to execute independently from deterministic real-time motor control, improving both responsiveness and system reliability.

---

# Contributing

When contributing to this subsystem:

* Keep ESP32-specific documentation within this README.
* Update the repository root `README.md` whenever changes affect communication protocols or multiple subsystems.
* Preserve existing documentation whenever possible and extend it with new sections rather than replacing existing content.
