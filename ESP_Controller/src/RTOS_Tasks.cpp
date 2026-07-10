#include "RTOS_Tasks.h"
#include <cstring>   // memcpy
#include <cmath>     // isfinite
#include "shared_data.h"

#include "Motor.h"
#include "Encoder.h"
#include "PID.h"
#include "UART.h"
//#include "Throttle.h"
#include "LCD.h"

#include "VoltSensor.h"
#include "CurrentSensor.h"
#include "TempSensor.h"

#include "esp_log.h"
#include "driver/gpio.h"

/* ================================================================
 *  PIN / PERIPHERAL DEFINITIONS
 * ================================================================ */

// --- Left motor driver pins (H-bridge IN3/IN4 direction, ENB = PWM enable) ---
#define IN1_PIN         GPIO_NUM_25
#define IN2_PIN         GPIO_NUM_32
#define ENA_PIN         GPIO_NUM_33

// --- Right motor driver pins (H-bridge IN1/IN2 direction, ENA = PWM enable) ---
#define IN3_PIN         GPIO_NUM_13
#define IN4_PIN         GPIO_NUM_12
#define ENB_PIN         GPIO_NUM_14

// --- Quadrature encoder input pins, used only by left_motor's shaft ---
#define ENCODER_PIN_A   GPIO_NUM_26
#define ENCODER_PIN_B   GPIO_NUM_27

// --- ADC channel reserved for throttle input (currently unused/disabled) ---
#define ADC_CHANNEL     ADC_CHANNEL_6

// --- UART2 configuration used for RPi <-> ESP32 speed command link ---
#define UART_TX_PIN     GPIO_NUM_17
#define UART_RX_PIN     GPIO_NUM_16
#define UART_BAUD_RATE  115200
#define UART_PORT       UART_NUM_2

// --- PID gains and timing constants (tuned for this motor/encoder setup) ---
#define PID_KP          1.23f
#define PID_KI          0.23f
#define PID_KD          0.01f
#define SAMPLINGTIME    0.01f   // 10 ms control loop period, must match task_MotorControl's period
#define TAU             0.02f  // derivative low-pass filter time constant

// Converts encoder RPM to linear surface speed (m/s) for a wheel of radius 0.034 m:
// speed = RPM * 2*PI*r / 60
#define RPM_TO_MS  0.0035604717f   // 2*PI*0.034/60

#define BUZZER_GPIO         GPIO_NUM_4
#define MOTOR_WATCHDOG_MS   2000   // if no valid UART frame arrives within this window, force stop

/* Ramp: max RPM change per 10ms control tick.
   20 RPM/tick = 2000 RPM/s — fast enough to feel responsive,
   slow enough to stop the feedforward from spiking on step changes. */
#define RAMP_RATE_RPM_PER_TICK  20

/* Active braking duration in control ticks (10 ms each).
   150 ticks = 1.5 s of reverse-polarity braking then coast.         */
#define BRAKE_TICKS  150

/* ================================================================
 *  DRIVER OBJECT INSTANCES
 * ================================================================ */
// Global driver objects — constructed once at startup, shared across tasks.
static Motor    left_motor(IN3_PIN, IN4_PIN, ENB_PIN);   // closed-loop motor (has encoder feedback)
static Motor    right_motor(IN1_PIN, IN2_PIN, ENA_PIN);  // open-loop motor (mirrors left_motor's output)
static Encoder  encoder(ENCODER_PIN_A, ENCODER_PIN_B);   // encoder attached to left_motor's shaft only
//static Throttle throttle(ADC_CHANNEL);
static UART     uart(UART_PORT, UART_TX_PIN, UART_RX_PIN, UART_BAUD_RATE);
static LCD      lcd;
static PID_Data pid;   // single PID instance, drives left_motor (right_motor just copies the output)

/* ================================================================
 *  IPC HANDLE DEFINITIONS
 * ================================================================ */
// speed_queue: holds the latest target RPM received over UART (producer: task_UART_RX, consumer: task_MotorControl)
QueueHandle_t        speed_queue       = nullptr;
// enc_queue: holds the latest measured actual RPM from the encoder (producer: task_MotorControl, consumers: task_UART_TX, task_LCD)
QueueHandle_t        enc_queue         = nullptr;
// system_eventgroup: holds cross-task status flags (EMERGENCY_STOP_BIT, LANE_ALERT_BIT)
EventGroupHandle_t   system_eventgroup = nullptr;
// sensor_mutex: guards shared_sensors from concurrent read/write between task_Read_Sensors and task_LCD
SemaphoreHandle_t    sensor_mutex      = nullptr;
sensor_data_t        shared_sensors    = {0.0f, 0.0f, 0.0f};
// last_uart_rx_tick: timestamp of the last valid UART frame, used by the motor watchdog
volatile TickType_t  last_uart_rx_tick = 0;

static const char *TAG_MOTOR = "MOTOR_TASK";
static const char *TAG_RX    = "UART_RX";

/* ================================================================
 *  RTOS_init
 *  Creates all IPC primitives (queues/event group/mutex) used by the
 *  tasks below. Must be called once, before any task starts running.
 * ================================================================ */
void RTOS_init(void)
{
    // Depth-1 "overwrite" queues: only the newest value matters, so there's
    // no need to buffer a backlog of stale speed/encoder samples.
    speed_queue = xQueueCreate(1, sizeof(int16_t));
    enc_queue   = xQueueCreate(1, sizeof(int16_t));
    configASSERT(speed_queue);
    configASSERT(enc_queue);

    system_eventgroup = xEventGroupCreate();
    configASSERT(system_eventgroup);

    sensor_mutex = xSemaphoreCreateMutex();
    configASSERT(sensor_mutex);

    // Seed both queues with 0 so the first xQueuePeek() in any task
    // never reads garbage/uninitialized data before the first real update.
    int16_t zero = 0;
    xQueueOverwrite(speed_queue, &zero);
    xQueueOverwrite(enc_queue,   &zero);

    ESP_LOGI("RTOS_INIT", "[OK] Queues, event group, mutex created");
}

/* ================================================================
 *  PERIPHERAL INIT
 *  One-time hardware bring-up: LCD, motors, encoder, UART, and PID
 *  state. Must run after RTOS_init() and before any task starts.
 * ================================================================ */
void peripherals_init(void)
{
    ESP_LOGI("INIT", ">>> peripherals_init START");

    lcd.LCD_Setup();
    ESP_LOGI("INIT", "[OK] LCD");
    
    left_motor.MotorSetup();
    ESP_LOGI("INIT", "[OK] Motor  INA=%d INB=%d PWM=%d",
             (int)IN3_PIN, (int)IN4_PIN, (int)ENB_PIN);

    right_motor.MotorSetup();
    ESP_LOGI("INIT", "[OK] Motor  INA=%d INB=%d PWM=%d",
             (int)IN1_PIN, (int)IN2_PIN, (int)ENA_PIN);

    encoder.EncoderSetup();
    ESP_LOGI("INIT", "[OK] Encoder  A=%d B=%d",
             (int)ENCODER_PIN_A, (int)ENCODER_PIN_B);

    //throttle.ThrottleSetup();
    //ESP_LOGI("INIT", "[OK] Throttle  ADC_CH=%d", (int)ADC_CHANNEL);

    uart.UART_init();
    ESP_LOGI("INIT", "[OK] UART  port=%d TX=%d RX=%d baud=%d",
             (int)UART_PORT, (int)UART_TX_PIN, (int)UART_RX_PIN, UART_BAUD_RATE);

    // Default direction on power-up: both motors set to BACKWARD.
    // task_MotorControl treats BACKWARD as the "normal running" direction
    // and only switches to FORWARD momentarily during active braking.
    left_motor.MoveBackward();
    right_motor.MoveBackward();
    ESP_LOGI("INIT", "[OK] Motor direction: BACKWARD (both motors)");

    // Initialize PID gains/state; pid_init() resets integrator/derivative history.
    pid = {PID_KP, PID_KI, PID_KD, TAU, SAMPLINGTIME,
           0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    pid_init(&pid);
    ESP_LOGI("INIT", "[OK] PID  Kp=%.3f Ki=%.3f Kd=%.3f",
             PID_KP, PID_KI, PID_KD);

    ESP_LOGI("INIT", ">>> peripherals_init DONE  watchdog=%dms",
             MOTOR_WATCHDOG_MS);
}

/* ================================================================
 *  STACK OVERFLOW HOOK
 *  FreeRTOS calls this automatically if a task's stack overflows.
 *  We log the offending task name then halt via configASSERT so the
 *  fault is obvious instead of silently corrupting memory.
 * ================================================================ */
extern "C" void vApplicationStackOverflowHook(TaskHandle_t, char *pcTaskName)
{
    ESP_LOGE("STACK", "OVERFLOW in task: '%s'", pcTaskName);
    configASSERT(0);
}

/* ================================================================
 *  TASK 1 — task_MotorControl   Core 1 | Priority 3 | 10 ms
 *
 *  FIX 1 — Setpoint ramp:
 *    Instead of applying the received RPM directly to the PID,
 *    we ramp an internal setpoint toward the target at RAMP_RATE_RPM_PER_TICK
 *    per 10ms tick. This prevents the feedforward term (KFF*setpoint) from
 *    injecting a full-speed spike the moment a new target arrives.
 *
 *  FIX 2 — Active braking:
 *    When target drops to 0, instead of just setting PWM=0 (freewheeling),
 *    we flip the motor direction for BRAKE_TICKS × 10ms to apply
 *    counter-torque and stop the shaft quickly. After braking we return
 *    to the normal BACKWARD direction and wait for the next command.
 *
 *  NOTE — right_motor:
 *    right_motor has no encoder, so it is run fully open-loop: it simply
 *    mirrors whatever direction/PWM command left_motor receives, on the
 *    same tick. No PID, ramp, or braking logic was changed — right_motor
 *    just echoes left_motor's output.
 * ================================================================ */
void task_MotorControl(void *pvParameters)
{
    const TickType_t period      = pdMS_TO_TICKS(10);       // 10 ms control period, matches SAMPLINGTIME
    TickType_t       wakeTime    = xTaskGetTickCount();      // reference tick for precise periodic wakeups
    uint32_t         iter        = 0;                        // tick counter, used for periodic debug logging
    float            last_pwm    = 0.0f;                     // last PWM value applied to both motors
    int              last_actual = 0;                        // last measured actual RPM from the encoder
    float            ramped_rpm  = 0.0f;                     // internal setpoint, ramps toward target_rpm
    bool             was_zero    = true;                     // tracks target_rpm==0 on the previous tick (edge detection)
    int              brake_timer = 0;                        // countdown of remaining active-braking ticks
    bool             was_braking = false;                    // tracks EMERGENCY_STOP_BIT state on the previous tick

    ESP_LOGI(TAG_MOTOR, "Started on core %d  watchdog=%dms  ramp=%d RPM/tick",
             xPortGetCoreID(), MOTOR_WATCHDOG_MS, RAMP_RATE_RPM_PER_TICK);

    while (1)
    {
        /* 1. Read target RPM from queue */
        // xQueuePeek (not receive) so the value stays in the queue for
        // the next tick too — task_MotorControl never "consumes" it.
        int16_t target_rpm = 0;
        xQueuePeek(speed_queue, &target_rpm, 0);

        /* 2. Watchdog */
        // If no valid UART frame has arrived recently, force target to 0
        // regardless of what's sitting in speed_queue (fail-safe stop).
        bool watchdog_fired = false;
        if (last_uart_rx_tick != 0 &&
            (xTaskGetTickCount() - last_uart_rx_tick) >
             pdMS_TO_TICKS(MOTOR_WATCHDOG_MS))
        {
            target_rpm     = 0;
            watchdog_fired = true;
        }

        /* 3. Brake flag ─────────────────────────────────────────────────── */
        // EMERGENCY_STOP_BIT is set by task_UART_RX when the RPi sends the
        // e-stop flag; this takes priority over everything else below.
        bool brake_flag = (xEventGroupGetBits(system_eventgroup) & EMERGENCY_STOP_BIT) != 0;

        if (brake_flag)
        {
            if (!was_braking)
            {
                // Entering emergency brake state: reset PID/ramp/encoder so
                // there's no stale integral term or setpoint when we resume.
                pid_init(&pid);
                last_pwm    = 0.0f;
                ramped_rpm  = 0.0f;
                brake_timer = 0;
                was_zero    = true;
                encoder.ResetEncoderCount();
                ESP_LOGW(TAG_MOTOR, "BRAKE_BIT set — calling motor.Brake()");
            }
            was_braking = true;
            left_motor.Brake();
            right_motor.Brake();

            vTaskDelayUntil(&wakeTime, period);
            iter++;
            continue;   // skip all normal control logic while e-stop is active
        }

        if (was_braking)
        {
            // Just came out of emergency brake: explicitly re-arm speed=0
            // and restore the default BACKWARD direction before resuming.
            left_motor.SetSpeed(0);
            left_motor.MoveBackward();
            right_motor.SetSpeed(0);
            right_motor.MoveBackward();
            ESP_LOGW(TAG_MOTOR, "BRAKE_BIT cleared — direction restored BACKWARD");
        }
        was_braking = false;
        /* ─────────────────────────────────────────────────────────────────── */

        /* 4. Detect target → 0 transition: reset PID and start braking */
        // Rising edge of "target==0" triggers a fresh active-braking sequence
        // (separate from the emergency e-stop path above).
        bool is_zero = (target_rpm == 0);
        if (is_zero && !was_zero)
        {
            pid_init(&pid);
            last_pwm    = 0.0f;
            last_actual = 0;
            ramped_rpm  = 0.0f;
            encoder.ResetEncoderCount();
            brake_timer = BRAKE_TICKS;
            left_motor.MoveForward();   // reverse polarity vs. normal BACKWARD = counter-torque braking
            right_motor.MoveForward();
            ESP_LOGI(TAG_MOTOR, "Target→0: PID reset, braking for %d ticks",
                     BRAKE_TICKS);
        }
        was_zero = is_zero;

        /* 5. Manage active braking state */
        if (brake_timer > 0)
        {
            brake_timer--;
            // Fixed braking PWM (not PID-controlled) applied in the reversed
            // direction set above, for BRAKE_TICKS consecutive ticks.
            left_motor.SetSpeed(1500);
            right_motor.SetSpeed(1500);
            if (brake_timer == 0)
            {
                // Braking window elapsed: stop and restore normal direction.
                left_motor.SetSpeed(0);
                left_motor.MoveBackward();
                right_motor.SetSpeed(0);
                right_motor.MoveBackward();
                ESP_LOGI(TAG_MOTOR, "Braking done — direction restored BACKWARD");
            }
            printf("Actual: %4d rpm --> %.4f m/s     Received: %.4f m/s --> %4d rpm\n",
                   last_actual,  (float)last_actual * RPM_TO_MS,
                   (float)target_rpm * RPM_TO_MS,  (int)target_rpm);
            vTaskDelayUntil(&wakeTime, period);
            iter++;
            continue;   // skip ramp/PID/apply steps below while actively braking
        }

        /* 6. Ramp the setpoint toward target_rpm */
        // Slew-rate limit: ramped_rpm can move at most RAMP_RATE_RPM_PER_TICK
        // per 10ms tick, smoothing step changes in target_rpm.
        if (!is_zero)
        {
            float diff = (float)target_rpm - ramped_rpm;
            if (diff >  RAMP_RATE_RPM_PER_TICK) diff =  RAMP_RATE_RPM_PER_TICK;
            if (diff < -RAMP_RATE_RPM_PER_TICK) diff = -RAMP_RATE_RPM_PER_TICK;
            ramped_rpm += diff;
        }
        else
        {
            ramped_rpm = 0.0f;
        }

        /* 7. Encoder — skip PID if sample not ready */
        // CalculateSpeed() returns a negative sentinel when no new edge
        // count is available yet, so we don't feed stale data to the PID.
        int raw_speed   = encoder.CalculateSpeed();
        bool new_sample = (raw_speed >= 0);
        if (new_sample)
        {
            last_actual = raw_speed;
            int16_t actual_for_tx = (int16_t)last_actual;
            xQueueOverwrite(enc_queue, &actual_for_tx);   // publish latest actual RPM for TX/LCD tasks
        }

        /* 8. PID — only when running and fresh encoder sample available */
        if (!is_zero && !watchdog_fired && new_sample)
            last_pwm = pid_CalcPWM(&pid, ramped_rpm, (float)last_actual);

        /* 9. Apply to motor */
        // right_motor has no encoder/PID of its own — it always mirrors
        // whatever PWM last_pwm the closed-loop left_motor computed.
        if (!is_zero && !watchdog_fired)
        {
            left_motor.SetSpeed((int)last_pwm);
            right_motor.SetSpeed((int)last_pwm);
        }
        else if (watchdog_fired)
        {
            left_motor.SetSpeed(0);
            right_motor.SetSpeed(0);
            last_pwm   = 0.0f;
            ramped_rpm = 0.0f;
        }

        /* 10. Status line */
        printf("Actual: %4d rpm --> %.4f m/s     Received: %.4f m/s --> %4d rpm\n",
               last_actual,  (float)last_actual * RPM_TO_MS,
               (float)target_rpm * RPM_TO_MS,  (int)target_rpm);

        /* 11. Debug every ~1 s */
        iter++;
        if (iter % 100 == 0)   // 100 ticks * 10ms = ~1 second
        {
            ESP_LOGW(TAG_MOTOR,
                     "[%lu] target=%d ramped=%.0f actual=%d pwm=%.0f wd=%s",
                     (unsigned long)iter,
                     (int)target_rpm, ramped_rpm, last_actual, last_pwm,
                     watchdog_fired ? "FIRED!" : "ok");
        }

        vTaskDelayUntil(&wakeTime, period);
    }
}

/* ================================================================
 *  TASK 2 — task_UART_RX   Core 0 | Priority 3
 *
 *  Packet format sent by the RPi (6 bytes):
 *    byte[0]    = 0xAA  (header / start-of-frame)
 *    byte[1..4] = IEEE 754 float32, Little-Endian  (speed in m/s)
 *    byte[5]    = 0x55  (footer / end-of-frame)
 *
 *  The RPi packs with:
 *    struct.pack('<BBfB', 0xAA, float(speed), 0x55)   — 6 bytes
 *
 *  On a valid frame we convert m/s → RPM and push to speed_queue.
 *  Any frame whose header or footer doesn't match is discarded and
 *  counted as a bad read so we never act on corrupt data.
 * ================================================================ */
// NOTE: the docstring above describes a 6-byte frame, but RX_FRAME_LEN
// below is 7 bytes — a flags byte (buf[5]) was added between the float
// and the footer for emergency-stop/lane-departure signaling.
#define RX_HEADER     0xAAu
#define RX_FOOTER     0x55u
#define RX_FRAME_LEN  7          // 1 (header) + 4 (float32) + 1 (Emg flag+) + 1 (footer)

void task_UART_RX(void *pvParameters)
{
    uint8_t  buf[RX_FRAME_LEN];
    uint32_t total_frames = 0;   // count of accepted, valid frames
    uint32_t bad_reads    = 0;   // count of malformed/partial/non-finite frames

    ESP_LOGI(TAG_RX, "Started on core %d  frame=%d bytes [0xAA][f32 LE][flags][0x55]",
             xPortGetCoreID(), RX_FRAME_LEN);

    while (1)
    {
        // Blocking (or timeout-based, depending on UART_read's implementation)
        // read of exactly one frame's worth of bytes.
        int received = uart.UART_read(buf, sizeof(buf));

        if (received == RX_FRAME_LEN)
        {
            /* Validate header and footer */
            if (buf[0] != RX_HEADER || buf[6] != RX_FOOTER)
            {
                bad_reads++;
                if (bad_reads % 50 == 1)   // throttle logging to avoid flooding on a bad link
                {
                    ESP_LOGW(TAG_RX,
                             "bad frame markers: hdr=0x%02X ftr=0x%02X | "
                             "bad=%lu ok=%lu",
                             buf[0], buf[6],
                             (unsigned long)bad_reads,
                             (unsigned long)total_frames);
                }
                continue;
            }

            /* Unpack float32 from bytes [1..4] via memcpy (safe type-pun) */
            float speed_ms = 0.0f;
            memcpy(&speed_ms, &buf[1], sizeof(float));

            /* Sanity-check: reject NaN / Inf / negative values */
            if (!isfinite(speed_ms) || speed_ms < 0.0f)
            {
                bad_reads++;
                ESP_LOGW(TAG_RX, "non-finite or negative speed=%.4f — discarded",
                         speed_ms);
                continue;
            }

            total_frames++;

            /* ── Flag byte [5]: bit0 = emergency stop, bit1 = lane departure ── */
            uint8_t emergency_stop = (buf[5] >> 0) & 0x01u;
            uint8_t lane_departure = (buf[5] >> 1) & 0x01u;

            /* Lane departure — always update regardless of e-stop */
            // Level-triggered, not edge-triggered: bit is re-evaluated every
            // frame, so the buzzer task always reflects the current state.
            if (lane_departure)
                xEventGroupSetBits(system_eventgroup, LANE_ALERT_BIT);
            else
                xEventGroupClearBits(system_eventgroup, LANE_ALERT_BIT);

            if (emergency_stop)
            {
                // Set the e-stop bit and refresh the watchdog timestamp, but
                // skip updating speed_queue — task_MotorControl's brake path
                // takes over entirely while this bit is set.
                xEventGroupSetBits(system_eventgroup, EMERGENCY_STOP_BIT);
                last_uart_rx_tick = xTaskGetTickCount();
                ESP_LOGW(TAG_RX, "[frame #%lu] EMERGENCY_STOP flag set  lane_dep=%d",
                         (unsigned long)total_frames, lane_departure);
                continue;
            }

            xEventGroupClearBits(system_eventgroup, EMERGENCY_STOP_BIT);
            /* ─────────────────────────────────────────────────────────── */

            /* Convert m/s → RPM */
            int16_t rpm = (int16_t)(speed_ms / RPM_TO_MS);

            if (total_frames % 10 == 1)   // log roughly every 10th accepted frame
            {
                ESP_LOGI(TAG_RX,
                         "[frame #%lu] speed=%.4f m/s → rpm=%d  flags=0x%02X "
                         "raw=[0x%02X 0x%02X 0x%02X 0x%02X 0x%02X 0x%02X 0x%02X]",
                         (unsigned long)total_frames,
                         speed_ms, (int)rpm, buf[5],
                         buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6]);
            }

            /* Clamp to allowed RPM range */
            if (rpm < (int16_t)MIN_SPEED_RPM) rpm = (int16_t)MIN_SPEED_RPM;
            if (rpm > (int16_t)MAX_SPEED_RPM) rpm = (int16_t)MAX_SPEED_RPM;

            xQueueOverwrite(speed_queue, &rpm);   // publish new target; task_MotorControl picks it up next tick
            last_uart_rx_tick = xTaskGetTickCount();   // refresh watchdog timestamp on every valid, non-e-stop frame
        }
        else
        {
            // Partial read or timeout with no data — likely a dropped byte,
            // link glitch, or nothing sent yet.
            bad_reads++;
            if (bad_reads % 50 == 1)
            {
                ESP_LOGW(TAG_RX,
                         "partial/no data: got %d of %d bytes | bad=%lu ok=%lu",
                         received, RX_FRAME_LEN,
                         (unsigned long)bad_reads,
                         (unsigned long)total_frames);
            }
        }
    }
}
/* ================================================================
 *  TASK 3 — task_UART_TX   Core 0 | Priority 2 | 100 ms
 *
 *  Reads actual RPM from enc_queue, converts to surface speed (m/s),
 *  and sends a 4-byte Little-Endian IEEE 754 float over UART2 every 100ms.
 *
 *  Conversion:  speed (m/s) = RPM * 2*PI*r / 60
 *               r = 0.034 m (3.4 cm shaft extension radius)
 *               constant = 0.0035604717
 *
 *  Frame layout (4 bytes, LE float):
 *    byte[0..3] = IEEE 754 float, Little-Endian
 *
 *  The Python receiver unpacks with: struct.unpack('<f', data)[0]
 * ================================================================ */

static const char *TAG_TX = "UART_TX";

void task_UART_TX(void *pvParameters)
{
    const TickType_t period    = pdMS_TO_TICKS(100);   // 100 ms TX cadence, independent of the 10 ms control loop
    TickType_t       wakeTime  = xTaskGetTickCount();
    uint32_t         tx_count  = 0;
    uint32_t         no_data_count = 0;   // counts ticks where enc_queue had nothing yet

    ESP_LOGI(TAG_TX, "Started on core %d — sending speed (m/s) every 100ms, r=0.034m",
             xPortGetCoreID());

    while (1)
    {
        /* 1. Read latest actual RPM from enc_queue (non-blocking) */
        int16_t actual_rpm = 0;
        BaseType_t q_ok = xQueuePeek(enc_queue, &actual_rpm, 0);

        if (q_ok != pdTRUE)
        {
            no_data_count++;
            /* Queue empty means encoder hasn't produced a sample yet —
               send 0.0 rad/s so the receiver always gets a frame        */
            actual_rpm = 0;
        }

        /* 2. Convert RPM → rad/s */
        // (comment says rad/s but this is actually linear m/s via RPM_TO_MS — see conversion note above)
        float speed_ms = (float)actual_rpm * RPM_TO_MS;

        /* 3. Pack as 4-byte Little-Endian IEEE 754 float */
        uint8_t frame[4];
        memcpy(frame, &speed_ms, sizeof(float));   // safe type-pun via memcpy

        /* 4. Send over UART2 */
        uart.UART_write((const char*)frame, sizeof(frame));

        tx_count++;

        /* 5. Debug log every 10 transmissions (~1 s) */
        if (tx_count % 10 == 1)
        {
            ESP_LOGI(TAG_TX,
                     "[tx #%lu] rpm=%d | speed=%.4f m/s | "
                     "frame=[0x%02X 0x%02X 0x%02X 0x%02X] | "
                     "queue=%s | no_data=%lu",
                     (unsigned long)tx_count,
                     (int)actual_rpm,
                     speed_ms,
                     frame[0], frame[1], frame[2], frame[3],
                     (q_ok == pdTRUE) ? "OK" : "EMPTY",
                     (unsigned long)no_data_count);
        }

        vTaskDelayUntil(&wakeTime, period);
    }
}

/* ================================================================
 *  TASK 4 — task_Read_Sensors
 *
 *  Runs every 100 ms on Core 0 at Priority 2.
 *  Instantiates the three sensor objects once (static locals so they
 *  live for the lifetime of the task), calls their setup on first
 *  entry, then reads voltage / current / temperature every tick and
 *  writes the results into shared_sensors under sensor_mutex.
 *
 *  The mutex is held only for the memcpy-sized critical section —
 *  never during ADC reads, which can block for several microseconds.
 * ================================================================ */

static const char *TAG_SNS = "SENSOR_TASK";

void task_Read_Sensors(void *pvParameters)
{
    const TickType_t period   = pdMS_TO_TICKS(100);
    TickType_t       wakeTime = xTaskGetTickCount();

    /* ── One-time sensor construction & setup ── */
    // static local + null-check pattern: lazily initialize the ADC unit
    // and sensor objects exactly once, the first time this task runs,
    // without needing a separate global init function.
    static adc_oneshot_unit_handle_t adc1_handle = nullptr;
    if (adc1_handle == nullptr)
    {
        adc_oneshot_unit_init_cfg_t init_cfg = {
            .unit_id  = ADC_UNIT_1,
            .clk_src  = ADC_RTC_CLK_SRC_DEFAULT,
            .ulp_mode = ADC_ULP_MODE_DISABLE,
        };
        ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_cfg, &adc1_handle));
        ESP_LOGI(TAG_SNS, "[OK] ADC1 unit created");
    }

    static VoltageSensor  volt_sensor(ADC_CHANNEL_7, adc1_handle);
    static CurrentSensor  curr_sensor(ADC_CHANNEL_3, adc1_handle);
    static TempSensor     temp_sensor(ADC_CHANNEL_0, adc1_handle);

    static bool sensors_ready = false;
    if (!sensors_ready)
    {
        volt_sensor.voltageSensorSetup();
        curr_sensor.currentSensorSetup();
        temp_sensor.tempSensorSetup();
        curr_sensor.calibrateZeroOffset();   // zero the ACS712 at rest
        sensors_ready = true;
        ESP_LOGI(TAG_SNS, "[OK] All sensors initialised  V=CH7 I=CH3 T=CH0");
    }

    ESP_LOGI(TAG_SNS, "Started on core %d  period=100ms", xPortGetCoreID());

    while (1)
    {
        /* ── 1. Read sensors (outside mutex — ADC reads can take ~10 µs) ── */
        float voltage     = volt_sensor.readInputMilliVolts() / 1000.0f;   // mV -> V
        float current     = curr_sensor.readCurrent();
        float temperature = temp_sensor.readTemperatureCelsius();

        /* ── 2. Sanity-clamp: replace any non-finite value with 0 ── */
        // Prevents a transient bad ADC read (NaN/Inf) from propagating
        // into the shared struct and corrupting downstream displays/logic.
        if (!isfinite(voltage))     voltage     = 0.0f;
        if (!isfinite(current))     current     = 0.0f;
        if (!isfinite(temperature)) temperature = 0.0f;

        /* ── 3. Write into shared struct under mutex ── */
        if (xSemaphoreTake(sensor_mutex, pdMS_TO_TICKS(10)) == pdTRUE)
        {
            shared_sensors.voltage     = voltage;
            shared_sensors.current     = current;
            shared_sensors.temperature = temperature;
            xSemaphoreGive(sensor_mutex);
        }
        else
        {
            // If the mutex can't be acquired within 10ms, skip this update
            // rather than block indefinitely and risk missing the next period.
            ESP_LOGW(TAG_SNS, "sensor_mutex timeout — skipping write");
        }

        vTaskDelayUntil(&wakeTime, period);
    }
}

/* ================================================================
 *  TASK 5 — task_LCD
 *
 *  Runs every 200 ms on Core 0 at Priority 1.
 *  Copies shared_sensors under mutex (no LCD work while holding it),
 *  peeks enc_queue for actual RPM, then refreshes both LCD rows.
 *
 *  Display layout (16 columns):
 *    Row 1:  V:xx.xx  A:x.xx      (voltage V, current A)
 *    Row 2:  T:xx.xC S:x.xx       (temperature °C, speed m/s)
 *
 *  LCD_Clear() is intentionally avoided every cycle — we overwrite
 *  in place to prevent flicker. A full clear is only done on startup.
 * ================================================================ */
static const char *TAG_LCD = "LCD_TASK";

void task_LCD(void *pvParameters)
{
    const TickType_t period   = pdMS_TO_TICKS(200);
    TickType_t       wakeTime = xTaskGetTickCount();

    // One-time splash screen on startup.
    lcd.LCD_Clear();
    lcd.LCD_move_cursor(1, 1);
    lcd.LCD_Display_string("  Motor  Ctrl  ");
    lcd.LCD_move_cursor(2, 1);
    lcd.LCD_Display_string("  Initialising  ");
    ESP_LOGI(TAG_LCD, "[1] '  Motor  Ctrl  '");
    ESP_LOGI(TAG_LCD, "[2] '  Initialising  '");
    vTaskDelay(pdMS_TO_TICKS(1500));
    lcd.LCD_Clear();

    ESP_LOGI(TAG_LCD, "Started on core %d  period=200ms", xPortGetCoreID());

    bool brake_displayed = false;   // tracks whether the brake screen is currently shown (avoid re-drawing every tick)

    while (1)
    {
        bool brake_flag = (xEventGroupGetBits(system_eventgroup) & EMERGENCY_STOP_BIT) != 0;

        if (brake_flag)
        {
            if (!brake_displayed)
            {
                // Draw the brake screen only once on entry, not every 200ms tick.
                lcd.LCD_Clear();
                lcd.LCD_move_cursor(1, 1);
                lcd.LCD_Display_string("  !! BRAKE !!   ");
                lcd.LCD_move_cursor(2, 1);
                lcd.LCD_Display_string("   Speed = 0    ");
                brake_displayed = true;
                ESP_LOGI(TAG_LCD, "Brake screen active");
                ESP_LOGI(TAG_LCD, "[1] '  !! BRAKE !!   '");
                ESP_LOGI(TAG_LCD, "[2] '   Speed = 0    '");
            }
            vTaskDelayUntil(&wakeTime, period);
            continue;   // skip normal sensor/speed display while braking
        }

        if (brake_displayed)
        {
            // Leaving brake state: clear once so the normal display below
            // isn't drawn over leftover brake-screen characters.
            lcd.LCD_Clear();
            brake_displayed = false;
            ESP_LOGI(TAG_LCD, "Brake released — resuming normal display");
        }

        sensor_data_t snap = {0.0f, 0.0f, 0.0f};
        if (xSemaphoreTake(sensor_mutex, pdMS_TO_TICKS(10)) == pdTRUE)
        {
            snap = shared_sensors;   // local copy so we don't hold the mutex during LCD writes
            xSemaphoreGive(sensor_mutex);
        }
        else
        {
            // Mutex unavailable in time — fall back to whatever `snap` was
            // last (all zeros the very first time), rather than blocking.
            ESP_LOGW(TAG_LCD, "sensor_mutex timeout — using stale data");
        }

        int16_t actual_rpm = 0;
        xQueuePeek(enc_queue, &actual_rpm, 0);
        float actual_ms = (float)actual_rpm * RPM_TO_MS;
        if (!isfinite(actual_ms)) actual_ms = 0.0f;

        char row1[17];
        snprintf(row1, sizeof(row1), "V:%5.2f  A:%4.2f",
                 snap.voltage, snap.current);

        char row2[17];
        snprintf(row2, sizeof(row2), "T:%4.1fC S:%4.2f",
                 snap.temperature, actual_ms);

        lcd.LCD_move_cursor(1, 1);
        lcd.LCD_Display_string(row1);
        lcd.LCD_move_cursor(2, 1);
        lcd.LCD_Display_string(row2);
        ESP_LOGI(TAG_LCD, "[1] '%s'", row1);
        ESP_LOGI(TAG_LCD, "[2] '%s'", row2);

        vTaskDelayUntil(&wakeTime, period);
    }
}

/* ================================================================
 *  TASK 6 — task_Buzzer  (stub)
 *  Polls LANE_ALERT_BIT every 50 ms and drives the buzzer GPIO high
 *  while the bit is set. No debouncing/patterning — a simple level
 *  follower.
 * ================================================================ */
void task_Buzzer(void *pvParameters)
{
    gpio_set_direction(BUZZER_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(BUZZER_GPIO, 0);

    while (1)
    {
        EventBits_t bits = xEventGroupGetBits(system_eventgroup);
        if (bits & LANE_ALERT_BIT)
            gpio_set_level(BUZZER_GPIO, 1);
        else
            gpio_set_level(BUZZER_GPIO, 0);

        vTaskDelay(pdMS_TO_TICKS(50));   // 50 ms poll — fast enough for human perception
    }
}