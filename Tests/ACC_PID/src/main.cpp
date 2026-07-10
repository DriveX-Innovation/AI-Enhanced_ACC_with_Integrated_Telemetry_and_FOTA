#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"

#include "Throttle.h"
#include "Motor.h"
#include "PID.h"
#include "Encoder.h"

// Pin definitions
#define INA_PIN GPIO_NUM_25
#define INB_PIN GPIO_NUM_32
#define PWM_PIN GPIO_NUM_33
#define ENCODER_PIN_A GPIO_NUM_26
#define ENCODER_PIN_B GPIO_NUM_27
// Configuration constants
// Encoder task delay in milliseconds
#define ENCODER_TASK_DELAY_MS 20
// PID constants
#define PID_KP 3.0f
#define PID_KI 0.05f
#define PID_KD 0.0013f
#define PID_TASK_DELAY_MS 50
// Throttle task delay in milliseconds
#define THROTTLE_TASK_DELAY_MS 25
// Motor task delay in milliseconds
#define MOTOR_TASK_DELAY_MS 5

int targetSpeed = 0;
int motorSpeed = 0;
int speed = 0;

extern "C" void app_main()
{
    ESP_LOGI("Application", "Starting PID Control Application");
    static Motor motor(INA_PIN, INB_PIN, PWM_PIN);
    static PID pid(PID_KP, PID_KI, PID_KD);
    static Encoder encoder(ENCODER_PIN_A, ENCODER_PIN_B);
    static Throttle throttle;

    motor.MotorSetup();
    throttle.ThrottleSetup();
    motor.Forward();
    encoder.EncoderSetup();

    while(1)
    {
        // Read throttle and calculate speed
        targetSpeed = throttle.ReadThrottle();
        // Read encoder speed
        motorSpeed = encoder.GetEncoderSpeedDigitalOutput();
        // Compute PID output
        speed = pid.Calculate(targetSpeed, motorSpeed);
        printf("%d,%d\n", targetSpeed, speed);
        // Set motor speed
        motor.SetSpeed(speed);
        // Delay for next iteration
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}