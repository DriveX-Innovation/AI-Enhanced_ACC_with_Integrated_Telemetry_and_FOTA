#pragma once

#include "driver/gpio.h"
#include "driver/mcpwm_prelude.h"
#include "esp_log.h"    
#include "esp_err.h"

// PWM settings
#define RESOLUTION_HZ 2000000                  //  2 MHz
#define PWM_FREQ 20000                         // 20 kHz
#define PERIOD_TICKS (RESOLUTION_HZ / PWM_FREQ) // Auto-calculated based on frequency 
#define PWM_RESOLUTION 12                          // 12-bit input
#define MAX_DUTY_CYCLE ((1 << PWM_RESOLUTION) - 1) // 4095


class Motor
{
private:
    int IN1_pin;
    int IN2_pin;
    int PWM_pin;
    mcpwm_timer_handle_t MotorTimer_handle;
    mcpwm_oper_handle_t MotorOperator_handle;
    mcpwm_gen_handle_t MotorGenerator_handle;
    mcpwm_cmpr_handle_t MotorComparator_handle;

public:
    Motor(int _IN1_pin, int _IN2_pin, int _PWM_pin);
    ~Motor();
    void MotorSetup();
    void SetSpeed(int speed); // speed: 0 - 4095 for 12-bit resolution
    void Forward();
    void Backward();
    void Stop();
    void Brake();
};
