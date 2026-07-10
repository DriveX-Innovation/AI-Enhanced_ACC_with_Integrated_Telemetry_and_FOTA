#pragma once

#include "esp_timer.h"

#define MAX_PWM 4095
#define MIN_PWM 0

class PID
{
private:
    const float Kp;
    const float Ki;
    const float Kd;
    int64_t last_time;
    float previous_error;
public:
    PID(float kp, float ki, float kd);
    ~PID();
    int Calculate(const float& setpoint, const float& measured_value);
    void Reset();
};