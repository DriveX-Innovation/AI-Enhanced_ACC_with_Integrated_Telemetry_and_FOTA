#include "PID.h"

/*
The PID controller algorithm implemented in this code follows the standard PID control formula:

In continuous form, the PID control law is given by:
u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt

In discrete form, this can be represented as:
u(n) = Kp * e(n) + Ki * sum(e(n) * dt) + Kd * (e(n) - e(n-1)) / dt

where:
- u(n) is the control output at the current time step
- e(n) is the current error (setpoint - measured_value)
- dt is the time difference between the current and previous time step
- Kp, Ki, Kd are the proportional, integral, and derivative gains respectively

The implementation includes:
- Proportional term: Kp * error
- Integral term: Ki * integral (with clamping to prevent windup)
- Derivative term: Kd * derivative
- Output clamping to ensure the control signal remains within defined PWM limits.
- Time management using esp_timer to calculate delta_time for accurate integration and differentiation.
- A Reset function to reinitialize the controller state.
*/

PID::PID(float kp, float ki, float kd)
    : Kp(kp), Ki(ki), Kd(kd)
{
    last_time = esp_timer_get_time();
    previous_error = 0.0f;
}

PID::~PID()
{
}

int PID::Calculate(const float& setpoint, const float& measured_value)
{
    int64_t current_time = esp_timer_get_time();
    // Convert microseconds to seconds
    float delta_time = (current_time - last_time) / 1e6f; 

    last_time = current_time;

    // Prevent division by zero, 1e-3f is a small value to avoid instability
    if (delta_time <= 0)
        delta_time = 1e-3f; 
    
    // Calculate error
    float error = setpoint - measured_value;

    static float integral = 0.0f;
    integral += error * delta_time;

    // Clamp integral to prevent windup
    if(integral > MAX_PWM) 
        integral = MAX_PWM;
    else if(integral < MIN_PWM) 
        integral = MIN_PWM;

    float derivative = (error - previous_error) / delta_time;
    previous_error = error;

    float output = Kp * error + Ki * integral + Kd * derivative;

    // Clamp output to PWM limits
    if (output > MAX_PWM)
        output = MAX_PWM;
    else if (output < MIN_PWM)
        output = MIN_PWM;

    return (int)output;
}

void PID::Reset()
{
    previous_error = 0.0f;
    last_time = esp_timer_get_time();
}
