#include "PID.h"

void pid_init(PID_Data *pid)
{
    if (pid == nullptr) return;

    //initially clear all variables
    pid->Integrator=0.0f;
    pid->Differentiator=0.0f;
    pid->PrevError=0.0f;
    pid->PrevMeasurement=0.0f;

    pid->PID_Ouput=0.0f;
}

int rpm_To_pwm(float Value, float Min_Value, float Max_Value, float Required_Min, float Required_Max)
{
    return (Value - Min_Value) * (Required_Max - Required_Min) / (Max_Value - Min_Value) + Required_Min;
}

float pid_CalcPWM(PID_Data *pid , float Setpoint, float Measured_Value)
{
    if (pid == nullptr) return 0.0f;

    float Current_error=Setpoint-Measured_Value;

    float Proportional= pid->Kp * Current_error;

    // apply low pass filter 
    //CurrentError - PrevError = (Setpoint - measured2)-(Setpoint - measured1)
    //                         = -measured2+measured1 (Note: measured 1 is the prevmeasure)
    //                         = -(measured2 - prevMeasure) 
    pid->Differentiator = -(2.0f * pid->Kd * (Measured_Value - pid->PrevMeasurement)
                    + (2.0f * pid->Tau - pid->Sampling_Time) * pid->Differentiator)
                    / (2.0f * pid->Tau + pid->Sampling_Time);


    float Integrator_temp = pid->Integrator + 0.5f * pid->Ki * pid->Sampling_Time * (Current_error + pid->PrevError);
    //dynamic integrator clamping to avoid wind-up
    if (Integrator_temp > MAX_PWM)
    {
        Integrator_temp = MAX_PWM;
    }
    else if (Integrator_temp < MIN_PWM)
    {
        Integrator_temp = MIN_PWM;
    }

    //Calculate the unsaturated Output                    
    float Unsaturated_Output = Proportional + Integrator_temp + pid->Differentiator;

    float output = Unsaturated_Output;
    if(output > MAX_RPM)
    {
        output = MAX_RPM;
    }
    else if (output < MIN_RPM)
    {
        output = MIN_RPM;
    }

    //if(Unsaturated_Output == output) //comparison of floats never get true
    if (fabs(Unsaturated_Output - output) < 0.0001f)
    {
        // Only accept integrator update if NOT saturated
        pid->Integrator=Integrator_temp;
    }

    // Save states
    pid->PrevError=Current_error; //put the current error in the previous error
    pid->PrevMeasurement=Measured_Value; // put the current measured value in the prevmeasurment
    pid->PID_Ouput=output;

    float feedforward = KFF * Setpoint;  // direct RPM estimate
    pid->PID_Ouput += feedforward ;

    //map rpm(0 --> 210) to pwm(0 --> 4095) then return the mapped value
    return (int)rpm_To_pwm(pid->PID_Ouput,MIN_RPM,MAX_RPM,0.0f,4095.0f);

    //return pid->PID_Ouput;
}