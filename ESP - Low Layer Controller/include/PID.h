#pragma once

#include <cmath>
#include <cstddef>   // for nullptr


#define MAX_PWM 210
#define MIN_PWM 0
#define MIN_RPM  0.0f
#define MAX_RPM  210.0f
//#define MAX_RPM  230.0f

#define KFF 0.8f

typedef struct 
{
    //Controller Gains
    float Kp;
    float Ki;
    float Kd;

    float Tau; //Derivative Low Pass filter Time Constant

    // //integrator limits
    // float Min_Integrator_Limit;
    // float Max_Integrator_Limit;

    // //Output Limits
    // float Min_DutyCycle;
    // float Max_DutyCycle;

    float Sampling_Time; //Sampling time in second

    //Controller Memory
    float PrevError;
    float PrevMeasurement;
    float Integrator;
    float Differentiator;

    float PID_Ouput;
}PID_Data;

void pid_init(PID_Data *pid);

//calculate the new value of RPM that should be applied
float pid_CalcPWM(PID_Data *pid , float Setpoint, float Measured_Value);


// class pid
// {
//     private:
    
//         //Controller Gains
//         float Kp;
//         float Ki;
//         float Kd;

//         float Tau; //Derivative Low Pass filter Time Constant

//         //Output Limits
//         float LimMin;
//         float LimMax;

//         float Samp_Time; //Sampling time in second

//         //Controller Memory
//         float PrevError;
//         float PrevMeasurement;
//         float Integrator;
//         float Differentiator;

//         float PID_Ouput;

//     public:

//         pid(float kp, float ki, float kd); //constructor

//         ~pid(); //destructor

//         void pid_init();
// };