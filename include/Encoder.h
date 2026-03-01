#pragma once

#include "driver/pulse_cnt.h"
#include "hal/pcnt_types.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_err.h"

#define PPR 11                  // Pulses per revolution for the encoder
#define GEAR_RATIO 34.0f        // Gear ratio of the motor gearbox
#define MIN_SPEED 0.0f             // Minimum speed in RPM
#define MAX_SPEED 210.0f        // Maximum speed in RPM
#define MIN_DIGITAL_OUTPUT 700.0f
#define MAX_DIGITAL_OUTPUT 4095.0f
#define SAMPLE_PERIOD_US 20000 // 20 ms

enum EncoderDirection
{
    ENCODER_STOPPED = 0,
    ENCODER_FORWARD = 1,
    ENCODER_REVERSE = -1
};

class Encoder
{
private:
    int EncoderPinA;
    int EncoderPinB;
    pcnt_unit_handle_t EncoderPCNT_handle_Pin;
    pcnt_channel_handle_t EncoderPCNT_channel_PinA;
    pcnt_channel_handle_t EncoderPCNT_channel_PinB;

    int lastCount;
    int64_t lastTime_us = 0;

public : 
    Encoder(int encoderpinA, int encoderpinB);
    ~Encoder();
    void EncoderSetup();
    int GetEncoderCount();
    int CalculateSpeed();
    void ResetEncoderCount();
    int GetEncoderSpeedDigitalOutput();
    int GetEncoderDirection();
};