#pragma once

#include "driver/pulse_cnt.h"
#include "hal/pcnt_types.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_err.h"

#define PPR 11  // Pulses per revolution for the encoder
#define MIN_SPEED 0
#define MAX_SPEED 150
#define MIN_DIGITAL_OUTPUT 0
#define MAX_DIGITAL_OUTPUT 4095
#define SPEED_FILTER_SIZE 20  // Number of samples for speed filtering

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
    float speedSamples[SPEED_FILTER_SIZE] = {0};
    int speedIndex = 0;

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