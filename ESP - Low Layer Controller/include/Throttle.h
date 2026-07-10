#pragma once

#include "esp_adc/adc_oneshot.h"
#include "esp_err.h"
#include "hal/adc_types.h"
#include "driver/gpio.h"

#define THROTTLE_DEFAULT_VREF 3.3f // Default reference voltage for ADC
#define THROTTLE_MAX_PERCENTAGE 100.0f
#define THROTTLE_DEFAULT_BITS 12   // Default ADC resolution in bits
#define THROTTLE_MAX_ADC_VALUE(bits) ((1 << (bits)) - 1) // Max ADC value based on bits


#define MINIMUM_ADC 0.0f
#define MAXIMUM_ADC 4095.0f

#define MINIMUM_DIGITAL 700.0f
#define MAXIMUM_DIGITAL 4095.0f

// #define MAX_SPEED 210.0f // Maximum speed in RPM
#define MAX_SPEED 140.0f // Maximum speed in RPM
#define MIN_SPEED 0.0f   // Minimum speed in RPM



class Throttle
{
private:
    adc_oneshot_unit_handle_t throttle_adc_handle;
    adc_channel_t throttle_channel;

public:
    Throttle(adc_channel_t channel); // Destructor to clean up ADC resources
    ~Throttle();
    void ThrottleSetup();
    int ReadThrottle(); // raw ADC

    float ReadVoltage();    // voltage
    float ReadPercentage(); // 0–100%

    int GetRpmFromThrottle();
};
