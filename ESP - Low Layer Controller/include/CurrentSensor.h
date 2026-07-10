#pragma once
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_err.h"
#include "hal/adc_types.h"
#include "driver/gpio.h"

#define ADC_RESOLUTION 4095 // 12-bit
#define ADC_VREF 1100       // Reference voltage in mV


// Diagnostic thresholds
enum class CurrentSensorStatus
{
    OK,
    OVER_CURRENT,
    UNDER_CURRENT,
    SENSOR_DISCONNECTED,
    SENSOR_FLOATING
};

class CurrentSensor
{

private:
    adc_oneshot_unit_handle_t Currentsensor_handle;           // ADC handle for one-shot reading
    adc_channel_t adc_channel;                                // ADC channel for voltage measurement
    adc_cali_handle_t adc_cali_handle;                        // ADC calibration handle

    bool calibrated;

    static constexpr int SAMPLE_COUNT = 64;

    // ACS712 5A powered at 5V
    static constexpr float ACS712_MV_PER_AMP = 185.0f; // scaled for 3.3V
    static constexpr float MAX_CURRENT_A = 5.0f;
    static constexpr float MIN_CURRENT_A = 0.0f;

    float zero_offset_mV;

    uint32_t readRawMilliVolts();

public:
    // Constructor and Destructor
    CurrentSensor(adc_channel_t channel, adc_oneshot_unit_handle_t adc_handle);
    ~CurrentSensor();

    esp_err_t currentSensorSetup();
    void calibrateZeroOffset();

    uint32_t readCalibratedMilliVolts();
    float readCurrent();
    CurrentSensorStatus diagnose(float current);
};