#pragma once

#include "esp_err.h"
#include "hal/adc_types.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "driver/gpio.h"

#define ADC_VREF 3.3f
#define ADC_RESOLUTION 4096 // 12-bit resolution
#define VOLTAGE_DIVIDER_RATIO  0.2 // R1 = 30kΩ, R2 = 7.5kΩ, so the ratio is R2 / (R1 + R2) = 7.5kΩ / (30kΩ + 7.5kΩ) = 0.2
#define ADC_VREF_VOLTDRIVER 3300.0
#define CALIBRATION_FACTOR 1

enum class VoltageSensorStatus
{
    OK,
    OVER_VOLTAGE,
    UNDER_VOLTAGE,
    OPEN_CIRCUIT,
    ADC_FAULT
};

class VoltageSensor {

private:
    adc_oneshot_unit_handle_t Voltsensor_handle; // ADC handle for one-shot reading
    adc_channel_t adc_channel; // ADC channel for voltage measurement
    adc_cali_handle_t adc_cali_handle; // ADC calibration handle
    bool calibrated; // Flag to indicate if calibration was successful
    static constexpr int SAMPLE_COUNT = 64; // Number of samples for averaging
    static constexpr uint32_t MAX_INPUT_MV = 16500;        // 16.5V limit
    static constexpr uint32_t MIN_VALID_MV = 500;          // below 0.5V suspicious
    static constexpr uint32_t FLOATING_VARIATION_MV = 200; // noise detection

public:
    // Constructor and Destructor
    VoltageSensor(adc_channel_t channel, adc_oneshot_unit_handle_t adc_handle);
    ~VoltageSensor();

    // Method to initialize the voltage sensor
    void voltageSensorSetup();

    // Method to read voltage value
    uint32_t readCalibratedMilliVolts();
    uint32_t readInputMilliVolts();
    // Method to diagnose voltage sensor status
    VoltageSensorStatus diagnose(uint32_t input_mv);
    VoltageSensorStatus readInputWithStatus(uint32_t &voltage_mv);
};