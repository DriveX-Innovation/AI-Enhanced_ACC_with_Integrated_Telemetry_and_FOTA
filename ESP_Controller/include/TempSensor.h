#pragma once
#include "esp_err.h"
#include "hal/adc_types.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "driver/gpio.h"

#define ADC_RESOLUTION 4095                   // 12-bit
#define ADC_VREF 1100                        // Reference voltage in mV

// Diagnostic thresholds
enum class TempSensorStatus
{
    OK,
    OVER_TEMPERATURE,
    UNDER_TEMPERATURE,
    SENSOR_DISCONNECTED,
    SENSOR_FLOATING
};

class TempSensor
{

private:
    adc_oneshot_unit_handle_t Tempsensor_handle;           // ADC handle for one-shot reading
    adc_channel_t adc_channel;                             // ADC channel for voltage measurement
    adc_cali_handle_t adc_cali_handle;                     // ADC calibration handle
    bool calibrated;                                       // Flag to indicate if calibration was successful
    static constexpr int SAMPLE_COUNT = 64;                // Number of samples for averaging
    constexpr static uint32_t FLOATING_VARIATION_MV = 100; // Noise threshold for floating detection
    // LM35 specific constants
    constexpr static float MAX_TEMP_C = 150.0f;            // Maximum expected temperature in Celsius
    constexpr static float MIN_TEMP_C = -10.0f;            // Minimum expected temperature
    constexpr static float LM35_MV_PER_DEGREE = 10.0f;           // LM35 outputs 10mV per degree Celsius

public:
    // Constructor and Destructor
    TempSensor(adc_channel_t channel, adc_oneshot_unit_handle_t adc_handle);
    ~TempSensor();

    // Method to initialize the temperature sensor
    void tempSensorSetup();

    // Method to read temperature value
    uint32_t readCalibratedMilliVolts();
    // Get temperature in Celsius based on the voltage reading
    float readTemperatureCelsius();
    // Diagnostic method to check sensor status
    TempSensorStatus diagnose(float temperature);
};