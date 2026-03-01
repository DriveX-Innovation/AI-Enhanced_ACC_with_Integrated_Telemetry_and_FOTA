#include "TempSensor.h"

/*
This code implements a temperature sensor class for the ESP32 microcontroller using the ADC (Analog to Digital Converter) to read voltage values from a temperature sensor (like the LM35) and convert them to temperature in Celsius. 
The class includes methods for setting up the sensor, reading calibrated voltage values, converting those values to temperature, and diagnosing the sensor status.
Key components of the implementation:
- ADC initialization and configuration for one-shot reading mode.
- ADC calibration using the line fitting method to improve accuracy.
- Averaging multiple samples to reduce noise and improve stability of readings.
- Detection of sensor floating by checking the variation in voltage readings.
- Conversion of voltage to temperature based on the characteristics of the LM35 sensor (10mV per degree Celsius).
- Diagnostic method to check if the sensor is disconnected, over temperature, under temperature, or operating normally.
- Proper resource management in the destructor to clean up ADC handles and calibration resources.
- Use of ESP-IDF error handling macros to ensure robust operation and easy debugging.
Note: The code assumes that the ADC channel used for the temperature sensor is properly connected to the LM35 or similar sensor, 
and that the ESP-IDF environment is correctly set up for development.
The implementation also includes constants for maximum and minimum expected temperatures, as well as a noise threshold for detecting floating conditions.
*/

TempSensor::TempSensor(adc_channel_t channel, adc_oneshot_unit_handle_t adc_handle) : 
    Tempsensor_handle(adc_handle), adc_channel(channel), 
    adc_cali_handle(nullptr), calibrated(false)
{
}

TempSensor::~TempSensor()
{
    // Check if the ADC handle is valid before attempting to delete it
    if (Tempsensor_handle != nullptr)
    {
        ESP_ERROR_CHECK(adc_oneshot_del_unit(Tempsensor_handle));
        Tempsensor_handle = nullptr; // Set to nullptr after deletion
    }
    // Check if the ADC calibration handle is valid before attempting to delete it
    if (adc_cali_handle != nullptr)
    {
        ESP_ERROR_CHECK(adc_cali_delete_scheme_line_fitting(adc_cali_handle));
        adc_cali_handle = nullptr; // Set to nullptr after deletion
    }
}

void TempSensor::tempSensorSetup()
{
    // Channel configuration for voltage measurement ( Specifically GPIO35 corresponds to ADC_CHANNEL_7 )
    adc_oneshot_chan_cfg_t channel_config =
        {
            .atten = ADC_ATTEN_DB_12, // Set attenuation to 11dB for a wider voltage range (up to ~3.3V)
            .bitwidth = ADC_BITWIDTH_12,
        };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(Tempsensor_handle, adc_channel, &channel_config));

    // ADC calibration setup using line fitting method
    adc_cali_line_fitting_config_t cali_config =
        {
            .unit_id = ADC_UNIT_1,
            .atten = ADC_ATTEN_DB_12, // Must match the attenuation used in channel configuration
            .bitwidth = ADC_BITWIDTH_12,
            .default_vref = 1100, // Default reference voltage in mV (adjust based on your hardware)
        };

    // Create the ADC calibration scheme and check if it was successful
    esp_err_t ret = adc_cali_create_scheme_line_fitting(&cali_config, &adc_cali_handle);
    calibrated = (ret == ESP_OK);
}

uint32_t TempSensor::readCalibratedMilliVolts()
{
    int raw_value = 0;
    int voltage_mV = 0;
    int sum_mv = 0;
    int min_mv = 100000;
    int max_mv = 0;

    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        ESP_ERROR_CHECK(adc_oneshot_read(Tempsensor_handle, adc_channel, &raw_value));

        if (calibrated)
        {
            adc_cali_raw_to_voltage(adc_cali_handle, raw_value, &voltage_mV);
        }
        else
        {
            voltage_mV = (raw_value * ADC_VREF) / ADC_RESOLUTION;
        }

        sum_mv += voltage_mV;

        if (voltage_mV < min_mv)
            min_mv = voltage_mV;
        if (voltage_mV > max_mv)
            max_mv = voltage_mV;
    }

    if ((max_mv - min_mv) > FLOATING_VARIATION_MV)
    {
        return -1; // Sensor floating
    }

    return sum_mv / SAMPLE_COUNT;
}

float TempSensor::readTemperatureCelsius()
{
    int voltage_mV = readCalibratedMilliVolts();

    if (voltage_mV < 0)
        return -1000.0f; // Error value

    // LM35: 10mV per °C
    return voltage_mV / LM35_MV_PER_DEGREE;
}

TempSensorStatus TempSensor::diagnose(float temperature)
{
    if (temperature < -100.0f)
    {
        return TempSensorStatus::SENSOR_DISCONNECTED;
    }

    if (temperature > MAX_TEMP_C)
    {
        return TempSensorStatus::OVER_TEMPERATURE;
    }

    if (temperature < MIN_TEMP_C)
    {
        return TempSensorStatus::UNDER_TEMPERATURE;
    }

    return TempSensorStatus::OK;
}