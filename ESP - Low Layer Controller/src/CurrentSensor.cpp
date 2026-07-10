#include "CurrentSensor.h"

CurrentSensor::CurrentSensor(adc_channel_t channel, adc_oneshot_unit_handle_t adc_handle) : 
    Currentsensor_handle(adc_handle), adc_channel(channel),
    adc_cali_handle(nullptr), calibrated(false)
{
}

CurrentSensor::~CurrentSensor()
{
    // Check if the ADC handle is valid before attempting to delete it
    if (Currentsensor_handle != nullptr)
    {
        ESP_ERROR_CHECK(adc_oneshot_del_unit(Currentsensor_handle));
        Currentsensor_handle = nullptr; // Set to nullptr after deletion
    }
    // Check if the ADC calibration handle is valid before attempting to delete it
    if (adc_cali_handle != nullptr)
    {
        ESP_ERROR_CHECK(adc_cali_delete_scheme_line_fitting(adc_cali_handle));
        adc_cali_handle = nullptr; // Set to nullptr after deletion
    }
}

esp_err_t CurrentSensor::currentSensorSetup()
{
    // Channel configuration for voltage measurement ( Specifically GPIO35 corresponds to ADC_CHANNEL_7 )
    adc_oneshot_chan_cfg_t channel_config =
        {
            .atten = ADC_ATTEN_DB_12, // Set attenuation to 11dB for a wider voltage range (up to ~3.3V)
            .bitwidth = ADC_BITWIDTH_12,
        };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(Currentsensor_handle, adc_channel, &channel_config));

    // ADC calibration setup using line fitting method
    adc_cali_line_fitting_config_t cali_config =
        {
            .unit_id = ADC_UNIT_1,
            .atten = ADC_ATTEN_DB_12, // Must match the attenuation used in channel configuration
            .bitwidth = ADC_BITWIDTH_12,
        };

    // Create the ADC calibration scheme and check if it was successful
    esp_err_t ret = adc_cali_create_scheme_line_fitting(&cali_config, &adc_cali_handle);
    calibrated = (ret == ESP_OK);
    return ret;
}

uint32_t CurrentSensor::readRawMilliVolts()
{
    int raw_value = 0;
    int voltage_mV = 0;
    int sum_mv = 0;

    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        ESP_ERROR_CHECK(adc_oneshot_read(Currentsensor_handle, adc_channel, &raw_value));

        if (calibrated)
            adc_cali_raw_to_voltage(adc_cali_handle, raw_value, &voltage_mV);
        else
            voltage_mV = (raw_value * ADC_VREF) / ADC_RESOLUTION;

        sum_mv += voltage_mV;
    }

    return (uint32_t)(sum_mv / SAMPLE_COUNT);
}


void CurrentSensor::calibrateZeroOffset()
{
    uint64_t sum = 0;
    const int calibration_samples = 200;

    for (int i = 0; i < calibration_samples; i++)
    {
        sum += readRawMilliVolts();
    }

    zero_offset_mV = (float)(sum) / calibration_samples;
}

uint32_t CurrentSensor::readCalibratedMilliVolts()
{
    int raw_value = 0;
    int voltage_mV = 0;
    int sum_mv = 0;
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        ESP_ERROR_CHECK(adc_oneshot_read(Currentsensor_handle, adc_channel, &raw_value));
        if (calibrated)
        {
            adc_cali_raw_to_voltage(adc_cali_handle, raw_value, &voltage_mV);
        }
        else
        {
            voltage_mV = (raw_value * ADC_VREF) / ADC_RESOLUTION;
        }
        sum_mv += voltage_mV;
    }
    return sum_mv / SAMPLE_COUNT;
}

float CurrentSensor::readCurrent()
{
    float voltage_mV = (float)readCalibratedMilliVolts();
    // ACS712: 100mV per Amp (for 30A version)
    return (voltage_mV - zero_offset_mV) / ACS712_MV_PER_AMP;
}

CurrentSensorStatus CurrentSensor::diagnose(float current)
{
    if (current > MAX_CURRENT_A)
        return CurrentSensorStatus::OVER_CURRENT;

    if (current < -0.2f)
        return CurrentSensorStatus::UNDER_CURRENT;

    if (zero_offset_mV < 1000 || zero_offset_mV > 2300)
        return CurrentSensorStatus::SENSOR_DISCONNECTED;

    if (current > -0.05f && current < 0.05f)
        return CurrentSensorStatus::OK;

    return CurrentSensorStatus::OK;
}