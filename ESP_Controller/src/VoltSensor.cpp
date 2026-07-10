#include "VoltSensor.h"

/*
This code defines the implementation of the VoltageSensor class, which is responsible for reading and calibrating voltage measurements from an ADC channel on an ESP32 microcontroller. 
The class uses the ESP-IDF ADC driver to configure the ADC unit and channel, and it also sets up calibration using the line fitting method.
Key functionalities include:
- Constructor and Destructor: The constructor initializes member variables, while the destructor ensures that any allocated resources
    (ADC handles) are properly released.
- voltageSensorSetup(): This method initializes the ADC unit and configures the specified ADC channel for voltage measurement.
    It also sets up the calibration scheme using line fitting.
- readCalibratedMilliVolts(): This method reads raw ADC values multiple times, converts them to millivolts using the calibration data,
    and returns the average voltage. It also checks for floating ADC conditions by comparing the maximum and minimum readings.
- readInputMilliVolts(): This method reads the calibrated voltage and applies a voltage divider ratio to calculate the actual input voltage.
- diagnose(): This method checks the input voltage against defined thresholds to determine if there is an over
    voltage condition or if the sensor is open-circuit (disconnected).
- readInputWithStatus(): This method reads the input voltage and returns both the voltage value and a status indicating 
    if the reading is valid, over-voltage, or open-circuit.
The implementation ensures that resources are managed correctly and that the voltage readings are accurate and reliable, with built-in diagnostics for common issues.
*/

VoltageSensor::VoltageSensor(adc_channel_t channel, adc_oneshot_unit_handle_t adc_handle) : 
    Voltsensor_handle(adc_handle), adc_channel(channel),
    adc_cali_handle(nullptr), calibrated(false)
{   
}


VoltageSensor::~VoltageSensor() 
{
    // Check if the ADC handle is valid before attempting to delete it
    if (Voltsensor_handle != nullptr)
    {
        ESP_ERROR_CHECK(adc_oneshot_del_unit(Voltsensor_handle));
        Voltsensor_handle = nullptr; // Set to nullptr after deletion
    }
    // Check if the ADC calibration handle is valid before attempting to delete it
    if (adc_cali_handle != nullptr) 
    {
        ESP_ERROR_CHECK(adc_cali_delete_scheme_line_fitting(adc_cali_handle));
        adc_cali_handle = nullptr; // Set to nullptr after deletion
    }
}

void VoltageSensor::voltageSensorSetup()
{
    // Channel configuration for voltage measurement ( Specifically GPIO35 corresponds to ADC_CHANNEL_7 )
    adc_oneshot_chan_cfg_t channel_config = 
    {
        .atten = ADC_ATTEN_DB_12, // Set attenuation to 11dB for a wider voltage range (up to ~3.3V)
        .bitwidth = ADC_BITWIDTH_12, 
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(Voltsensor_handle, adc_channel, &channel_config));

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

uint32_t VoltageSensor::readCalibratedMilliVolts()
{
    int raw_value = 0;
    int voltage_mV = 0;
    int sum_mv = 0;
    int min_mv = 100000;
    int max_mv = 0;
    // Take multiple samples and average them to reduce noise
    for (int i = 0; i < SAMPLE_COUNT; i++) 
    {
        // Read raw ADC value
        ESP_ERROR_CHECK(adc_oneshot_read(Voltsensor_handle, adc_channel, &raw_value));
        if (calibrated) 
        {
            adc_cali_raw_to_voltage(adc_cali_handle, raw_value, &voltage_mV); // Convert raw ADC value to calibrated voltage in mV
        } 
        else 
        {
            // If calibration failed, return uncalibrated voltage in mV
            voltage_mV = (raw_value * ADC_VREF_VOLTDRIVER) / ADC_RESOLUTION; // Convert to mV
        }
        sum_mv += voltage_mV;
        if (voltage_mV < min_mv)
            min_mv = voltage_mV;
        if (voltage_mV > max_mv)
            max_mv = voltage_mV;
    }
    
    if ((max_mv - min_mv) > FLOATING_VARIATION_MV)
    {
        return -1; // Special value indicating floating ADC
    }

    return sum_mv / SAMPLE_COUNT; // Return the average calibrated voltage in mV
}

uint32_t VoltageSensor::readInputMilliVolts()
{
    uint32_t adc_mv = readCalibratedMilliVolts();
    return adc_mv * CALIBRATION_FACTOR / VOLTAGE_DIVIDER_RATIO; // scaled to real input voltage
}

VoltageSensorStatus VoltageSensor::diagnose(uint32_t input_mv)
{
    // ---- Over-voltage detection ----
    if (input_mv >= MAX_INPUT_MV)
    {
        return VoltageSensorStatus::OVER_VOLTAGE;
    }

    // ---- Open-circuit (disconnected input) ----
    if (input_mv < MIN_VALID_MV)
    {
        return VoltageSensorStatus::OPEN_CIRCUIT;
    }

    return VoltageSensorStatus::OK;
}

VoltageSensorStatus VoltageSensor::readInputWithStatus(uint32_t &voltage_mv)
{
    int adc_mv = readCalibratedMilliVolts();

    if (adc_mv < 0)
    {
        return VoltageSensorStatus::OPEN_CIRCUIT;
    }

    // Apply divider ratio (×5)
    voltage_mv = adc_mv * 5;

    return diagnose(voltage_mv);
}