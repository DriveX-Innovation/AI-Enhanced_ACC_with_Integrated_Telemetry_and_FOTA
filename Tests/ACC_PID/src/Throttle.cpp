#include "Throttle.h"

/*
    Throttle class implementation for reading throttle position via ADC.
    Uses ESP32 ADC OneShot API to read analog values from a specified channel.
    Assumes throttle is connected to GPIO34 (ADC_CHANNEL_6).
    Provides methods to read raw ADC value, voltage, and percentage.
    Default VREF is set to 3.3V and default ADC resolution is 12 bits.
    Features:
    - Constructor initializes ADC handle and channel.
    - Destructor cleans up ADC resources.
    - ThrottleSetup() configures the ADC unit and channel.
    - ReadThrottle() returns raw ADC value.
    - ReadVoltage() converts raw ADC value to voltage.
    - ReadPercentage() converts raw ADC value to percentage (0-100%).
    - Constants defined for default VREF, ADC bits, and max percentage.
    - Macro to calculate max ADC value based on bits.
    - Error handling via ESP_ERROR_CHECK macro.
    - Designed for use in embedded systems with ESP32 microcontroller.
*/


Throttle::Throttle()
    : throttle_adc_handle(nullptr), throttle_channel(ADC_CHANNEL_6) // GPIO34 corresponds to ADC_CHANNEL_6
{
}

Throttle::~Throttle()
{
    // Clean up ADC resources if necessary
    if (throttle_adc_handle != nullptr)
    {
        ESP_ERROR_CHECK(adc_oneshot_del_unit(throttle_adc_handle));
    }
}

void Throttle::ThrottleSetup()
{
    // Initialize ADC unit
    adc_oneshot_unit_init_cfg_t unit_cfg =
    {
        .unit_id = ADC_UNIT_1,
        .clk_src = ADC_RTC_CLK_SRC_DEFAULT, // Use default RTC clock source which is 8MHz
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };

    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &throttle_adc_handle));

    // Configure ADC channel for throttle ( Specifically GPIO34 )
    adc_oneshot_chan_cfg_t throttle_adc_config = 
    {
        .atten = ADC_ATTEN_DB_12, // 0-3.3V range
        .bitwidth = ADC_BITWIDTH_12 // 12-bit resolution, 0-4095
    };

    ESP_ERROR_CHECK(adc_oneshot_config_channel(throttle_adc_handle, throttle_channel, &throttle_adc_config));
}

int Throttle::ReadThrottle()
{
    int raw = 0;
    ESP_ERROR_CHECK(adc_oneshot_read(throttle_adc_handle, throttle_channel, &raw));
    return raw;
}

float Throttle::ReadVoltage()
{
    int raw = ReadThrottle();
    return (raw * THROTTLE_DEFAULT_VREF) / THROTTLE_MAX_ADC_VALUE(THROTTLE_DEFAULT_BITS); // Convert to volts
}

float Throttle::ReadPercentage()
{
    int raw = ReadThrottle();
    return (raw * THROTTLE_MAX_PERCENTAGE) / THROTTLE_MAX_ADC_VALUE(THROTTLE_DEFAULT_BITS); // Convert to %
}
