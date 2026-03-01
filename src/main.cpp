/****************************************************************************/
/****************************- Include Files -*******************************/
/****************************************************************************/
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"

#include "Encoder.h"
#include "VoltSensor.h"
#include "TempSensor.h"
#include "CurrentSensor.h"
#include "LCD.h"


/***************************************************************************/
/****************************- Pins definitions -***************************/
/***************************************************************************/
// Define GPIO pins for the encoder and voltage sensor
#define ENCODER_PIN_A GPIO_NUM_34
#define ENCODER_PIN_B GPIO_NUM_35
#define VOLTAGE_SENSOR_ADC_CHANNEL ADC_CHANNEL_7 // GPIO35 corresponds to ADC_CHANNEL_7
#define TEMP_SENSOR_ADC_CHANNEL ADC_CHANNEL_0    // GPIO36 corresponds to ADC_CHANNEL_0
#define CURRENT_SENSOR_ADC_CHANNEL ADC_CHANNEL_3   // GPIO39 corresponds to ADC_CHANNEL_3  
/***************************************************************************/
/****************************- Application Code -***************************/
/***************************************************************************/
adc_oneshot_unit_handle_t adc_handle = nullptr; // Global ADC handle for one-shot mode
extern "C" void app_main()
{
    // Initialize ADC unit for voltage sensor and temperature sensor
    adc_oneshot_unit_init_cfg_t unit_config =
        {
            .unit_id = ADC_UNIT_1,
            .clk_src = ADC_RTC_CLK_SRC_DEFAULT, // Use default RTC clock source which is 8MHz
            .ulp_mode = ADC_ULP_MODE_DISABLE,
        };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_config, &adc_handle));

    // Create instances of the Encoder and VoltageSensor classes
    VoltageSensor voltageSensor(VOLTAGE_SENSOR_ADC_CHANNEL, adc_handle);
    TempSensor tempSensor(TEMP_SENSOR_ADC_CHANNEL, adc_handle);
    CurrentSensor currentSensor(CURRENT_SENSOR_ADC_CHANNEL, adc_handle);
    LCD lcd;


    // Initialize the voltage sensor
    voltageSensor.voltageSensorSetup();
    // Initialize the temperature sensor
    tempSensor.tempSensorSetup();
    // Initialize the current sensor
    currentSensor.currentSensorSetup();
    // Initialize the LCD
    //lcd.LCD_Setup();

    while (true) 
    {
        // Read and print the calibrated voltage in millivolts
        uint32_t input_voltage_mV = voltageSensor.readInputMilliVolts();
        float temperature_celsius = tempSensor.readTemperatureCelsius();
        float current_amperes = currentSensor.readCurrent();
        printf("Voltage: %lu mV, Temperature: %.2f C, Current: %.2f A\n", input_voltage_mV, temperature_celsius, current_amperes);
        // Display the readings on the LCD
        /*lcd.LCD_Clear();
        lcd.LCD_move_cursor(0, 0);
        lcd.LCD_Display_string("V:");
        lcd.LCD_Display_integer(input_voltage_mV);
        lcd.LCD_Display_string("mV");
        lcd.LCD_move_cursor(1, 0);
        lcd.LCD_Display_string("T:");
        lcd.LCD_Display_float(temperature_celsius);
        lcd.LCD_Display_string("C ");
        lcd.LCD_Display_string("I:");
        lcd.LCD_Display_float(current_amperes);
        lcd.LCD_Display_string("A");*/
        // Add a delay before the next reading
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1 second
    }

}