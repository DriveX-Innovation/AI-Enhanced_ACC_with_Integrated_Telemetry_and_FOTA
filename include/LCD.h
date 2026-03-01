#pragma once

#include "esp_lcd_io_i80.h"
#include "hal/lcd_types.h"
#include "esp_lcd_types.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_err.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include <stdio.h>

#define LCD_MODE 8 // THIS CAN BE 8 OR 16, DEPENDING ON THE LCD MODE YOU ARE USING. IN THIS CASE, IT IS SET TO 8-BIT MODE.
// data pins for 8-bit mode
#define LCD_D0 GPIO_NUM_18
#define LCD_D1 GPIO_NUM_19
#define LCD_D2 GPIO_NUM_5
#define LCD_D3 GPIO_NUM_21
#define LCD_D4 GPIO_NUM_22
#define LCD_D5 GPIO_NUM_23
#define LCD_D6 GPIO_NUM_4
#define LCD_D7 GPIO_NUM_0
// control pins
#define LCD_RS GPIO_NUM_8                 // REGISTER SELECT PIN (COMMAND OR DATA)
#define LCD_EN GPIO_NUM_2                 // ENABLE PIN
#define LCD_PIXEL_CLOCK_HZ 1 * 1000 * 1000 // PIXEL CLOCK FREQUENCY (1 MHz)

#define LCD_CMD_FUNCTION_SET 0x38
#define LCD_CMD_DISPLAY_ON 0x0C
#define LCD_CMD_CLEAR 0x01
#define LCD_CMD_ENTRY_MODE 0x06

class LCD
{
private:
    esp_lcd_i80_bus_handle_t i80_bus_handle; // handle for the LCD bus
    esp_lcd_panel_io_handle_t io_handle;     // handle for the LCD panel I/O
    uint8_t address;                         // used for storing the address of the cursor position
    bool initialized;                        // Flag to track if the LCD has been initialized
    static constexpr uint8_t LCD_ROWS = 2;
    static constexpr uint8_t LCD_COLS = 16;

public:
    LCD();
    ~LCD();

    void LCD_Setup();
    void LCD_Clear();
    void LCD_move_cursor(int row, int col);
    void LCD_Display_char(const char data);
    void LCD_Display_string(const char *data);
    void LCD_Display_integer(int number);
    void LCD_Display_float(float number);
};
