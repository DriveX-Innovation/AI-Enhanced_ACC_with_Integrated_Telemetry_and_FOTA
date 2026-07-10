#pragma once

#include <stdint.h>
#include <stdio.h>
#include "driver/gpio.h"
#include "esp_rom_sys.h"

// ─── Pin Definitions ────────────────────────────────────────────────────────
#define LCD_RS GPIO_NUM_22
#define LCD_EN GPIO_NUM_23
#define LCD_D4 GPIO_NUM_5
#define LCD_D5 GPIO_NUM_18
#define LCD_D6 GPIO_NUM_19
#define LCD_D7 GPIO_NUM_21

// ─── LCD Dimensions ─────────────────────────────────────────────────────────
#define LCD_ROWS 2
#define LCD_COLUMNS 16

// ─── LCD Commands ────────────────────────────────────────────────────────────
#define LCD_CMD_CLEAR 0x01        // Clear display
#define LCD_CMD_HOME 0x02         // Return cursor home
#define LCD_CMD_ENTRY_MODE 0x06   // Increment cursor, no display shift
#define LCD_CMD_DISPLAY_ON 0x0C   // Display on, cursor off, blink off
#define LCD_CMD_DISPLAY_OFF 0x08  // Display off
#define LCD_CMD_FUNCTION_SET 0x28 // 4-bit mode, 2 lines, 5x8 font
#define FOUR_BITS_1 0x20          // Switch to 4-bit mode

// ─── LCD Class ───────────────────────────────────────────────────────────────
class LCD
{
public:
    LCD();
    ~LCD();

    /**
     * @brief  Initialize GPIO pins and run the LCD power-on sequence.
     *         Must be called once before any other LCD function.
     */
    void LCD_Setup();

    /**
     * @brief  Send a command byte to the LCD (RS = 0).
     * @param  cmd  Command byte to transmit (sent as two 4-bit nibbles).
     */
    void LCD_Send_Cmd(uint8_t cmd);

    /**
     * @brief  Move the cursor to the specified row and column (1-indexed).
     * @param  row  Row number (1 or 2).
     * @param  col  Column number (1 – LCD_COLUMNS).
     */
    void LCD_move_cursor(uint8_t row, uint8_t col);

    /**
     * @brief  Clear the display and wait for the operation to complete.
     */
    void LCD_Clear();

    /**
     * @brief  Write a single character at the current cursor position (RS = 1).
     * @param  data  ASCII character to display.
     */
    void LCD_Display_char(const char data);

    /**
     * @brief  Write a null-terminated string starting at the current cursor position.
     * @param  data  Pointer to the string to display.
     */
    void LCD_Display_string(const char *data);

    /**
     * @brief  Write a signed integer at the current cursor position.
     * @param  number  Integer value to display.
     */
    void LCD_Display_integer(int number);

    /**
     * @brief  Write a floating-point number (2 decimal places) at the current cursor position.
     * @param  number  Float value to display.
     */
    void LCD_Display_float(float number);

private:
    /**
     * @brief  Toggle the Enable pin to latch the current data/command nibble.
     *         Called internally after setting the data lines.
     */
    void LCD_enable();
};