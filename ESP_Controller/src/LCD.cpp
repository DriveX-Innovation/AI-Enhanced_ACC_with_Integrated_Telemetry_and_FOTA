#include "LCD.h"

// Constructor and Destructor
LCD::LCD()
{
}

LCD::~LCD()
{
}

void LCD::LCD_enable()
{
    // This function can be used to enable the LCD if needed, but in this implementation, the LCD is enabled during setup.
    gpio_set_level(LCD_EN, 1); // Set the Enable pin high to enable the LCD
    esp_rom_delay_us(1);    // Short delay to ensure the LCD is enabled before sending commands
    gpio_set_level(LCD_EN, 0); // Set the Enable pin low after enabling
    esp_rom_delay_us(50);    // Short delay to allow the LCD to process the enable signal
}

// initialize the LCD first by setting up the LCD bus, then the LCD panel I/O, and finally the LCD panel itself. After that, it will clear the LCD screen and set the cursor to the home position (0, 0).
void LCD::LCD_Setup()
{
    // Configure GPIO pins for LCD control and data lines
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LCD_RS) | (1ULL << LCD_EN) | (1ULL << LCD_D4) | (1ULL << LCD_D5) | (1ULL << LCD_D6) | (1ULL << LCD_D7),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf));

    esp_rom_delay_us(200000); // Delay to allow LCD to power up

    int cmd = 0x03;
    gpio_set_level(LCD_D4, (cmd >> 0) & 0x01);
    gpio_set_level(LCD_D5, (cmd >> 1) & 0x01);
    gpio_set_level(LCD_D6, (cmd >> 2) & 0x01);
    gpio_set_level(LCD_D7, (cmd >> 3) & 0x01);
    esp_rom_delay_us(5000);

    gpio_set_level(LCD_D4, (cmd >> 0) & 0x01);
    gpio_set_level(LCD_D5, (cmd >> 1) & 0x01);
    gpio_set_level(LCD_D6, (cmd >> 2) & 0x01);
    gpio_set_level(LCD_D7, (cmd >> 3) & 0x01);
    esp_rom_delay_us(150);

    gpio_set_level(LCD_D4, (cmd >> 0) & 0x01);
    gpio_set_level(LCD_D5, (cmd >> 1) & 0x01);
    gpio_set_level(LCD_D6, (cmd >> 2) & 0x01);
    gpio_set_level(LCD_D7, (cmd >> 3) & 0x01);
    esp_rom_delay_us(150);

    LCD_Send_Cmd(FOUR_BITS_1);
    esp_rom_delay_us(150);
    LCD_Send_Cmd(LCD_CMD_FUNCTION_SET);
    LCD_Send_Cmd(LCD_CMD_CLEAR);
    LCD_Send_Cmd(LCD_CMD_ENTRY_MODE);
    LCD_Send_Cmd(LCD_CMD_DISPLAY_ON);
}

void LCD::LCD_Send_Cmd(uint8_t cmd)
{
    // This function can be used to send a command to the LCD. It sets the RS pin low for command mode and sends the command data.
    gpio_set_level(LCD_RS, 0); // Set RS low for command mode
    // Send the command data to the LCD (this is a placeholder, actual implementation depends on how you are sending data to the LCD)
    // Send 7 6 5 4 bits first (higher nibble)
    gpio_set_level(LCD_D4, (cmd >> 4) & 0x01); // Send higher nibble
    gpio_set_level(LCD_D5, (cmd >> 5) & 0x01);
    gpio_set_level(LCD_D6, (cmd >> 6) & 0x01);
    gpio_set_level(LCD_D7, (cmd >> 7) & 0x01);
    LCD_enable(); // Toggle the enable pin to send the command
    // Send 3 2 1 0 bits next (lower nibble)
    gpio_set_level(LCD_D4, cmd & 0x01); // Send lower nibble
    gpio_set_level(LCD_D5, (cmd >> 1) & 0x01);
    gpio_set_level(LCD_D6, (cmd >> 2) & 0x01);
    gpio_set_level(LCD_D7, (cmd >> 3) & 0x01);
    LCD_enable(); // Toggle the enable pin to send the command

}

void LCD::LCD_move_cursor(uint8_t row, uint8_t col)
{
    uint8_t COMMAND = 0x80;
    if (row > LCD_ROWS || row < 1 || col > LCD_COLUMNS || col < 1)
    {
        LCD_Send_Cmd(COMMAND);
    }
    else if (row == 1)
    {
        COMMAND = 0x80 + (col - 1);
        LCD_Send_Cmd(COMMAND);
    }
    else if (row == 2)
    {
        COMMAND = 0xC0 + (col - 1);
        LCD_Send_Cmd(COMMAND);
    }
}

void LCD::LCD_Clear()
{
    LCD_Send_Cmd(LCD_CMD_CLEAR); // Send the clear screen command to the LCD
    esp_rom_delay_us(2000);        // Short delay to allow the LCD to process the clear command
}

void LCD::LCD_Display_char(const char data)
{
    gpio_set_level(LCD_RS, 1); // Set RS high for data mode
    // Send the character data to the LCD (this is a placeholder, actual implementation depends on how you are sending data to the LCD)
    // Send 7 6 5 4 bits first (higher nibble)
    gpio_set_level(LCD_D4, (data >> 4) & 0x01); // Send higher nibble
    gpio_set_level(LCD_D5, (data >> 5) & 0x01);
    gpio_set_level(LCD_D6, (data >> 6) & 0x01);
    gpio_set_level(LCD_D7, (data >> 7) & 0x01);
    LCD_enable(); // Toggle the enable pin to send the data
    // Send 3 2 1 0 bits next (lower nibble)
    gpio_set_level(LCD_D4, data & 0x01); // Send lower nibble
    gpio_set_level(LCD_D5, (data >> 1) & 0x01);
    gpio_set_level(LCD_D6, (data >> 2) & 0x01);
    gpio_set_level(LCD_D7, (data >> 3) & 0x01);
    LCD_enable(); // Toggle the enable pin to send the data
}

void LCD::LCD_Display_string(const char *data)
{
    while (*data)
    {
        LCD_Display_char(*data++); // Display each character in the string
    }
}

void LCD::LCD_Display_integer(int number)
{
    char buffer[16]; // Buffer to hold the string representation of the integer
    snprintf(buffer, sizeof(buffer), "%d", number); // Convert integer to string
    LCD_Display_string(buffer); // Display the string on the LCD
}

void LCD::LCD_Display_float(float number)
{
    char buffer[16]; // Buffer to hold the string representation of the float
    snprintf(buffer, sizeof(buffer), "%.2f", number); // Convert float to string with 2 decimal places
    LCD_Display_string(buffer); // Display the string on the LCD
}