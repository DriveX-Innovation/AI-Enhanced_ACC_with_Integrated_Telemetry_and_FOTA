#include "LCD.h"

// Constructor and Destructor
LCD::LCD() : i80_bus_handle(nullptr), io_handle(nullptr), address(0), initialized(false)
{
}

LCD::~LCD()
{
    // delete by arrangment:  io_handle, i80_bus_handle:

    // Check if the LCD panel I/O handle is valid before attempting to delete it
    if (io_handle != nullptr)
    {
        esp_lcd_panel_io_del(io_handle);
        io_handle = nullptr;
    }
    // Check if the LCD bus handle is valid before attempting to delete it
    if (i80_bus_handle != nullptr)
    {
        esp_lcd_del_i80_bus(i80_bus_handle);
        i80_bus_handle = nullptr;
    }
}

// initialize the LCD first by setting up the LCD bus, then the LCD panel I/O, and finally the LCD panel itself. After that, it will clear the LCD screen and set the cursor to the home position (0, 0).
void LCD::LCD_Setup()
{

    // Check if the LCD is already initialized to prevent reinitialization
    if (initialized)
    {
        ESP_LOGW("LCD", "LCD already initialized");
        return;
    }
    // LCD bus configuration
    esp_lcd_i80_bus_config_t bus_config = {};

    bus_config.clk_src = LCD_CLK_SRC_DEFAULT;
    bus_config.dc_gpio_num = LCD_RS;
    bus_config.wr_gpio_num = LCD_EN;
    bus_config.data_gpio_nums[0] = LCD_D0;
    bus_config.data_gpio_nums[1] = LCD_D1;
    bus_config.data_gpio_nums[2] = LCD_D2;
    bus_config.data_gpio_nums[3] = LCD_D3;
    bus_config.data_gpio_nums[4] = LCD_D4;
    bus_config.data_gpio_nums[5] = LCD_D5;
    bus_config.data_gpio_nums[6] = LCD_D6;
    bus_config.data_gpio_nums[7] = LCD_D7;
    bus_config.bus_width = LCD_MODE;
    bus_config.max_transfer_bytes = 100; // this value of the buffer that will be used for sending data to the LCD(16*2=32 char).

    ESP_ERROR_CHECK(esp_lcd_new_i80_bus(&bus_config, &i80_bus_handle));

    // LCD panel I/O configuration
    esp_lcd_panel_io_i80_config_t io_config = {};
    io_config.cs_gpio_num = -1;             // No CS pin used, set to -1
    io_config.pclk_hz = LCD_PIXEL_CLOCK_HZ; // Set pixel clock frequency
    io_config.lcd_cmd_bits = 8;             // Number of bits for LCD commands
    io_config.lcd_param_bits = 8;           // Number of bits for LCD parameters
    io_config.dc_levels.dc_idle_level = 0;  // D/C line level in IDLE phase
    io_config.dc_levels.dc_cmd_level = 0;   // D/C line level in CMD phase
    io_config.dc_levels.dc_data_level = 1;  // D/C line level in DATA phase
    io_config.trans_queue_depth = 10;       // Transaction queue size for higher throughput

    ESP_ERROR_CHECK(esp_lcd_new_panel_io_i80(i80_bus_handle, &io_config, &io_handle));

    // LCD panel configuration (at 16*2 normal lcd we must write this configuration manually)
    // 1. Function Set: 8-bit mode, 2 lines, 5x8 font
    esp_lcd_panel_io_tx_param(io_handle, LCD_CMD_FUNCTION_SET, NULL, 0);
    vTaskDelay(pdMS_TO_TICKS(5));

    // 2. Display Control: Display ON, Cursor OFF, Blink OFF
    esp_lcd_panel_io_tx_param(io_handle, LCD_CMD_DISPLAY_ON, NULL, 0);

    // 3. Clear Display
    esp_lcd_panel_io_tx_param(io_handle, LCD_CMD_CLEAR, NULL, 0);
    vTaskDelay(pdMS_TO_TICKS(2));

    // 4. Entry Mode Set: Increment cursor
    esp_lcd_panel_io_tx_param(io_handle, LCD_CMD_ENTRY_MODE, NULL, 0);

    initialized = true; // Set the initialized flag to true after successful setup
}
void LCD::LCD_move_cursor(int row, int col)
{

    if (!initialized)
        return; // Ensure the LCD is initialized before attempting to move the cursor

    // Validate row and column values (assuming a 16x2 LCD)
    if (row < 0 || row >= LCD_ROWS || col < 0 || col >= LCD_COLS)
    {
        ESP_LOGW("LCD", "Invalid cursor position");
        return;
    }

    if (row == 0)
    {
        address = 0x00 + col;
    }
    else if (row == 1)
    {
        address = 0x40 + col;
    }
    // Send the command to set the DDRAM address (cursor position)
    esp_err_t err = esp_lcd_panel_io_tx_param(io_handle, 0x80 | address, NULL, 0);
    if (err != ESP_OK)
    {
        // Handle error appropriately
        ESP_LOGE("LCD", "Failed to move cursor: %s", esp_err_to_name(err));
    }
}

void LCD::LCD_Clear()
{

    if (!initialized)
        return; // Ensure the LCD is initialized before attempting to clear the display

    // Send the command to clear the display
    esp_err_t err = esp_lcd_panel_io_tx_param(io_handle, LCD_CMD_CLEAR, NULL, 0);
    if (err != ESP_OK)
    {
        // Handle error appropriately
        ESP_LOGE("LCD", "Failed to clear display: %s", esp_err_to_name(err));
    }
    vTaskDelay(pdMS_TO_TICKS(5)); // Delay to allow the clear command to process
    LCD_move_cursor(0, 0);        // Move cursor back to home position after clearing
}

void LCD::LCD_Display_char(const char data)
{
    if (!initialized)
        return; // Ensure the LCD is initialized before attempting to display a character

    // Send the character data to the LCD
    esp_err_t err = esp_lcd_panel_io_tx_color(io_handle, -1, &data, 1);
    if (err != ESP_OK)
    {
        // Handle error appropriately
        ESP_LOGE("LCD", "Failed to display character: %s", esp_err_to_name(err));
    }
}

void LCD::LCD_Display_string(const char *data)
{
    if (!initialized)
        return; // Ensure the LCD is initialized before attempting to display a string

    if (!data)
        return;

    // Send the string data to the LCD
    esp_err_t err = esp_lcd_panel_io_tx_color(io_handle, -1, data, strlen(data));
    if (err != ESP_OK)
    {
        ESP_LOGE("LCD", "Failed to display string: %s", esp_err_to_name(err));
    }
}

void LCD::LCD_Display_integer(int number)
{
    if (!initialized)
        return; // Ensure the LCD is initialized before attempting to display an integer

    // Buffer size 12: fits 10 digits + optional '-' sign + null terminator '\0'
    char buffer[12];

    // snprintf convert the number into the string buffer safely
    snprintf(buffer, sizeof(buffer), "%d", number);

    // Send the resulting string to the LCD
    LCD_Display_string(buffer);
}

void LCD::LCD_Display_float(float number)
{
    if (!initialized)
        return;

    char buffer[20];

    // Handle negative numbers
    bool is_negative = false;
    if (number < 0)
    {
        is_negative = true;
        number = -number;
    }

    // Round to 2 decimal places
    number += 0.005f;

    int int_part = (int)number;
    int frac_part = (int)((number - int_part) * 100);

    if (is_negative)
    {
        snprintf(buffer, sizeof(buffer), "-%d.%02d", int_part, frac_part);
    }
    else
    {
        snprintf(buffer, sizeof(buffer), "%d.%02d", int_part, frac_part);
    }

    LCD_Display_string(buffer);
}