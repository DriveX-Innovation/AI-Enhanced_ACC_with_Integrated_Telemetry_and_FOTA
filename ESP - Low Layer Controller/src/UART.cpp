#include "UART.h"

UART::UART(uart_port_t _port, int _tx_pin, int _rx_pin, int _baud_rate)
    : port(_port), tx_pin(_tx_pin), rx_pin(_rx_pin), baud_rate(_baud_rate),
    parity(UART_PARITY_DISABLE), stop_bits(UART_STOP_BITS_1)
{
    this->data_bits = UART_DATA_8_BITS;
    this->parity = UART_PARITY_DISABLE;
    this->stop_bits = UART_STOP_BITS_1;
    this->uart_queue = nullptr;
}
UART::~UART() {
    // Clean up resources if necessary
    if (uart_queue) {
        vQueueDelete(uart_queue);
        uart_queue = nullptr;
    }
    uart_driver_delete(port);
}
void UART::UART_init()
{
    // Configure UART parameters
    uart_config_t uart_config = {
        .baud_rate  = this->baud_rate,
        .data_bits  = this->data_bits,
        .parity     = this->parity,
        .stop_bits  = this->stop_bits,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT
    };

    // Install UART driver
    int intr_alloc_flags = 0; // Default interrupt allocation flags
    esp_err_t err = uart_driver_install(port, uart_buffer_size, uart_buffer_size, 10, &uart_queue, intr_alloc_flags);
    if (err != ESP_OK)
    {
        ESP_LOGI("UART", "Failed to install UART driver: %d", err   );
        return;
    }

    // Configure UART parameters
    err = uart_param_config(port, &uart_config);
    if (err != ESP_OK) {
        // Handle error
        ESP_LOGI("UART", "Failed to configure UART parameters: %d", err);
    }

    // Set UART pins
    err = uart_set_pin(port, tx_pin, rx_pin, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    if (err != ESP_OK) {
        // Handle error
    }
}
void UART::UART_write(const char* data, size_t length)
{
    uart_write_bytes(port, data, length);
}
int UART::UART_read(uint8_t* buffer, size_t length)
{
    return uart_read_bytes(port, buffer, length, 20 / portTICK_PERIOD_MS);
}
void UART::UART_get_queue(QueueHandle_t* queue)
{
    *queue = this->uart_queue;
}