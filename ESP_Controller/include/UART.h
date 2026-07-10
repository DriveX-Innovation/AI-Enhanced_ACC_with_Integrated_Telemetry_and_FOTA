#pragma once

#include <freertos/FreeRTOS.h>
#include "freertos/queue.h"

#include "driver/uart.h"
#include "hal/uart_types.h"
#include "driver/gpio.h"
#include "esp_log.h"

class UART
{
    private:
        uart_port_t port;
        int tx_pin;
        int rx_pin;
        int baud_rate;
        uart_word_length_t data_bits;
        uart_parity_t parity;
        uart_stop_bits_t stop_bits;
        QueueHandle_t uart_queue;
        const int uart_buffer_size = (1024 * 2);
    public:
        UART(uart_port_t _port, int _tx_pin, int _rx_pin, int _baud_rate);
        ~UART();
        void UART_init();
        void UART_write(const char* data, size_t length);
        int UART_read(uint8_t* buffer, size_t length);
        void UART_get_queue(QueueHandle_t* queue);
};
