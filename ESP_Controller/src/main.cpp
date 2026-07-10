#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "esp_log.h"
#include "RTOS_Tasks.h"


extern void peripherals_init(void);   // defined in RTOS_Tasks.cpp

extern "C" void app_main(void)
{
    ESP_LOGE("MAIN", ">>> app_main reached <<<"); // visible at any log level

    RTOS_init();        // Create queues / event group / mutex
    peripherals_init(); // Motor, encoder, UART, PID

    xTaskCreatePinnedToCore(task_MotorControl, "CTRL",   4096, NULL, 3, NULL, 1); //in Core 1
    xTaskCreatePinnedToCore(task_UART_RX,      "RX",     4096, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(task_UART_TX,      "TX",     2048, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(task_Read_Sensors, "SNS",    2048, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(task_Buzzer,       "BUZZER", 2048, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(task_LCD,          "LCD",    4096, NULL, 1, NULL, 0);

    /* app_main's stack is reclaimed after return — all work is in tasks */
    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    
}