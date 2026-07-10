#pragma once

#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"

/* ================================================================
 *  PROTOCOL CONSTANTS
 * ================================================================ */
#define UART_WATCHDOG_MS    200     // e-stop if RPi silent > 200ms
#define MAX_SPEED_RPM       3000
#define MIN_SPEED_RPM       0

/* ================================================================
 *  EVENT GROUP BITS
 * ================================================================ */
#define LANE_ALERT_BIT        BIT0    // set = car outside lane → buzzer ON
#define EMERGENCY_STOP_BIT    BIT1    // set = RPi requests emergency stop / brake

/* ================================================================
 *  PACKET RECEIVED FROM RASPBERRY PI
 * ================================================================ */
typedef struct {
    int16_t  target_speed;
    uint8_t  lane_Departure_flag;
    uint8_t  Emergency_Stop_flag;
} rpi_packet_t;

/* ================================================================
 *  SENSOR DATA
 * ================================================================ */
typedef struct {
    float voltage;
    float current;
    float temperature;
} sensor_data_t;

/* ================================================================
 *  FREERTOS IPC HANDLES
 * ================================================================ */
extern QueueHandle_t        speed_queue;
extern QueueHandle_t        enc_queue;
extern EventGroupHandle_t   system_eventgroup;   // LANE_ALERT_BIT | EMERGENCY_STOP_BIT
extern SemaphoreHandle_t    sensor_mutex;
extern sensor_data_t        shared_sensors;
extern volatile TickType_t  last_uart_rx_tick;