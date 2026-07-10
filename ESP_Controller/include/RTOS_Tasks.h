#pragma once

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <math.h>
#include <string.h>        // memcpy()

//Creates all FreeRTOS queues, event groups, and mutexes.
void RTOS_init(void);

/**********Tasks Declarations**************/

//Apply pid logic and read the motor encoder ..will be on core 1..highest priority
void task_MotorControl(void *pvParameters);

/*handle receiving from rpi a packet contain:
1-target speed that calculated from mpc
2-lane departure warning flag (1:car exit its lane so operate buzzer  0:car stay in its lane so turn off buzzer)
3-emergency stop flag)*/
void task_UART_RX(void *pvParameters);

//handle sending the actual speed (calculated from encoder) to rpi 
void task_UART_TX(void *pvParameters);

//read sensors readings
void task_Read_Sensors(void *pvParameters);

//show the sensor readings on the lcd
void task_LCD(void *pvParameters);

//on or off the buzzer based on the lane departure warning flag that received from rpi
void task_Buzzer(void *pvParameters);