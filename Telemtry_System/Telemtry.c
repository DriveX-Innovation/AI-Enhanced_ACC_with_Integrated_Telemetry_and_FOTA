/*
 * Telemtry.c
 *
 *  Created on: Nov 10, 2025
 *      Author: Zyad Montaser
 */

#include <avr/io.h>
#include <avr/interrupt.h> /* For Timer ISR */
#include <util/delay.h> /*for delay function*/

#include "std_types.h"
#include "common_macros.h"

#include "gpio.h"
#include "adc.h"
#include "lcd.h"
#include "DC_motor.h"
#include "EXTI.h"
#include "lm35_sensor.h"

#define NUMBER_OF_WINDOWS  20

#define MOTOR_ENCODER_DO_PORT_ID  PORTD_ID
#define MOTOR_ENCODER_DO_PIN_ID	  PIN2_ID

uint32 T1=0,T2=0,T=0;
uint32 Timer1_overflow_counter = 0;
boolean MeasurementDone=0;
boolean NewMeasurementAvailable=0;

/* --- Timer1 overflow ISR --- */
ISR(TIMER1_OVF_vect)
{
    /* increment overflow counter (each overflow = 65536 ticks) */
	Timer1_overflow_counter++;
}

/* this will be called from EXTI0 ISR */
void Encoder_Callback(void)
{
    /* Read TCNT1 and current overflow count atomically (in ISR context global ints are disabled) */
    uint16 tcnt1 = TCNT1;                     /* current Timer1 counter */
    uint32 ovf = Timer1_overflow_counter;       /* current overflow count */

    if (MeasurementDone==1)
    {
        /* second edge: create 64-bit end timestamp */
        T2 = ((uint64)ovf << 16) + (uint64)tcnt1;
        /* compute difference (end - start) in microseconds; timer tick = 1 us (F_CPU=1MHz, prescaler=1) */
        if (T2 >= T1)
            T = (uint32)(T2 - T1);
        else
            T = 0; /* shouldn't normally happen */

        MeasurementDone = 0;
        NewMeasurementAvailable = 1;
    }
    else
    {
        /* first edge: store start timestamp */
        T1 = ((uint64)ovf << 16) + (uint64)tcnt1;
        MeasurementDone = 1;
    }
}

/* Initialize Timer1: Normal mode, prescaler = 1, enable overflow interrupt */
void Timer1_Init_prescaler1(void)
{
    TCCR1A = 0x00;                /* Normal mode */
    TCCR1B = (1 << CS10);         /* clkI/O / 1 (no prescaling) => 1 tick = 1 us at F_CPU=1MHz */
    TCNT1 = 0x0000;               /* clear counter */
    TIMSK |= (1 << TOIE1);        /* enable Timer1 overflow interrupt */
    SREG  |= (1<<7);			  /* Enable gloabal interrupt flag I-bit*/
}

void LCD_display_RPM(int rpm)
{
    LCD_moveCursor(0, 0);
    LCD_displayString("RPM: ");
    LCD_moveCursor(0, 5);
    LCD_intgerToString(rpm);
}

int main()
{
    uint64 local_T_us = 0;
    uint32 Motor_RPM = 0;

	uint8 temperature;

	/* Create configuration structure for ADC driver */
	ADC_ConfigType ADC_Configurations = {internal_VREF,F_CPU_8};

	/*passing the configuration structure to the function by address*/
	ADC_init(&ADC_Configurations);

	LCD_init();	/* initialize LCD */
	Timer1_Init_prescaler1(); /* initialize Timer1 */
	EXTI0_Init(); /* initialize EXTI0 */
	EXTI0_SetCallBack(&Encoder_Callback);

	LCD_displayStringRowColumn(1,0,"Temp = ");

	while (1)
	{
		temperature=LM35_getTemperature();

		LCD_moveCursor(1,7);
		LCD_intgerToString(temperature);

		if(NewMeasurementAvailable==1)
		{
			CLEAR_BIT(SREG,7); /*disable interrupts*/
			local_T_us=T;
			NewMeasurementAvailable=0;
			SET_BIT(SREG,7); /*enable interrupts*/

			if(local_T_us==0)
			{
				Motor_RPM=0;
			}
			else
			{
				uint64 denominator = (uint64)local_T_us * (uint64)NUMBER_OF_WINDOWS;
				if (denominator == 0)
					Motor_RPM = 0;
				else
					Motor_RPM = (uint32)(60000000ULL / denominator);
			}
		}

		LCD_display_RPM(Motor_RPM);
		_delay_ms(250);
	}

}
