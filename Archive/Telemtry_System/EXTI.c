/*
 * EXTI.c
 *
 *  Created on: Nov 10, 2025
 *      Author: Zyad Montaser
 */

#include <avr/io.h> /* To use the EXTI Registers */
#include <avr/interrupt.h> /* For EXTI ISRs */
#include "std_types.h"
#include "EXTI.h"


/* Global variables to hold the address of the call back function in the application */
static volatile void (*g_callBackPtr0)(void) = NULL_PTR;
static volatile void (*g_callBackPtr1)(void) = NULL_PTR;
static volatile void (*g_callBackPtr2)(void) = NULL_PTR;

EXTI_Sense_Control EXTI0_Sense_Control=EXTI0_SENSE_MODE;
EXTI_Sense_Control EXTI1_Sense_Control=EXTI1_SENSE_MODE;

void EXTI0_Init(void)
{
	DDRD  &= (~(1<<PD2));               // Configure INT0/PD2 as input pin
	MCUCR = (MCUCR & 0xFC)|EXTI0_Sense_Control;   // Trigger INT0 with the selected sense control
	GICR  |= (1<<INT0);                 // Enable external interrupt pin INT0
	SREG  |= (1<<7);                    // Enable interrupts by setting I-bit
}

void EXTI1_Init(void)
{
	DDRD  &= (~(1<<PD3));               // Configure INT1/PD3 as input pin
	MCUCR = (MCUCR & 0xF3) | (EXTI1_Sense_Control<<2);   // Trigger INT1 with the selected sense control
	GICR  |= (1<<INT1);                 // Enable external interrupt pin INT1
	SREG  |= (1<<7);                    // Enable interrupts by setting I-bit
}

void EXTI2_Init(void)
{
	DDRB   &= (~(1<<PB2));   // Configure INT2/PB2 as input pin
	MCUCSR = (MCUCSR & 0xBF) | (EXTI2_Sense_Control<<6);     // Trigger INT2 with the raising edge
	GICR   |= (1<<INT2);	 // Enable external interrupt pin INT2
	SREG   |= (1<<7);        // Enable interrupts by setting I-bit
}

void EXTI0_SetCallBack(void(*a_ptr)(void))
{
	/* Save the address of the Call back function in a global variable */
	g_callBackPtr0 = a_ptr;
}

void EXTI1_SetCallBack(void(*a_ptr)(void))
{
	/* Save the address of the Call back function in a global variable */
	g_callBackPtr1 = a_ptr;
}

void EXTI2_SetCallBack(void(*a_ptr)(void))
{
	/* Save the address of the Call back function in a global variable */
	g_callBackPtr2 = a_ptr;
}

/* External INT0 Interrupt Service Routine */
ISR(INT0_vect)
{
	if(g_callBackPtr0 != NULL_PTR)
	{
		(*g_callBackPtr0)();/* Call the Call Back function*/
	}
}

/* External INT1 Interrupt Service Routine */
ISR(INT1_vect)
{
	if(g_callBackPtr1 != NULL_PTR)
	{
		(*g_callBackPtr1)();/* Call the Call Back function*/
	}
}

/* External INT2 Interrupt Service Routine */
ISR(INT2_vect)
{
	if(g_callBackPtr2 != NULL_PTR)
	{
		(*g_callBackPtr2)();/* Call the Call Back function*/
	}
}

