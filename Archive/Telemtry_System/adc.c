 /******************************************************************************
 *
 * Module: ADC
 *
 * File Name: adc.c
 *
 * Description: Source file for the ATmega32 ADC driver
 *
 * Author: Zyad Montaser
 *
 *******************************************************************************/

#include "avr/io.h" /* To use the ADC Registers */
#include "adc.h"
#include "common_macros.h" /* To use the macros like SET_BIT */

/*******************************************************************************
 *                      Functions Definitions                                  *
 *******************************************************************************/

void ADC_init(const ADC_ConfigType * Config_Ptr)
{
	/* ADMUX Register Bits Description:
	 * REFS1:0 = 01 to choose AVCC = 5v as reference voltage
	 * ADLAR   = 0 right adjusted
	 * MUX4:0  = 00000 to choose channel 0 as initialization
	 */
	 /*insert value of ref volt in last 2 bits (REFS1 , REFS0) of ADMUX register */
	ADMUX = ( (ADMUX & 0x3F) | (Config_Ptr->ref_volt <<6) );

	/* ADCSRA Register Bits Description:
	 * ADEN    = 1 Enable ADC
	 * ADIE    = 0 Disable ADC Interrupt
	 * ADATE   = 0 Disable Auto Trigger
	 * ADPS2:0 = 111 to choose ADC_Clock = F_CPU/128 = 16Mhz/128 = 125Khz --> ADC must operate in range 50-200Khz
	 */
	/*insert value of prescaler in first 3 bits (ADPS2,ADPS1,ADPS0) of ADCSRA register*/
	ADCSRA = ( (ADCSRA & 0xF8) | (Config_Ptr ->prescaler) );
	SET_BIT(ADCSRA,ADEN); /*enable ADC*/
}

uint16 ADC_readChannel(uint8 channel_num)
{
	ADMUX= ((ADMUX & 0xE0) | (channel_num & 0x1F));/*insert channel num in 1st 5bits in ADMUX MUX4:0*/
	SET_BIT(ADCSRA,ADSC); /* Start conversion write '1' to ADSC */
	while(BIT_IS_CLEAR(ADCSRA,ADIF)); /* (polling)Wait for conversion to complete, ADIF becomes '1' */
	SET_BIT(ADCSRA,ADIF); /* Clear ADIF by write '1' to it :) */
	return ADC; /* Read the digital value from the data register */
}
