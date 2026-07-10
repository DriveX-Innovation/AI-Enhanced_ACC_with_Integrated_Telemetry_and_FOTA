 /******************************************************************************
 *
 * Module: ADC
 *
 * File Name: adc.h
 *
 * Description: header file for the ATmega32 ADC driver
 *
 * Author: Zyad Montaser
 *
 *******************************************************************************/

#ifndef ADC_H_
#define ADC_H_

#include "std_types.h"

/*******************************************************************************
 *                                Definitions                                  *
 *******************************************************************************/
#define ADC_MAXIMUM_VALUE    1023 /* 2 power 10*/
#define ADC_REF_VOLT_VALUE   2.56

/*******************************************************************************
 *                         Types Declaration                                   *
 *******************************************************************************/


/*
 * enumeration defines 3 reference voltage ordered with the same sequence of the ATmega 32 manual
 * REFS1 --> 0 & REFS0 --> 0 : use external reference volt (AREF)
 * REFS1 --> 0 & REFS0 --> 1 : use AVCC as reference volt (5V)
 * REFS1 --> 0 & REFS0 --> 0 : reserved (nothing)
 * REFS1 --> 0 & REFS0 --> 0 : use internal AREF as reference volt (2.56V) as mentioned in manual
 * */
typedef enum
{
	AREF,AVCC,Resrved,internal_VREF
}ADC_ReferenceVolatge;


/*enumeration defines 8 prescalers ordered with the same sequence of the ATmega 32 manual
 * ADPS2 --> 0 & ADPS1 --> 0 & ADPS0 --> 0 : division factor =2
 * ADPS2 --> 0 & ADPS1 --> 0 & ADPS0 --> 1 : division factor =2
 * ADPS2 --> 0 & ADPS1 --> 1 & ADPS0 --> 0 : division factor =4
 * etc....
 *  */
typedef enum
{
	FCPU_2,F_CPU_2,F_CPU_4,F_CPU_8,F_CPU_16,F_CPU_32,F_CPU_64,F_CPU_128
}ADC_Prescaler;


/*configuration structure for ADC driver*/
typedef struct
{
 ADC_ReferenceVolatge ref_volt;
 ADC_Prescaler prescaler;
}ADC_ConfigType;

/*******************************************************************************
 *                      Functions Prototypes                                   *
 *******************************************************************************/

/*
 * Description :
 * Function responsible for initialize the ADC driver.
 */
void ADC_init(const ADC_ConfigType * Config_Ptr);

/*
 * Description :
 * Function responsible for read analog data from a certain ADC channel
 * and convert it to digital using the ADC driver.
 */
uint16 ADC_readChannel(uint8 channel_num);

#endif /* ADC_H_ */
