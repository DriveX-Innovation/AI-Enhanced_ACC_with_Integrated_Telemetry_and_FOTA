/*
 * EXTI.h
 *
 *  Created on: Nov 10, 2025
 *      Author: Zyad Montaser
 */

#ifndef EXTI_H_
#define EXTI_H_

// For INT2 only : 0 for falling edge	1 for rising edge (there is no interrupt with any change)
#define EXTI2_Sense_Control	0


typedef enum
{	//this for INT0 and INT1 only
	LOW_LEVEL,ANY_CHANGE,FALLING_EDGE,RISING_EDGE
}EXTI_Sense_Control;

#define EXTI0_SENSE_MODE   RISING_EDGE
#define EXTI1_SENSE_MODE   ANY_CHANGE

/* External INT0 enable and configuration function */
void EXTI0_Init(void);
void EXTI0_SetCallBack(void(*a_ptr)(void));

/* External INT1 enable and configuration function */
void EXTI1_Init(void);
void EXTI1_SetCallBack(void(*a_ptr)(void));

/* External INT2 enable and configuration function */
void EXTI2_Init(void);
void EXTI2_SetCallBack(void(*a_ptr)(void));

#endif /* EXTI_H_ */
