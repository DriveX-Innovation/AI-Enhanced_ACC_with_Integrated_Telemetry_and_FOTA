///*
// * Motor_encoder.c
// *
// *  Created on: Nov 10, 2025
// *      Author: Zyad Montaser
// */
//
//#include "Motor_encoder.h"
//#include "std_types.h"
//#include "common_macros.h"
//
//#include "EXTI.h"
//
//uint32 Motor_RPM=0;
//uint32 T1=0,T2=0,T=0;
//uint32 Timer1_overflow_counter = 0;
//boolean MeasurementDone=0;
//boolean NewMeasurementAvailable=0;
//
//void Calculations(void)
//{
//	if(MeasurementDone==1)
//	{
//		T2=getCurrentTime();
//		T=T2-T1;
//		MeasurementDone=0;
//		clearTimer();
//	}
//	else
//	{
//		startTimer();
//		T1=getCurrentTime();
//		MeasurementDone=1;
//	}
//}
//uint32 Calculate_Motor_RPM(void)
//{
//
//	EXTI0_Init();
//	EXTI0_SetCallBack(&Calculations);
//
//	Motor_RPM=60000000/(T*NUMBER_OF_WINDOWS);
//	return Motor_RPM;
//
//}
