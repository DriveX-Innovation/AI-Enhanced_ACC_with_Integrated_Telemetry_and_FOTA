#include "Motor.h"

/*
    Implements motor control using MCPWM peripheral.
    Configures GPIO pins for motor direction and PWM output.
    Provides methods to set speed and direction of the motor.

    MCPWM Configuration for Motor Control
    Clock Frequency: 80MHz
    Desired PWM
    Frequency: 20kHz
    Calculations:
    1. Timer Resolution:
    Period Ticks = Clock Frequency / Desired Frequency
                = 80MHz / 20kHz
                = 4000 ticks
    2. Duty Cycle Ticks:
    For 12-bit resolution, max duty cycle value = 4095
    Duty Cycle Ticks = (Desired Duty Cycle / Max Duty Cycle) * Period Ticks
                    = (speed / 4095) * 4000
    Note:
    - Ensure that the GPIO pins used for IN1, IN2, and PWM are correctly defined when creating a Motor object.
    - 'speed' parameter in SetSpeed method should be between 0 and 4095.
    - Ensure that the MCPWM peripheral is properly initialized before using this class.
    - The MotorSetup method must be called once after creating a Motor object to configure the MCPWM settings.
    - The destructor cleans up MCPWM resources to prevent memory leaks.
*/

Motor::Motor(int _IN1_pin,int _IN2_pin, int _PWM_pin)
    : IN1_pin(_IN1_pin),IN2_pin(_IN2_pin), PWM_pin(_PWM_pin),
    MotorTimer_handle(nullptr), MotorOperator_handle(nullptr),
    MotorGenerator_handle(nullptr), MotorComparator_handle(nullptr)
{
}

Motor::~Motor()
{
    // Delete only if handles were created
    if (MotorTimer_handle)
    {
        ESP_LOGI("Motor", "Disabling and deleting timer");
        mcpwm_timer_disable(MotorTimer_handle);
        mcpwm_del_timer(MotorTimer_handle);
        MotorTimer_handle = nullptr;
    }
    if (MotorOperator_handle)
    {
        ESP_LOGI("Motor", "Deleting operator");
        mcpwm_del_operator(MotorOperator_handle);
        MotorOperator_handle = nullptr;
    }
    if (MotorGenerator_handle)
    {
        ESP_LOGI("Motor", "Deleting generator");
        mcpwm_del_generator(MotorGenerator_handle);
        MotorGenerator_handle = nullptr;
    }
    if (MotorComparator_handle)
    {
        ESP_LOGI("Motor", "Deleting comparator");
        mcpwm_del_comparator(MotorComparator_handle);
        MotorComparator_handle = nullptr;
    }
}

void Motor::MotorSetup()
{
    // Attach the chosen GPIO to MCPWM operator A
    gpio_config_t io_conf = 
    {
        .pin_bit_mask = ((1ULL << IN1_pin)|(1ULL << IN2_pin) ), // Set bit mask for IN1 and IN2 pins   
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf));

    // Initialize IN1 and IN2 to low
    gpio_set_level((gpio_num_t)IN1_pin, 0);
    gpio_set_level((gpio_num_t)IN2_pin, 0);

    // Configure PWM pin for MCPWM output
    gpio_reset_pin((gpio_num_t)PWM_pin);
    gpio_set_direction((gpio_num_t)PWM_pin, GPIO_MODE_OUTPUT);

    // Configure MCPWM timer
    mcpwm_timer_config_t pwm_config =
    {
        .group_id = 0,               
        .clk_src = MCPWM_TIMER_CLK_SRC_DEFAULT, 
        .resolution_hz = RESOLUTION_HZ, 
        .count_mode = MCPWM_TIMER_COUNT_MODE_UP, 
        .period_ticks = PERIOD_TICKS, 
        .intr_priority = 0,
        .flags = 
        {
            .update_period_on_empty = 0,
            .update_period_on_sync = 0,
            .allow_pd = 0,
        }
    };
    // Create MCPWM timer
    ESP_ERROR_CHECK(mcpwm_new_timer(&pwm_config, &MotorTimer_handle)); 

    // Configure MCPWM operator
    mcpwm_operator_config_t oper_config =
    {
        .group_id = 0, 
        .intr_priority = 0,
        .flags =
        {
            .update_gen_action_on_tez = 0,
            .update_gen_action_on_tep = 0,
            .update_gen_action_on_sync = 0,
            .update_dead_time_on_tez = 0,
            .update_dead_time_on_tep = 0,
            .update_dead_time_on_sync = 0,
        }
    };
    // Create MCPWM operator
    ESP_ERROR_CHECK(mcpwm_new_operator(&oper_config, &MotorOperator_handle));

    // Connect operator to timer
    ESP_ERROR_CHECK(mcpwm_operator_connect_timer(MotorOperator_handle, MotorTimer_handle));
    
    // Configure Generator A
    mcpwm_generator_config_t gen_config =
    {
        .gen_gpio_num = PWM_pin, // GPIO for PWM output
        .flags = 
        {
            .invert_pwm = 0,
            .io_loop_back = 0,
            .io_od_mode = 0,
            .pull_up = 0,
            .pull_down = 0,
        }
    };
    // Create MCPWM generator
    ESP_ERROR_CHECK(mcpwm_new_generator(MotorOperator_handle, &gen_config, &MotorGenerator_handle));

    // Config Comparator
    mcpwm_comparator_config_t cmpr_config =
    {
        .intr_priority = 0,
        .flags = 
        {
            .update_cmp_on_tez = 1,
            .update_cmp_on_tep = 0,
            .update_cmp_on_sync = 0,
        } 
    };
    // Create MCPWM comparator
    ESP_ERROR_CHECK(mcpwm_new_comparator(MotorOperator_handle, &cmpr_config, &MotorComparator_handle));

    // Action: when timer hits zero → set HIGH
    mcpwm_gen_timer_event_action_t action_on_empty = {
        .direction = MCPWM_TIMER_DIRECTION_UP,
        .event = MCPWM_TIMER_EVENT_EMPTY,
        .action = MCPWM_GEN_ACTION_HIGH,
    };
    ESP_ERROR_CHECK(mcpwm_generator_set_action_on_timer_event(
        MotorGenerator_handle, 
        action_on_empty
    ));

    // Action: when compare value reached → set LOW
    mcpwm_gen_compare_event_action_t action_on_compare = {
        .direction = MCPWM_TIMER_DIRECTION_UP,
        .comparator = MotorComparator_handle,
        .action = MCPWM_GEN_ACTION_LOW,
    };
    ESP_ERROR_CHECK(mcpwm_generator_set_action_on_compare_event(
        MotorGenerator_handle, 
        action_on_compare
    ));

    // Enable the timer
    ESP_ERROR_CHECK(mcpwm_timer_enable(MotorTimer_handle));
    // Start the timer
    ESP_ERROR_CHECK(mcpwm_timer_start_stop(MotorTimer_handle, MCPWM_TIMER_START_NO_STOP)); 
}

void Motor::SetSpeed(int speed)
{
    // Set the PWM duty cycle to control motor speed
    // Speed should be between 0 and 4095 for 12-bit resolution
    // Speed is coming from 0 to MAX_DUTY_CYCLE as input from ADC ( Throttle )
    // Clamp speed value
    if (speed < 0)
        speed = 0;
    if (speed > MAX_DUTY_CYCLE)
        speed = MAX_DUTY_CYCLE;
    if (MotorComparator_handle == nullptr)
    {
        ESP_LOGW("Motor", "SetSpeed called but MotorComparator_handle is NULL - ignoring");
        return;
    }
    uint32_t duty_ticks = (uint32_t)((((float)speed / (float)MAX_DUTY_CYCLE)) * PERIOD_TICKS);
    // Apply duty
    mcpwm_comparator_set_compare_value(MotorComparator_handle, duty_ticks);
}

void Motor::MoveForward()
{
    gpio_set_level((gpio_num_t)IN1_pin,0);
    gpio_set_level((gpio_num_t)IN2_pin,1);
}

void Motor::MoveBackward()
{
    gpio_set_level((gpio_num_t)IN1_pin,1);
    gpio_set_level((gpio_num_t)IN2_pin,0);
}

void Motor::Brake()
{
    gpio_set_level((gpio_num_t)IN1_pin,1);
    gpio_set_level((gpio_num_t)IN2_pin,1);
}
