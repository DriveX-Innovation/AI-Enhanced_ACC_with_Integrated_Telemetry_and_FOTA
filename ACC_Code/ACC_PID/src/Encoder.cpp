#include "Encoder.h"

Encoder::Encoder(int encoderpinA, int encoderpinB) 
: EncoderPinA(encoderpinA), EncoderPinB(encoderpinB), EncoderPCNT_handle_Pin(nullptr), 
EncoderPCNT_channel_PinA(nullptr), EncoderPCNT_channel_PinB(nullptr)
{
}

Encoder::~Encoder()
{
    // Safe cleanup with null checks
    if (EncoderPCNT_handle_Pin)
    {
        pcnt_unit_stop(EncoderPCNT_handle_Pin);
        pcnt_unit_disable(EncoderPCNT_handle_Pin);
    }

    if (EncoderPCNT_channel_PinA)
    {
        pcnt_del_channel(EncoderPCNT_channel_PinA);
    }

    if (EncoderPCNT_channel_PinB)
    {
        pcnt_del_channel(EncoderPCNT_channel_PinB);
    }

    if (EncoderPCNT_handle_Pin)
    {
        pcnt_del_unit(EncoderPCNT_handle_Pin);
    }
}

void Encoder::EncoderSetup()
{
    // Configure Pin as input with pull-down
    gpio_config_t io_conf = 
    {
        .pin_bit_mask = (1ULL << EncoderPinA) | (1ULL << EncoderPinB), // Set bit mask for both pins
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf)); 

    // Configure PCNT unit
    pcnt_unit_config_t pcnt_config =
    {
        .low_limit = -32768,
        .high_limit = 32767,
        .intr_priority = 0,
        .flags =
            {
                .accum_count = false,
            },
    };
    ESP_ERROR_CHECK(pcnt_new_unit(&pcnt_config, &EncoderPCNT_handle_Pin));

    // Configure PCNT channel for EncoderPinA
    pcnt_chan_config_t EncoderPCNT_channelA_config =
    {
    .edge_gpio_num = EncoderPinA,
    .level_gpio_num = EncoderPinB,
    .flags =
        {
            .invert_edge_input = 0,
            .invert_level_input = 0,
            .virt_edge_io_level = 0,
            .virt_level_io_level = 0,
            .io_loop_back = 0,
        },
    };
    ESP_ERROR_CHECK(pcnt_new_channel(EncoderPCNT_handle_Pin, &EncoderPCNT_channelA_config, &EncoderPCNT_channel_PinA));

    // Set actions for EncoderPinA
    // On rising edge of A, increase count if B is high, decrease if B is low
    ESP_ERROR_CHECK(
    pcnt_channel_set_edge_action(
        EncoderPCNT_channel_PinA,
        PCNT_CHANNEL_EDGE_ACTION_INCREASE,
        PCNT_CHANNEL_EDGE_ACTION_DECREASE
    ));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(
        EncoderPCNT_channel_PinA,
        PCNT_CHANNEL_LEVEL_ACTION_INVERSE,
        PCNT_CHANNEL_LEVEL_ACTION_KEEP
    ));

    // -------- Channel B --------
    pcnt_chan_config_t chanB_config = {
        .edge_gpio_num = EncoderPinB,
        .level_gpio_num = EncoderPinA,
        .flags =
        {
            .invert_edge_input = 0,
            .invert_level_input = 0,
            .virt_edge_io_level = 0,
            .virt_level_io_level = 0,
            .io_loop_back = 0,
        },
    };

    ESP_ERROR_CHECK(pcnt_new_channel(
        EncoderPCNT_handle_Pin,
        &chanB_config,
        &EncoderPCNT_channel_PinB
    ));

    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(
        EncoderPCNT_channel_PinB,
        PCNT_CHANNEL_EDGE_ACTION_INCREASE,
        PCNT_CHANNEL_EDGE_ACTION_DECREASE
    ));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(
        EncoderPCNT_channel_PinB,
        PCNT_CHANNEL_LEVEL_ACTION_KEEP,
        PCNT_CHANNEL_LEVEL_ACTION_INVERSE
    ));

    pcnt_glitch_filter_config_t filter = {
        .max_glitch_ns = 10000,
    };
    pcnt_unit_set_glitch_filter(EncoderPCNT_handle_Pin, &filter);

    pcnt_unit_enable(EncoderPCNT_handle_Pin);
    pcnt_unit_clear_count(EncoderPCNT_handle_Pin);
    pcnt_unit_start(EncoderPCNT_handle_Pin);
}

int Encoder::GetEncoderCount()
{
    int count = 0;

    ESP_ERROR_CHECK(pcnt_unit_get_count(EncoderPCNT_handle_Pin, &count));
    return count;
}

int Encoder::CalculateSpeed()
{
    static int lastCount = 0;
    static int64_t lastTime_us = 0;

    int64_t now_us = esp_timer_get_time();
    int64_t dt_us = now_us - lastTime_us;

    if (dt_us <= 0)
        return 0;
    lastTime_us = now_us;

    int current = GetEncoderCount();
    int delta = current - lastCount;
    lastCount = current;

    // Motor shaft RPM
    float rpm_motor =
        (delta * 60.0f * 1e6f) /
        (dt_us * PPR * 4.0f);

    // Gearbox output RPM (25GA370 = 34:1)
    float rpm_output = rpm_motor / 34.0f;

    return (int)rpm_output;
}

void Encoder::ResetEncoderCount()
{
    ESP_ERROR_CHECK(pcnt_unit_clear_count(EncoderPCNT_handle_Pin));
}

int Encoder::GetEncoderSpeedDigitalOutput()
{
    int speed = CalculateSpeed(); // Calculate speed over 100 ms interval
    // Map speed to digital output range
    if (speed < MIN_SPEED) speed = MIN_SPEED;
    if (speed > MAX_SPEED) speed = MAX_SPEED;

    // Mapping speed to digital output range
    return ( (speed - MIN_SPEED) * (MAX_DIGITAL_OUTPUT - MIN_DIGITAL_OUTPUT) / (MAX_SPEED - MIN_SPEED) ) + MIN_DIGITAL_OUTPUT;
}

int Encoder::GetEncoderDirection()
{
    static int lastCount = 0;

    int current = GetEncoderCount();
    int delta = current - lastCount;
    lastCount = current;

    if (delta > 0)
        return ENCODER_FORWARD;
    if (delta < 0)
        return ENCODER_REVERSE;
    return ENCODER_STOPPED;
}
