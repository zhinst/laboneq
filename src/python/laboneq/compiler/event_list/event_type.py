# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from typing import Any


class EventType:
    SECTION_START = "SECTION_START"
    SECTION_END = "SECTION_END"
    PLAY_START = "PLAY_START"
    PLAY_END = "PLAY_END"
    ACQUIRE_START = "ACQUIRE_START"
    ACQUIRE_END = "ACQUIRE_END"
    LOOP_STEP_START = "LOOP_STEP_START"
    LOOP_STEP_END = "LOOP_STEP_END"
    LOOP_ITERATION_END = "LOOP_ITERATION_END"
    LOOP_END = "LOOP_END"
    PARAMETER_SET = "PARAMETER_SET"
    DELAY_START = "DELAY_START"
    DELAY_END = "DELAY_END"
    RESET_PRECOMPENSATION_FILTERS = "RESET_PRECOMPENSATION_FILTERS"
    INITIAL_RESET_HW_OSCILLATOR_PHASE = "INITIAL_RESET_HW_OSCILLATOR_PHASE"
    RESET_HW_OSCILLATOR_PHASE = "RESET_HW_OSCILLATOR_PHASE"
    SET_OSCILLATOR_FREQUENCY_START = "SET_OSCILLATOR_FREQUENCY_START"
    INITIAL_OSCILLATOR_FREQUENCY = "INITIAL_OSCILLATOR_FREQUENCY"
    SUBSECTION_START = "SUBSECTION_START"
    SUBSECTION_END = "SUBSECTION_END"
    DIGITAL_SIGNAL_STATE_CHANGE = "DIGITAL_SIGNAL_STATE_CHANGE"
    PRNG_SETUP = "PRNG_SETUP"
    DROP_PRNG_SETUP = "DROP_PRNG_SETUP"
    DRAW_PRNG_SAMPLE = "DRAW_PRNG_SAMPLE"
    DROP_PRNG_SAMPLE = "DROP_PRNG_SAMPLE"
    PPC_SWEEP_STEP_START = "PPC_SWEEP_STEP_START"
    PPC_SWEEP_STEP_END = "PPC_SWEEP_STEP_END"


SchedulerEvent = dict[str, Any]
EventList = list[SchedulerEvent]
