# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import math

from sortedcontainers import SortedDict


def normalize_phase(phase):
    if phase < 0:
        retval = phase + (int(-phase / 2 / math.pi) + 1) * 2 * math.pi
    else:
        retval = phase
    retval = retval % (2 * math.pi)
    return retval


def resample_state(sample_times, state_progression):
    """Resample state at given times, when state changes are known at independently defined times.

    Given a progression of state changes over discrete (integer) time, specified as a SortedDict of (time, state),
    and a collection of events, given as an iterable containing the times at which to sample, determine, for each event,
    the state at the time the event happens.
    Returns a SortedDict of (time, state_tuple) where the times are the times from the events,
    and state_tuple is the corresponding (time, state) tuple taken from the state_progression.

    The algorithm iterates only once through both input collections.
    """
    # Create iterators for the sample times and the state progression
    state_progression_iterator = iter(state_progression.items())
    sample_iterator = iter(sample_times)

    # Create an empty dictionary to store the resampled state
    retval = SortedDict()

    try:
        current_sample_time = next(sample_iterator)
    except StopIteration:
        # Return the empty dictionary if there are no sample times
        return retval

    # Try to get the first state change in the progression
    try:
        current_state = next(state_progression_iterator)
    except StopIteration:
        # Return the empty dictionary if there are no state changes
        return retval

    # Check if the earliest known state change is after the first sample time
    if current_state[0] > current_sample_time:
        raise RuntimeError(
            f"state undefined at time {current_sample_time}, earliest known state at {current_state[0]}"
        )

    # Get the next state change in the progression
    next_state = next(state_progression_iterator, None)

    while True:
        # If there is no next state change or the current sample time is before the next state change
        if next_state is None or current_sample_time < next_state[0]:
            # If the current sample time is after or equal to the current state change time
            if current_sample_time >= current_state[0]:
                # Add the current state change to the resampled state
                retval[current_sample_time] = current_state
                # Try to get the next sample time
                try:
                    previous_sample_time = current_sample_time
                    current_sample_time = next(sample_iterator)
                    # Check if the sample times are in ascending order
                    if current_sample_time < previous_sample_time:
                        raise RuntimeError(
                            f"Sample times must be sorted in ascending order, but current time {current_sample_time} is before previous time {previous_sample_time}"
                        )
                except StopIteration:
                    # Exit the loop if there are no more sample times
                    break
        else:
            current_state = next_state
            try:
                next_state = next(state_progression_iterator)
            except StopIteration:
                next_state = None

    return retval
