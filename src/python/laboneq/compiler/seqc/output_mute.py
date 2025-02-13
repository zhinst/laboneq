# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from laboneq.core.utilities.pulse_sampler import length_to_samples
from laboneq.core.exceptions import LabOneQException


if TYPE_CHECKING:
    from laboneq.compiler.seqc.seqc_tracker import SeqCTracker
    from laboneq.compiler.common.device_type import DeviceType


def ceil(value: int, grid: int):
    return value + (-value) % grid


class OutputMute:
    """Mute played samples.

    Arguments:
        device_type: Device type
            Given device must support output mute.
        generator: SeqC generator
        duration_min: Minimum duration for mute to be fully engaged.
            Therefore the minimum mute duration is: engage time + `duration_min` + disengage time
    """

    def __init__(
        self,
        device_type: DeviceType,
        generator: SeqCTracker,
        duration_min: float,
    ):
        assert device_type.supports_output_mute, (
            f"Device {device_type.name.upper()} does not support output mute."
        )
        self._device_type = device_type
        self._generator = generator

        # The minimum time for the muting required by the instrument
        device_duration_min = (
            # latency of just turning on and off the blanking...
            self._device_type.output_mute_engage_delay
            - self._device_type.output_mute_disengage_delay
            # ... plus a minimal playZero
            + self._device_type.min_play_wave / self._device_type.sampling_rate
        )
        if duration_min <= device_duration_min:
            msg = f"Output mute duration must be larger than {device_duration_min} s."
            raise LabOneQException(msg)
        samples_min = length_to_samples(
            duration_min,
            self._device_type.sampling_rate,
        )
        self._samples_min = (
            math.ceil(samples_min / self._device_type.sample_multiple)
            * self._device_type.sample_multiple
        )
        delay_engage = length_to_samples(
            self._device_type.output_mute_engage_delay,
            self._device_type.sampling_rate,
        )
        self.delay_engage = ceil(delay_engage, self._device_type.sample_multiple)
        delay_disengage = length_to_samples(
            -self._device_type.output_mute_disengage_delay,
            self._device_type.sampling_rate,
        )
        self.delay_disengage = ceil(delay_disengage, self._device_type.sample_multiple)

    @property
    def samples_min(self) -> int:
        return self._samples_min

    def can_mute(self, samples: int) -> bool:
        return samples >= self.samples_min

    def mute_samples(self, samples: int):
        """Mute samples to be played.

        If the length of samples exceeds `samples_min`, muting is applied,
        otherwise no action is done.
        """
        if self.can_mute(samples):
            self._generator.current_loop_stack_generator().add_play_zero_statement(
                self.delay_engage,
                self._device_type,
                self._generator.deferred_function_calls,
            )
            self._generator.add_function_call_statement(
                "setTrigger", [1], deferred=True
            )
            self._generator.current_loop_stack_generator().add_play_zero_statement(
                samples - self.delay_engage - self.delay_disengage,
                self._device_type,
                self._generator.deferred_function_calls,
            )
            self._generator.add_function_call_statement(
                "setTrigger", [0], deferred=True
            )
            self._generator.current_loop_stack_generator().add_play_zero_statement(
                self.delay_disengage,
                self._device_type,
                self._generator.deferred_function_calls,
            )
