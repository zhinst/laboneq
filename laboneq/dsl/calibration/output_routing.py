# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
    from laboneq.dsl.experiment import ExperimentSignal
    from laboneq.dsl.parameter import SweepParameter, LinearSweepParameter


@dataclass
class OutputRoute:
    """Output route of the given source channel.

    Route a generated signal from the source channel to the target output signal
    within the same device SG output channels.

    Attributes:
        source: Source signal or it's UID which will be added to the target signal channel.
        amplitude_scaling: Amplitude scaling factor of the source signal.
            Value must be one of or between 0 and 1.
            Can be swept in near-time sweeps.
        phase_shift: Phase shift applied to the source signal in radians.
            Value must be a real value.
            Can be swept in near-time sweeps.
    """

    source: str | LogicalSignal | ExperimentSignal
    amplitude_scaling: float | SweepParameter | LinearSweepParameter
    phase_shift: float | SweepParameter | LinearSweepParameter | None = None

    def __post_init__(self):
        # Avoid circular import
        from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
        from laboneq.dsl.experiment import ExperimentSignal
        from laboneq.dsl.parameter import Parameter

        if not isinstance(self.amplitude_scaling, Parameter):
            if not 0 <= self.amplitude_scaling <= 1:
                raise ValueError(
                    "`amplitude_scaling` must be one of or between 0 and 1."
                )

        if isinstance(self.source, LogicalSignal):
            self.source = self.source.path
        elif isinstance(self.source, ExperimentSignal):
            self.source = self.source.mapped_logical_signal_path
