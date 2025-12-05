# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

from laboneq.core.utilities.attrs_helpers import validated_field
from laboneq.dsl.parameter import LinearSweepParameter, Parameter, SweepParameter

if TYPE_CHECKING:
    from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
    from laboneq.dsl.experiment import ExperimentSignal


def _source_converter(
    value: str | LogicalSignal | ExperimentSignal | None,
) -> str | None:
    # Avoid circular import
    from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
    from laboneq.dsl.experiment import ExperimentSignal

    if isinstance(value, LogicalSignal):
        value = value.path
    elif isinstance(value, ExperimentSignal):
        value = value.mapped_logical_signal_path

    return value


def _amplitude_scaling_validator(
    _self: OutputRoute,
    _attribute: attrs.Attribute,
    value: float | SweepParameter | LinearSweepParameter,
) -> None:
    if not isinstance(value, Parameter):
        if not 0 <= value <= 1:
            raise ValueError("`amplitude_scaling` must be one of or between 0 and 1.")


@attrs.define
class OutputRoute:
    """Output route of the given source channel.

    Route a generated signal from the source channel to the target output signal
    within the same device SG output channels.

    Attributes:
        source: Source signal or its UID which will be added to the target signal channel.
        amplitude_scaling: Amplitude scaling factor of the source signal.
            Value must be one of or between 0 and 1.
            Can be swept in near-time sweeps.
        phase_shift: Phase shift applied to the source signal in radians.
            Value must be a real value.
            Can be swept in near-time sweeps.

    !!! version-changed "Changed in version 26.1.0"

        The types of the attributes are now validated when an `OutputRoute` instance is
        created or when an attribute is set. A `TypeError` is raised if the type of the
        supplied value is incorrect.
    """

    source: str | None = validated_field(default=None, converter=_source_converter)
    amplitude_scaling: float | SweepParameter | LinearSweepParameter = validated_field(
        # The default value (None) is not an allowed type, so amplitude scaling must
        # be specified or a type error will be raised. Setting the default here is required
        # though because `source` is optional.
        default=None,
        validator=_amplitude_scaling_validator,
    )
    phase_shift: float | SweepParameter | LinearSweepParameter | None = validated_field(
        default=None,
    )
