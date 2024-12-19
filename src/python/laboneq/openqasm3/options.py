# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
)


@attrs.define(kw_only=True)
class SingleProgramOptions:
    """LabOne Q Experiment options for a single OpenQASM program.

    Attributes:
        count:
            The number of acquire iterations.
        averaging_mode:
            The mode of how to average the acquired data.
        acquisition_type:
            The type of acquisition to perform.

            The acquisition type may also be specified within the
            OpenQASM program using `pragma zi.acqusition_type raw`,
            for example.

            If an acquisition type is passed here, it overrides
            any value set by a pragma.

            If the acquisition type is not specified, it defaults
            to [AcquisitionType.INTEGRATION]().
        reset_oscillator_phase:
            When true, reset all oscillators at the start of every
            acquisition loop iteration.
    """

    count: int = attrs.field(default=1)
    averaging_mode: AveragingMode = attrs.field(
        default=AveragingMode.CYCLIC, converter=AveragingMode
    )
    # NOTE: integration != AcquisitionType.INTEGRATION as the enum value is integration_trigger instead
    # TODO: Fix
    acquisition_type: AcquisitionType = attrs.field(
        default=None, converter=lambda x: AcquisitionType(x) if x is not None else x
    )
    reset_oscillator_phase: bool = attrs.field(default=False)

    @count.validator
    def _check_count(self, attribute, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("count must be an positive integer")

    @reset_oscillator_phase.validator
    def _check_reset_oscillator_phase(self, attribute, value):
        if not isinstance(value, bool):
            raise ValueError("reset_oscillator_phase must be boolean value")


@attrs.define(kw_only=True)
class MultiProgramOptions(SingleProgramOptions):
    """LabOne Q Experiment options for multiple OpenQASM programs.

    Attributes:
        count:
            The number of acquire iterations.
        averaging_mode:
            The mode of how to average the acquired data.
        acquisition_type:
            The type of acquisition to perform.

            The acquisition type may also be specified within the
            OpenQASM program using `pragma zi.acqusition_type raw`,
            for example.

            If an acquisition type is passed here, it overrides
            any value set by a pragma.

            If the acquisition type is not specified, it defaults
            to [AcquisitionType.INTEGRATION]().
        reset_oscillator_phase:
            When true, reset all oscillators at the start of every
            acquisition loop iteration.
        repetition_time:
            The length that any single program is padded to.
        batch_execution_mode:
            The execution mode for the sequence of programs. Can be any of the following:

            - "nt": The individual programs are dispatched by software.
            - "pipeline": The individual programs are dispatched by the sequence pipeliner.
            - "rt": All the programs are combined into a single real-time program.

            "rt" offers the fastest execution, but is limited by device memory.
            In comparison, "pipeline" introduces non-deterministic delays between
            programs of up to a few 100 microseconds. "nt" is the slowest.
        add_reset:
            If `True`, an active reset operation is added to the beginning of each program.
        add_measurement:
            If `True`, add measurement at the end for all qubits used.
        pipeline_chunk_count:
            The number of pipeline chunks to divide the experiment into.
    """

    repetition_time: float = attrs.field(
        default=1e-3, validator=attrs.validators.instance_of(float)
    )
    batch_execution_mode: str = attrs.field(
        default="pipeline", validator=attrs.validators.in_(("pipeline", "nt", "rt"))
    )
    add_reset: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    add_measurement: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )
    pipeline_chunk_count: int | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(int)),
    )
