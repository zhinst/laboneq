# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from laboneq.compiler.common.awg_info import AwgKey
from laboneq.data.scheduled_experiment import CompilerArtifact


class RealtimeStepBase:
    device_id: str
    awg_id: int
    nt_step: list[int]


class RTCompilerOutput:
    """Base class for a single run of a code generation backend."""


class CombinedOutput(abc.ABC):
    """Base class for compiler output _after_ linking individual runs of the code
    generation backend."""

    feedback_register_configurations: dict[AwgKey, Any]
    realtime_steps: list[RealtimeStepBase]
    total_execution_time: float
    max_execution_time_per_step: float

    @abc.abstractmethod
    def get_artifacts(self) -> CompilerArtifact:
        raise NotImplementedError


@dataclass
class RTCompilerOutputContainer:
    """Container (by device class) for the output of the code gen backend for a single
    run."""

    codegen_output: dict[int, RTCompilerOutput]
    schedule: dict[str, Any]
