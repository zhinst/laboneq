# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from laboneq.core.types.enums.acquisition_type import AcquisitionType
    from laboneq.core.types.enums.averaging_mode import AveragingMode
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.executor.executor import Statement


class CompilerArtifact:
    pass


@dataclass(frozen=True)
class HandleResultShape:
    shape: tuple[int, ...]
    axis_names: list[str | list[str]]
    axis_values: list[NumPyArray | list[NumPyArray]]
    chunked_axis_index: int | None
    # Maps axis to sorted indices of rows along that axis that correspond to this result
    # e.g. if shape = (3, 5, 7), and mask = {2: [6, 8]}, then this result fills the subarray [:, :, [6, 8]]
    match_case_mask: dict[int, list[int]] | None

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        if not isinstance(other, HandleResultShape):
            return False

        return (
            self.shape,
            self.axis_names,
            self.chunked_axis_index,
            self.match_case_mask,
        ) == (
            other.shape,
            other.axis_names,
            other.chunked_axis_index,
            other.match_case_mask,
        )


@dataclass
class RtLoopProperties:
    acquisition_type: AcquisitionType
    averaging_mode: AveragingMode
    shots: int
    chunk_count: int | None


@dataclass(frozen=True)
class ResultShapeInfo:
    shapes: dict[str, HandleResultShape]


@dataclass
class SoftwareVersions:
    # LabOne Q version in SemVer format.
    laboneq: str


@dataclass
class ScheduledExperiment:
    device_setup_fingerprint: str

    #: Compiler artifacts specific to backend(s)
    artifacts: CompilerArtifact

    #: list of events as scheduled by the compiler.
    schedule: dict[str, Any] | None

    #: Experiment execution model
    execution: Statement

    rt_loop_properties: RtLoopProperties

    result_shape_info: ResultShapeInfo

    #: Total duration of the real-time steps, in seconds.
    total_execution_time: float = 0.0

    #: Maximum duration of a single real-time step, in seconds.
    max_step_execution_time: float = 0.0

    #: Software versions used to produce this experiment.
    versions: SoftwareVersions = field(default_factory=lambda: SoftwareVersions(""))

    # Declared coprocessor payload per device-setup inventory key.
    # Opaque to LabOne Q.
    coprocessor_payloads: dict[str, bytes] = field(default_factory=dict)

    # Experiment-side coprocessor label -> device-setup inventory key.
    coprocessor_mappings: dict[str, str] = field(default_factory=dict)
