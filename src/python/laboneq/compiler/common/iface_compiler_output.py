# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.data.nt_step_key import NtStepKey


class NeartimeStepBase:
    key: NtStepKey


class RTCompilerOutput:
    """Base class for a single run of a code generation backend."""


class CombinedOutput:
    """Base class for compiler output _after_ linking individual runs of the code
    generation backend.

    Each concrete subclass owns a `result_handle_maps` field, which is later exposed on the
    corresponding `CompilerArtifact` via `get_artifacts()`. The key type is backend-specific;
    it only needs to be hashable and consistent between the map built here and the key the
    controller looks up at acquisition time.

    result_handle_maps: For each result source, contains a map representing the info about which index
                        in the result corresponds to which handle(s). If experiment is single shot, these maps
                        are supposed to contain info for one shot only - the result builder extrapolates over shots.
                        For a result source, the set of handles at each index can be different, depending on experiment
                        structure. E.g. if an experiment has acquisition on the same signal (with different handles) inside
                        and outside a sweep, the one outside happens more rarely hence its handle also appears in a few entries
                        in the map only.
                        Furthermore, the set of handles for an index is allowed to be empty, which means this result
                        does not correspond to any handle. This can happen because of some pecularities of instruments,
                        such as launching integration units independently not being possible, which means if only one
                        unit needs to produce results all the others are launched with it producind NaN results.
    """


@dataclass
class RTCompilerOutputContainer:
    """Container (by backend) for the output of the code gen backend for a single
    run."""

    backend: str
    codegen_output: RTCompilerOutput
    schedule: compiler_rs.PulseSheetSchedule | None
