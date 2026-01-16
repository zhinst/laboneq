# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from laboneq.data.recipe import NtStepKey
from laboneq.data.scheduled_experiment import CompilerArtifact, ResultSource


class NeartimeStepBase:
    key: NtStepKey


class RTCompilerOutput:
    """Base class for a single run of a code generation backend."""


class CombinedOutput(abc.ABC):
    """Base class for compiler output _after_ linking individual runs of the code
    generation backend.

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

    neartime_steps: list[NeartimeStepBase]
    result_handle_maps: dict[ResultSource, list[set[str]]]

    @abc.abstractmethod
    def get_artifacts(self) -> CompilerArtifact:
        raise NotImplementedError

    @abc.abstractmethod
    def get_raw_acquire_length(self, signal_id: str, handle: str) -> int: ...


@dataclass
class RTCompilerOutputContainer:
    """Container (by device class) for the output of the code gen backend for a single
    run."""

    codegen_output: dict[int, RTCompilerOutput]
    schedule: dict[str, Any] | None
