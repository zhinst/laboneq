# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0
"""Visualization for logical signals used in LabOne Q experiments"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Generic, Literal, Sequence, TypeVar

import attrs

from laboneq.core.compiler_output import peek_backend_name
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.compiled_experiment import (
    compiler_output_bytes,
)

if TYPE_CHECKING:
    import numpy as np

    from laboneq.core.types.compiled_experiment import CompiledExperiment


def prepare_signal_plots(
    *,
    compiled_experiment: CompiledExperiment,
    simulation_start_time: float,
    simulation_duration: float,
    signal_overrides: Sequence[str] | None,
    signal_patterns: Sequence[str] | None,
    integration_kernels_to_plot: Literal["all"] | Sequence[int],
    signal_plots_backend: SignalPlotsBackend[ChannelSimulationResult],
) -> list[Plot]:
    selected_experiment_signals = select_experiment_signals(
        compiled_experiment=compiled_experiment,
        signal_overrides=signal_overrides,
        patterns=signal_patterns,
    )

    simulated_outputs = signal_plots_backend.simulate_channel_output(
        compiled_experiment=compiled_experiment,
        experiment_signals=selected_experiment_signals,
        duration=simulation_duration,
        start_time=simulation_start_time,
    )

    return signal_plots_backend.prepare_plots(
        simulated_outputs=simulated_outputs,
        compiled_experiment=compiled_experiment,
        integration_kernels_to_plot=integration_kernels_to_plot,
    )


def select_experiment_signals(
    *,
    compiled_experiment: CompiledExperiment,
    signal_overrides: Sequence[str] | None = None,
    patterns: Sequence[str] | None = None,
) -> list[str]:
    experiment_signals = list(compiled_experiment.experiment.signals.keys())

    matching_signals = match(
        strings=signal_overrides or experiment_signals, patterns=patterns
    )

    if not matching_signals:
        raise ValueError(
            f"None of the signals in group_signal_names_as "
            f"({patterns}) match the signals used in the "
            f"experiment: {experiment_signals}."
        )

    return matching_signals


def match(
    *, strings: Sequence[str], patterns: Sequence[str] | None = None
) -> list[str]:
    if patterns is None:
        return strings

    matches = []
    for p in patterns:
        for s in strings:
            if p in s and s not in matches:
                matches.append(s)

    return matches


class ChannelType(Enum):
    IQ = 0
    RF = 1
    INPUT = 2


@attrs.define(frozen=True)
class Channel:
    uid: str
    name: str
    type: ChannelType
    experiment_signal: str = attrs.field(eq=False)


@attrs.define(frozen=True)
class ChannelSimulationResult:
    samples: np.ndarray
    time: np.ndarray
    channel: Channel


ChannelSimulationResultT = TypeVar(
    "ChannelSimulationResultT", bound=ChannelSimulationResult
)


@attrs.define
class Series:
    data: np.ndarray
    label: str


@attrs.define
class Plot:
    x: np.ndarray
    y: list[Series]
    channel: Channel
    title: str


class SignalPlotsBackend(ABC, Generic[ChannelSimulationResultT]):
    @abstractmethod
    def simulate_channel_output(
        self,
        *,
        compiled_experiment: CompiledExperiment,
        experiment_signals: Sequence[str],
        start_time: float,
        duration: float,
    ) -> list[ChannelSimulationResultT]: ...

    @abstractmethod
    def prepare_plots(
        self,
        *,
        simulated_outputs: Sequence[ChannelSimulationResultT],
        compiled_experiment: CompiledExperiment,
        integration_kernels_to_plot: Literal["all"] | Sequence[int],
    ) -> list[Plot]: ...


class SignalPlotsBackendRegistry:
    def __init__(self):
        self._backends: dict[str, SignalPlotsBackend] = {}

    def register(
        self,
        *,
        backend: SignalPlotsBackend,
        name: str,
    ):
        if name in self._backends:
            raise LabOneQException(
                f"'{name}' already has a registered "
                f"backend '{self._backends[name].__class__.__name__}'"
            )

        self._backends[name] = backend

    def from_name(self, name: str) -> SignalPlotsBackend:
        backend = self._backends.get(name)

        if backend is None:
            raise LabOneQException(
                f"No backend matching the provided artifacts type '{name}'. "
                f"Registered backends: {list(self._backends.keys())}"
            )

        return backend

    def from_compiled_experiment(
        self, compiled_experiment: CompiledExperiment
    ) -> SignalPlotsBackend:
        return self.from_name(
            peek_backend_name(compiler_output_bytes(compiled_experiment))
        )


backends = SignalPlotsBackendRegistry()
