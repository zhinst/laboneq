# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0
"""Simulation and visualization of signals in LabOne Q experiments"""

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
    max_simulation_duration: float,
    signal_overrides: Sequence[str] | None,
    signal_patterns: Sequence[str] | None,
    integration_kernels_to_plot: Literal["all"] | Sequence[int],
    signal_plots_backend: SignalPlotsBackend[ChannelSimulationResult],
) -> list[Plot]:
    """Prepare plots that show simulated experiment signals.

    This function takes a compiled experiment and simulates some of the channels of the
    instrument using the supplied backend. The same backend is used to transform the
    simulation results into ``Plot`` objects which may later be plotted.

    The signal selection is driven by compatibility with ``plot_simulation`` from
    ``laboneq.contrib.example_helpers.plotting.plot_helpers``, and it works with two
    different mechanisms. The more commonly used one is ``signal_patterns``. Any
    experiment signal of the compiled experiment that contains one of the "patterns"
    is selected. Supplying ``["drive"]`` together with an experiment that has the signals
    ``["q0/drive", "q0/measure", "q1/drive", "q1/measure"]`` would result in the selection
    of ``["q0/drive", "q1/drive"]``. When ``signal_overrides`` is supplied, only these
    signals are considered for selection via ``signal_patterns`` instead of all experiment
    signals. The returned plots are in the order of the selected signals.

    With ``simulation_start_time`` and ``max_simulation_duration`` it is possible to
    control which time slice of the experiment is simulated and plotted. Note that the
    ``max_simulation_duration`` parameter is only a maximum. Depending on the backend, the
    resulting plots may show a shorter time frame.

    ``integration_kernels_to_plot`` is another parameter that exists mostly for
    compatibility with the interface of ``plot_simulation``. It influences what is
    plotted for "acquire-type" experiment signals. It is most commonly used with
    ``"all"``, which means that all integration kernels for all signals are plotted.
    Alternatively a list of integers may be supplied, for example ``[1, 2]`` will plot
    the first and second kernel for all signals. Note that the exact behavior of this
    feature depends on the backend implementation.

    Args:
        compiled_experiment: A compiled experiment.
        simulation_start_time: The start time of the simulation in seconds.
        max_simulation_duration: The maximum duration of the simulation in seconds.
        signal_overrides: Signal names to select from instead of the signals of the
                          supplied experiment.
        signal_patterns: Strings to use for selection of signals to simulate and plot.
                         Signals which have any of the supplied strings as substrings
                        are selected.
        integration_kernels_to_plot: Indicates which integration kernels to plot for
                                     acquire signals. Applies to all signals.
        signal_plots_backend: An implementation of ``SignalPlotsBackend`` that can work
                              with the supplied compiled experiment.

    Returns:
        ``Plot`` objects resulting from the selected signals. One signal may
            result in more than one plot, but multiple signals may also appear in a
            single plot, this depends on the backend implementation.
    """
    selected_experiment_signals = select_experiment_signals(
        compiled_experiment=compiled_experiment,
        signal_overrides=signal_overrides,
        patterns=signal_patterns,
    )

    simulated_outputs = signal_plots_backend.simulate_channel_output(
        compiled_experiment=compiled_experiment,
        experiment_signals=selected_experiment_signals,
        max_duration=max_simulation_duration,
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
    """Select experiment signals that contain any of the strings in ``patterns``.

    If ``signal_overrides`` is present, these will be used instead of the actual
    experiment signals. Signal names that contain any of ``patterns`` as a substring
    are returned in the order they are found.

    Args:
        compiled_experiment: A compiled experiment.
        signal_overrides: Signal names to select from instead of the signals of the
                          supplied experiment.
        patterns: Strings to use for selection of signals to simulate
                  and plot. Signals which have any of the supplied strings as
                  substrings are selected.

    Returns:
        Signal names that match the supplied patterns.
    """
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
    """Returns ``strings`` that contain any of ``patterns`` as a substring."""
    if patterns is None:
        return strings

    matches = []
    for p in patterns:
        for s in strings:
            if p in s and s not in matches:
                matches.append(s)

    return matches


class ChannelType(Enum):
    """Enum that describes channel types relevant for plotting signals.

    This enum defines channel types as they are relevant for creating plots,
    they should not be used in other contexts, where the names may have different
    meanings or different terms are used to describe channels.

    Attributes:
        IQ: Output channel for complex signals.
        RF: Output channel for real signals.
        INPUT: Input channel that produces no output.
    """

    IQ = 0
    RF = 1
    INPUT = 2


@attrs.define(frozen=True)
class Channel:
    """A channel for which data is simulated and/or plotted.

    This class represents a physical device channel in the context of simulating and
    plotting experiment signals. Since output simulators simulate channels, but
    the interfaces for signal plotting ask the user to provide signal names, this
    class acts as a connection between the two concepts.

    The properties are mostly relevant for preparing plots, and should not be used
    in other contexts. ``experiment_signal`` is not used for comparisons, because
    multiple signals may be multiplexed on the same physical channel, but it's still
    the same channel with the same output simulation result.

    Attributes:
        uid: A unique ID identifying the channel.
        name: A name assigned to the channel.
        type: The type of the channel in the context of plotting signals.
        experiment_signal: An experiment signal mapped to this channel.
    """

    uid: str
    name: str
    type: ChannelType
    experiment_signal: str = attrs.field(eq=False)


@attrs.define(frozen=True)
class ChannelSimulationResult:
    """Base class for channel simulation results.

    This class is meant to be used as a base class for channel simulation results
    pertaining to different hardware-specific implementations. The interface strives
    to be minimal, and as such it only comprises numpy arrays for the time axis and
    the corresponding samples of the result and a ``Channel`` object as a form of
    metadata.

    Hardware-specific simulations may produce more data in addition, but these are
    handled in the respective subclasses.

    Attributes:
        samples: Simulated output values.
        time: Time axis corresponding to the samples.
        channel: The channel this result was produced for.
    """

    samples: np.ndarray
    time: np.ndarray
    channel: Channel


ChannelSimulationResultT = TypeVar(
    "ChannelSimulationResultT", bound=ChannelSimulationResult
)


@attrs.define
class Series:
    """A data series with a descriptive label for use in plots."""

    data: np.ndarray
    label: str


@attrs.define
class Plot:
    """Represents plot data.

    In the context of plotting channel simulation results, a plot has data for the
    x-axis, which usually represents time, and a number of data series with y-data.

    In addition to the raw data, ``Plot`` objects also carry a title and a ``Channel``
    object.

    ``Plot`` objects are essentially precursors to creating actual plots using
    matplotlib or other plotting libraries.

    !!! note
        It is very likely that the ``channel`` property is removed at a later point
        because it mostly supports the implementation of some specific plot
        deduplication rules that may be formulated differently.

    Attributes:
        x: x-data of the plot.
        y: A list of ``Series`` objects to represent data series.
        channel: The physical channel this plot was created for.
        title: Plot title.
    """

    x: np.ndarray
    y: list[Series]
    channel: Channel
    title: str


class SignalPlotsBackend(ABC, Generic[ChannelSimulationResultT]):
    """Abstract base class for signal plotting backends.

    In order to visualize signals in various ways, the data for the corresponding
    physical channels has to simulated. After that, the results of these simulation
    may have to be refined into data that can be plotted with libraries such as
    matplotlib.

    The simulation is highly hardware-specific, using different output simulators to
    produce the data for different hardware products. The structure of the results
    obtained from these simulators is also dependent on the output simulator
    implementation. This is the reason why the base class is generic in this respect,
    any backend will produce its specific result type in ``simulate_channel_output``
    and consume it in ``prepare_plots``.
    """

    @abstractmethod
    def simulate_channel_output(
        self,
        *,
        compiled_experiment: CompiledExperiment,
        experiment_signals: Sequence[str],
        start_time: float,
        max_duration: float,
    ) -> list[ChannelSimulationResultT]:
        """Simulate the channel output for specified experiment signals.

        Implementations must determine which physical channels the specified signals
        map to and simulate these starting from ``start_time`` and with a maximum
        duration of ``max_duration``. Implementations are free to simulate a shorter
        time span if the channels do not produce output for the entire requested
        duration, but they must cut off any data that goes beyond this.

        Args:
            compiled_experiment: A compiled experiment to base the simulation on.
            experiment_signals: Experiment signals for which to simulate channel output.
            start_time: Start time of the simulation in seconds, relative to the start
                        of the experiment.
            max_duration: Maximum duration of the simulation in seconds.

        Returns:
            Channel simulation results. Each backend must implement and support its own
                subclass of ``ChannelSimulationResult``.
        """

    @abstractmethod
    def prepare_plots(
        self,
        *,
        simulated_outputs: Sequence[ChannelSimulationResultT],
        compiled_experiment: CompiledExperiment,
        integration_kernels_to_plot: Literal["all"] | Sequence[int],
    ) -> list[Plot]:
        """Prepare plots from channel simulation results.

        Any implementing backend must transform a sequence of ``ChannelSimulationResultT``
        implementations into a list of ``Plot`` objects. There does not need to be a
        1:1 correspondence between results and plots, it's possible to produce multiple
        plots for one result or to combine multiple results into one plot.

        Implementations may also produce supporting plots such as integration kernels
        for acquire signals. The ``compiled_experiment`` is passed as a parameter so that
        additional data can be extracted from it. ``integration_kernels_to_plot`` is an
        argument that mostly exists for backwards compatibility, it is in practice passed
        through from higher levels where it is specified by the user (see
        ``prepare_signal_plots`` for details).

        Args:
            simulated_outputs: Channel simulation results supported by this backend.
            compiled_experiment: A compiled experiment to extract additional data from.
            integration_kernels_to_plot: A selection mechanism for which integration kernels
                                         to include in the plots.

        Returns:
            A list of ``Plot`` objects.
        """


class SignalPlotsBackendRegistry:
    """A registry for ``SignalPlotsBackend`` instances."""

    def __init__(self):
        self._backends: dict[str, SignalPlotsBackend] = {}

    def register(
        self,
        *,
        backend: SignalPlotsBackend,
        name: str,
    ):
        """Register a ``SignalPlotsBackend`` under the specified name."""
        if name in self._backends:
            raise LabOneQException(
                f"'{name}' already has a registered "
                f"backend '{self._backends[name].__class__.__name__}'"
            )

        self._backends[name] = backend

    def from_name(self, name: str) -> SignalPlotsBackend:
        """Return the backend registered under the specified name."""
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
        """Return the appropriate ``SignalPlotsBackend`` for the supplied experiment."""
        return self.from_name(
            peek_backend_name(compiler_output_bytes(compiled_experiment))
        )


backends = SignalPlotsBackendRegistry()
