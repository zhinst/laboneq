# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Literal, Mapping, Protocol, Sequence

import attrs
import numpy as np

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import IODirection, PhysicalChannelType
from laboneq.core.types.enums.acquisition_type import is_spectroscopy
from laboneq.simulator.output_simulator import OutputSimulator
from laboneq.simulator.signal_plots import (
    Channel,
    ChannelSimulationResult,
    ChannelType,
    Plot,
    Series,
    SignalPlotsBackend,
)

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.data.artifacts_qccs import WeightInfo
    from laboneq.dsl.device.io_units import PhysicalChannel
    from laboneq.dsl.experiment import Experiment


@attrs.define(frozen=True)
class QCCSChannelSimulationResult(ChannelSimulationResult):
    """Channel simulation results for QCCS experiments.

    The QCCS output simulator produces data for triggers and markers in addition
    to the standard time/samples, so these are captured here for plot preparation.

    Attributes:
        trigger: The trigger signals on this channel.
        marker: The marker signals on this channel.
    """

    trigger: np.ndarray
    marker: np.ndarray


class QCCSSignalPlotsBackend(SignalPlotsBackend[QCCSChannelSimulationResult]):
    """Signal plotting backend for QCCS experiments.

    This implementation of ``SignalPlotsBackend`` is made for use with experiments that
    have been compiled for QCCS instruments.
    """

    def simulate_channel_output(
        self,
        *,
        compiled_experiment: CompiledExperiment,
        experiment_signals: Sequence[str],
        start_time: float,
        max_duration: float,
    ) -> list[QCCSChannelSimulationResult]:
        """Simulate the channel output for specified signals of a QCCS experiment.

        This method produces one ``QCCSChannelSimulationResult`` for each supplied
        experiment signal, even if the results are identical due to signal multiplexing.
        Internally, each channel is only simulated once.

        Args:
            compiled_experiment: An experiment compiled for a QCCS instrument.
            experiment_signals: Experiment signals for which to simulate channel output.
            start_time: Start time of the simulation in seconds, relative to the start
                        of the experiment.
            max_duration: Maximum duration of the simulation in seconds.

        Returns:
            QCCS channel simulation results, one for each simulated signal.
        """
        channels = map_experiment_signals(
            compiled_experiment=compiled_experiment,
            experiment_signals=experiment_signals,
        )

        unique_channels = set(channels)
        results = {}

        simulator = OutputSimulator(
            compiled_experiment,
            max_simulation_length=max(start_time + max_duration, 1e-5),
        )

        for channel in unique_channels:
            results[channel] = simulator.get_snippet(
                channel.uid,
                start=start_time,
                output_length=max_duration,
                get_trigger=True,
                get_marker=True,
            )

        return [
            QCCSChannelSimulationResult(
                channel=c,
                samples=results[c].wave,
                time=results[c].time,
                trigger=results[c].trigger,
                marker=results[c].marker,
            )
            for c in channels
        ]

    def prepare_plots(
        self,
        *,
        simulated_outputs: Sequence[QCCSChannelSimulationResult],
        compiled_experiment: CompiledExperiment,
        integration_kernels_to_plot: Literal["all"] | Sequence[int],
    ) -> list[Plot]:
        """Prepare plots from ``QCCSChannelSimulationResult`` objects.

        This method takes simulation results and prepares plot data from them.
        Different channel types result in slightly different plots. Currently,
        the following plot types are supported:

            * ``IQChannelOutputPlot``
            * ``RFChannelOutputPlot``
            * ``InputPlot``
            * ``TriggerPlot``
            * ``MarkerPlot``

        Refer to their documentation for details. For adding new plot types, create
        a class that implements the ``PlotType`` protocol and add an instance of it
        in this function.

        Once all plots have been prepared they are deduplicated (see ``deduplicate``).

        Args:
            simulated_outputs: QCCS channel simulation results, for example obtained
                               from the ``simulate_channel_output`` method.
            compiled_experiment: An experiment compiled for a QCCS instrument.
            integration_kernels_to_plot: A selection mechanism for which integration
                                         kernels to include in the plots
                                         (see ``prepare_signal_plots``).
        """
        kernel_samples = _integration_weights_by_signal(compiled_experiment)

        plot_types = [
            IQChannelOutputPlot(),
            RFChannelOutputPlot(),
            InputPlot(
                kernels=kernel_samples,
                kernels_to_plot=integration_kernels_to_plot,
                experiment_is_spectroscopy=is_spectroscopy(
                    compiled_experiment.scheduled_experiment.rt_loop_properties.acquisition_type
                ),
            ),
            TriggerPlot(),
            MarkerPlot(),
        ]

        return deduplicate(
            plots=list(
                chain.from_iterable(
                    (
                        prepare_plots(result=r, types=plot_types)
                        for r in simulated_outputs
                    )
                )
            )
        )


def map_channel_type(channel: PhysicalChannel, direction: IODirection) -> ChannelType:
    """Determine ``ChannelType`` from QCCS-specific data structures."""
    if direction == IODirection.IN:
        return ChannelType.INPUT

    if channel.type == PhysicalChannelType.IQ_CHANNEL:
        return ChannelType.IQ

    return ChannelType.RF


def map_experiment_signals(
    *, compiled_experiment: CompiledExperiment, experiment_signals: Sequence[str]
) -> list[Channel]:
    """Produce ``Channel`` objects for the provided experiment signals."""
    channels = []
    for signal in experiment_signals:
        logical_signal_uid = to_logical_signal(
            experiment=compiled_experiment.experiment, experiment_signal=signal
        )

        logical_signal = compiled_experiment.device_setup.logical_signal_by_uid(
            logical_signal_uid
        )

        physical_channel = logical_signal.physical_channel

        channels.append(
            Channel(
                uid=physical_channel.uid,
                name=physical_channel.name,
                type=map_channel_type(
                    channel=physical_channel, direction=logical_signal.direction
                ),
                experiment_signal=signal,
            )
        )

    return channels


def to_logical_signal(*, experiment: Experiment, experiment_signal: str) -> str:
    """Map an experiment signal to a logical signal."""
    logical_signal = experiment.signals[
        experiment_signal
    ].mapped_logical_signal_path.removeprefix("/logical_signal_groups/")

    return logical_signal


class PlotType(Protocol):
    """Protocol for plots supported by ``QCCSSignalPlotsBackend``."""

    def matches(self, *, result: QCCSChannelSimulationResult) -> bool: ...
    def prepare_plot(self, *, result: QCCSChannelSimulationResult) -> Plot: ...


def prepare_plots(
    *,
    types: Sequence[PlotType],
    result: QCCSChannelSimulationResult,
) -> Sequence[Plot]:
    """Prepare plots from ``QCCSChannelSimulationResult`` objects."""
    return [pt.prepare_plot(result=result) for pt in types if pt.matches(result=result)]


class IQChannelOutputPlot:
    """Plot for IQ channels with two data series (real and imaginary)."""

    @staticmethod
    def matches(*, result: QCCSChannelSimulationResult) -> bool:
        channel = result.channel

        return channel.type == ChannelType.IQ

    @staticmethod
    def prepare_plot(*, result: QCCSChannelSimulationResult) -> Plot:
        channel = result.channel

        return Plot(
            x=result.time,
            y=[
                Series(data=result.samples.real, label="I"),
                Series(data=result.samples.imag, label="Q"),
            ],
            channel=channel,
            title=f"{format_channel(channel)} - {format_signals(channel)}".upper(),
        )


class RFChannelOutputPlot:
    """Plot for RF channels with one data series (real)."""

    @staticmethod
    def matches(*, result: QCCSChannelSimulationResult) -> bool:
        channel = result.channel

        return channel.type == ChannelType.RF

    @staticmethod
    def prepare_plot(*, result: QCCSChannelSimulationResult) -> Plot:
        channel = result.channel

        return Plot(
            x=result.time,
            y=[
                Series(data=result.samples.real, label=format_channel(channel)),
            ],
            channel=channel,
            title=f"{format_channel(channel)} - {format_signals(channel)}".upper(),
        )


class TriggerPlot:
    """Plot for triggers on non-QA channels with one data series."""

    @staticmethod
    def matches(*, result: QCCSChannelSimulationResult) -> bool:
        trigger_data = result.trigger

        return (
            trigger_data is not None
            and "qa" not in result.channel.name
            and np.any(trigger_data > 0)
        )

    @staticmethod
    def prepare_plot(*, result: QCCSChannelSimulationResult) -> Plot:
        channel = result.channel

        return Plot(
            x=result.time,
            y=[
                Series(data=result.trigger, label="Trigger"),
            ],
            channel=channel,
            title=f"{format_channel(channel)} - Trigger".upper(),
        )


class MarkerPlot:
    """Plot for markers with one or two data series, depending on the result data."""

    @staticmethod
    def matches(*, result: QCCSChannelSimulationResult) -> bool:
        marker_data = result.marker

        return marker_data is not None and np.any(marker_data > 0)

    @staticmethod
    def prepare_plot(*, result: QCCSChannelSimulationResult) -> Plot:
        channel = result.channel
        marker_data = result.marker

        series = [
            Series(data=marker_data.real, label="Marker 1"),
        ]

        if np.any(marker_data.imag):
            series.append(Series(data=marker_data.imag, label="Marker 2"))

        return Plot(
            x=result.time,
            y=series,
            channel=channel,
            title=f"{format_channel(channel)} - {format_signals(channel)} - Marker".upper(),
        )


class InputPlot:
    """Plot for acquire channels.

    This plot is different from the others in that it combines the data from the output
    simulator with data from additional sources, namely the integration kernels
    for acquire signals associated with the underlying experiment.

    This type of plot is only generated for input channels and only if the underlying
    experiment is not a spectroscopy experiment.

    Args:
        kernels: A mapping from experiment signals to their associated
                 integration kernels.
        kernels_to_plot: A selection mechanism for which kernels to plot
                         (see ``prepare_signal_plots``).
        experiment_is_spectroscopy: A boolean indicating whether the underlying
                                    experiment is a spectroscopy experiment.
    """

    def __init__(
        self,
        *,
        kernels: Mapping[str, Sequence[np.ndarray]],
        kernels_to_plot: Literal["all"] | Sequence[int],
        experiment_is_spectroscopy: bool,
    ):
        self._kernels = kernels
        self._kernels_to_plot = kernels_to_plot
        self._experiment_is_spectroscopy = experiment_is_spectroscopy

    def matches(self, *, result: QCCSChannelSimulationResult) -> bool:
        if self._experiment_is_spectroscopy:
            return False

        return (
            result.channel.type == ChannelType.INPUT
            and result.channel.experiment_signal in self._kernels
        )

    def prepare_plot(self, *, result: QCCSChannelSimulationResult) -> Plot:
        channel = result.channel

        samples = self._kernels[channel.experiment_signal]
        trigger_indices = np.argwhere(result.samples).flatten()

        series = []

        kernels_to_plot = self._kernels_to_plot
        if kernels_to_plot == "all":
            kernels_to_plot = np.arange(len(samples)) + 1

        if invalid_kernels := {k for k in kernels_to_plot if k > len(samples)}:
            valid_kernels = range(1, len(samples) + 1)

            raise LabOneQException(
                "Invalid integration kernels requested for plotting channel "
                f"{channel.uid}: {', '.join(str(x) for x in invalid_kernels)}. "
                f"Valid choices: {', '.join(str(x) for x in valid_kernels)}. "
                'Use "all" to plot all kernels for all selected signals.'
            )

        for k in kernels_to_plot:
            kernel = samples[k - 1]

            waveform = np.zeros_like(result.time, dtype=np.complex128)

            for i in trigger_indices:
                if i + len(kernel) > len(waveform):
                    kernel = kernel[: (len(waveform) - i)]

                waveform[i : i + len(kernel)] = kernel

            series += [
                Series(
                    data=waveform.real, label=f"w{k}-I" if len(samples) > 1 else "I"
                ),
                Series(
                    data=waveform.imag, label=f"w{k}-Q" if len(samples) > 1 else "Q"
                ),
            ]

        return Plot(
            x=result.time,
            y=series,
            channel=result.channel,
            title=f"{format_channel(channel)} - {format_signals(channel)}".upper(),
        )


def format_channel(channel: Channel) -> str:
    return channel.uid.replace("_", " ").replace("/", ": ")


def format_signals(channel: Channel) -> str:
    return channel.experiment_signal


def _integration_weights_by_signal(
    compiled_experiment: CompiledExperiment,
) -> dict[str, list]:

    scheduled_exp = compiled_experiment.scheduled_experiment

    awg_to_channel_signals = {}
    for integrator in scheduled_exp.artifacts.integrator_allocations:
        key = (integrator.device_id, integrator.awg)
        integration_unit_to_signals: dict[int, set[str]] = {}
        for int_ch in integrator.channels:
            integration_unit_to_signals.setdefault(int_ch, set()).add(
                integrator.signal_id
            )
        awg_to_channel_signals.setdefault(key, dict()).update(
            integration_unit_to_signals
        )

    rt_step_by_awg = set()
    weight_info_by_signal: dict[str, list[WeightInfo]] = {}
    for rt_init in scheduled_exp.artifacts.realtime_execution_init:
        key = (rt_init.device_id, rt_init.awg_index)
        integration_unit_to_signals = awg_to_channel_signals.get(key, {})
        if key not in rt_step_by_awg:
            rt_step_by_awg.add(key)
            for iw in scheduled_exp.artifacts.integration_weights[
                rt_init.kernel_indices_ref
            ]:
                for unit in iw.integration_units:
                    for signal in integration_unit_to_signals.get(unit, set()):
                        weight_info_by_signal.setdefault(signal, list()).append(iw)

    kernel_samples_by_signal: dict[str, list] = {
        signal: [] for signal in weight_info_by_signal
    }
    for signal, kernels in weight_info_by_signal.items():
        for kernel in kernels:
            waveform: None | np.ndarray = None
            for scale, suffix in [(1, ".wave"), (1, "_i.wave"), (1j, "_q.wave")]:
                new_wf = scheduled_exp.artifacts.waves.get(kernel.id + suffix)
                if new_wf is not None:
                    waveform = scale * new_wf.samples + (
                        waveform if waveform is not None else 0
                    )
            assert waveform is not None, "kernel not found"
            waveform = np.repeat(waveform, kernel.downsampling_factor)
            kernel_samples_by_signal[signal] += [waveform]

    return kernel_samples_by_signal


def deduplicate(*, plots: Sequence[Plot]) -> list[Plot]:
    """Deduplicate plots.

    Since the simulation of channels leads to duplicated plots in some cases
    (for example multiplexed signals), some of these duplications are resolved.

    !!! note
        This implementation is likely to change in the future, deduplicating more
        kinds of plots.
    """
    deduplicated_plots = []

    for plot in plots:
        if plot.title.endswith("TRIGGER"):
            if any(p for p in deduplicated_plots if p.title == plot.title):
                continue
            else:
                deduplicated_plots.append(plot)
                continue

        if plot.title.endswith("MARKER"):
            deduplicated_plots.append(plot)
            continue

        channel = plot.channel

        if "sg" in channel.name:
            if (
                existing_plot := next(
                    (p for p in deduplicated_plots if p.channel == channel), None
                )
            ) is not None:
                existing_plot.title += f"-{channel.experiment_signal}".upper()
                continue

        deduplicated_plots.append(plot)

    return deduplicated_plots
