# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Literal, Protocol, Sequence

import attrs
import numpy as np

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
    trigger: np.ndarray
    marker: np.ndarray


class QCCSSignalPlotsBackend(SignalPlotsBackend[QCCSChannelSimulationResult]):
    def simulate_channel_output(
        self,
        *,
        compiled_experiment: CompiledExperiment,
        experiment_signals: Sequence[str],
        start_time: float,
        duration: float,
    ) -> list[QCCSChannelSimulationResult]:
        channels = map_experiment_signals(
            compiled_experiment=compiled_experiment,
            experiment_signals=experiment_signals,
        )

        unique_channels = set(channels)
        results = {}

        simulator = OutputSimulator(
            compiled_experiment, max_simulation_length=max(start_time + duration, 1e-5)
        )

        for channel in unique_channels:
            results[channel] = simulator.get_snippet(
                channel.uid,
                start=start_time,
                output_length=duration,
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
        kernel_samples = _integration_weights_by_signal(compiled_experiment)

        plot_types = [
            IQChannelOutputPlot(),
            NonIQChannelOutputPlot(),
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
    if direction == IODirection.IN:
        return ChannelType.INPUT

    if channel.type == PhysicalChannelType.IQ_CHANNEL:
        return ChannelType.IQ

    return ChannelType.RF


def map_experiment_signals(
    *, compiled_experiment: CompiledExperiment, experiment_signals: Sequence[str]
) -> list[Channel]:
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
    logical_signal = experiment.signals[
        experiment_signal
    ].mapped_logical_signal_path.removeprefix("/logical_signal_groups/")

    return logical_signal


class PlotType(Protocol):
    def matches(self, *, result: QCCSChannelSimulationResult) -> bool: ...
    def prepare_plot(self, *, result: QCCSChannelSimulationResult) -> Plot: ...


def prepare_plots(
    *,
    types: Sequence[PlotType],
    result: QCCSChannelSimulationResult,
) -> Sequence[Plot]:
    return [pt.prepare_plot(result=result) for pt in types if pt.matches(result=result)]


class IQChannelOutputPlot:
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


class NonIQChannelOutputPlot:
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
    def __init__(self, *, kernels, kernels_to_plot, experiment_is_spectroscopy):
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

        if self._kernels_to_plot == "all":
            self._kernels_to_plot = np.arange(len(samples)) + 1

        for k in self._kernels_to_plot:
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
