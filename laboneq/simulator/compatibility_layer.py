# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.simulator.seqc_parser import (
    SeqCDescriptor,
    SeqCSimulation,
    SimpleRuntime,
    _analyze_compiled,
)
from laboneq.simulator.seqc_parser import run_single_source as run_single_source_impl
from laboneq.simulator.seqc_parser import simulate
from laboneq.simulator.wave_scroller import SimTarget, WaveScroller


@dataclass
class ChannelInfo:
    name: str
    ch: int
    target: SimTarget


def _get_channel_mapping(device_type: str, channels: List[int]) -> List[ChannelInfo]:
    # output key, channel, wave getter, True if output port delay applicable
    channel_mapping: List[ChannelInfo] = []
    if device_type != "SHFQA":
        for ch in channels:
            channel_mapping.append(ChannelInfo(f"{ch+1}", ch, SimTarget.PLAY))

    if device_type == "UHFQA":
        channel_mapping.append(ChannelInfo("QAResult", -1, SimTarget.ACQUIRE))

    if device_type == "SHFQA":
        channel_mapping = [
            ChannelInfo("0", 0, SimTarget.PLAY),
            ChannelInfo("1", 1, SimTarget.PLAY),
            ChannelInfo("QAResult", -1, SimTarget.ACQUIRE),
            ChannelInfo("QAResult_0", 0, SimTarget.ACQUIRE),
            ChannelInfo("QAResult_1", 1, SimTarget.ACQUIRE),
            ChannelInfo("osc0_freq", 0, SimTarget.FREQUENCY),
        ]

    if device_type == "SHFSG":
        channel_mapping = [
            ChannelInfo(f"{channels[0]+1}_I", 0, SimTarget.PLAY),
            ChannelInfo(f"{channels[0]+1}_Q", 1, SimTarget.PLAY),
            ChannelInfo("osc0_freq", 0, SimTarget.FREQUENCY),
        ]

    channel_mapping.append(ChannelInfo("trigger", 0, SimTarget.TRIGGER))

    return channel_mapping


def _build_compatibility_output(
    seqc_descriptor: SeqCDescriptor, sim: SeqCSimulation
) -> SimpleNamespace:
    channel_mapping = _get_channel_mapping(
        seqc_descriptor.device_type, seqc_descriptor.channels
    )
    output = {}
    times = {}
    times_at_port = {}
    for ch_info in channel_mapping:
        ws = WaveScroller(
            ch=[ch_info.ch],
            sim_targets=ch_info.target,
            sim=sim,
        )
        ws.calc_snippet(-0.5, 1)
        output[ch_info.name] = {
            SimTarget.PLAY: ws.wave_snippet,
            SimTarget.ACQUIRE: ws.acquire_snippet,
            SimTarget.TRIGGER: ws.trigger_snippet,
            SimTarget.FREQUENCY: ws.frequency_snippet,
        }[ch_info.target]

        times[ch_info.name] = ws.time_axis - (
            seqc_descriptor.output_port_delay if ws.is_output() else 0.0
        )
        times_at_port[ch_info.name] = ws.time_axis
    return SimpleNamespace(
        device_uid=seqc_descriptor.device_uid,
        awg_index=seqc_descriptor.awg_index,
        sample_frequency=seqc_descriptor.sampling_rate,
        output=output,
        times=times,
        times_at_port=times_at_port,
    )


def run_single_source(descriptor: SeqCDescriptor, waves, max_time, scale_factor):
    core_simulation = run_single_source_impl(descriptor, waves, max_time)
    return _build_compatibility_output(descriptor, core_simulation)


def analyze_compiler_output_memory(
    compiled: CompiledExperiment, max_time=None, scale_factors=None
):
    seqc_descriptors, _ = _analyze_compiled(compiled)
    simulations = simulate(compiled, max_time=max_time)

    simulated_waves = {}
    for core_id, core_simulation in simulations.items():
        seqc_descriptor = next(d for d in seqc_descriptors if d.name == core_id)
        simulated_waves[core_id] = _build_compatibility_output(
            seqc_descriptor, core_simulation
        )

    return simulated_waves


def find_signal_start_times_for_result(
    runtimes: Dict[str, SimpleRuntime], threshold=1e-6, holdoff=50e-9
):
    retval = {}
    for name, runtime in runtimes.items():
        retval[name] = find_signal_start_times(runtime, threshold, holdoff)
    return retval


def find_signal_start_times(runtime: SimpleRuntime, threshold=1e-6, holdoff=50e-9):
    """Find the start of pulses

    Algorithm: Rectify, boxcar-filter with given width. Return points where the signal
    crosses the threshold.
    """
    retval = {}
    for output_key, wave in runtime.output.items():
        sample_period = runtime.times[output_key][1] - runtime.times[output_key][0]
        window_width = int(holdoff / sample_period)
        wave = np.convolve(np.abs(wave), np.ones(window_width))[: -window_width + 1]
        shifted = np.concatenate([[0], wave[:-1]])
        events = np.logical_and(wave > threshold, shifted < threshold)
        found = {
            "samples": np.arange(len(wave))[events],
            "time": runtime.times[output_key][events],
            "time_at_port": runtime.times_at_port[output_key][events],
        }
        retval[output_key] = found
    return retval
