# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

import numpy as np
from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.simulator.seqc_parser import SeqCMetadata, SimpleRuntime, _analyze_compiled
from laboneq.simulator.seqc_parser import run_single_source as run_single_source_impl
from laboneq.simulator.seqc_parser import simulate
from laboneq.simulator.wave_scroller import WaveScroller


def _get_channel_mapping(
    device_type: str, channels: List[int]
) -> List[Tuple[str, int, Callable, bool]]:
    # output key, channel, wave getter, True if output port delay applicable
    channel_mapping: List[Tuple[str, int, Callable, bool]] = []
    if device_type != "SHFQA":
        for ch, i in enumerate(channels):
            channel_mapping.append((f"{i}", ch, WaveScroller.get_play_snippet, True))

    if device_type == "UHFQA":
        channel_mapping.append(
            ("QAResult", 0xFFFF, WaveScroller.get_acquire_snippet, False)
        )

    if device_type == "SHFQA":
        channel_mapping = [
            ("0", 0, WaveScroller.get_play_snippet, True),
            ("1", 1, WaveScroller.get_play_snippet, True),
            ("QAResult", 0xFFFF, WaveScroller.get_acquire_snippet, False),
            ("QAResult_0", 0x0001, WaveScroller.get_acquire_snippet, False),
            ("QAResult_1", 0x0002, WaveScroller.get_acquire_snippet, False),
            ("osc0_freq", 0, WaveScroller.get_freq_snippet, True),
        ]

    if device_type == "SHFSG":
        channel_mapping = [
            (f"{channels[0]}_I", 0, WaveScroller.get_play_snippet, True),
            (f"{channels[0]}_Q", 1, WaveScroller.get_play_snippet, True),
            ("osc0_freq", 0, WaveScroller.get_freq_snippet, True),
        ]

    return channel_mapping


def _build_compatibility_output(
    ws: WaveScroller, seqc_descriptor: SeqCMetadata
) -> SimpleNamespace:
    channel_mapping = _get_channel_mapping(
        seqc_descriptor.device_type, seqc_descriptor.channels
    )
    output = {}
    times = {}
    times_at_port = {}
    for (ch_out, ch, getter, is_out) in channel_mapping:
        time, wave = getter(
            ws,
            start_secs=-0.5,
            length_secs=1,
            prog=seqc_descriptor.name,
            ch=ch,
        )
        output[ch_out] = wave
        times[ch_out] = time - (seqc_descriptor.output_port_delay if is_out else 0.0)
        times_at_port[ch_out] = time
    return SimpleNamespace(
        device_uid=seqc_descriptor.device_uid,
        awg_index=seqc_descriptor.awg_index,
        sample_frequency=seqc_descriptor.sampling_rate,
        output=output,
        times=times,
        times_at_port=times_at_port,
    )


def run_single_source(descriptor: SeqCMetadata, waves, max_time, scale_factor):
    core_simulation = run_single_source_impl(descriptor, waves, max_time)
    ws = WaveScroller({descriptor.name: core_simulation})
    return _build_compatibility_output(ws, descriptor)


def analyze_compiler_output_memory(
    compiled: CompiledExperiment, max_time=None, scale_factors=None
):
    seqc_descriptors, _ = _analyze_compiled(compiled)
    simulation = simulate(compiled, max_time=max_time)
    ws = WaveScroller(simulation)

    simulated_waves = {}
    for core_id in simulation.keys():
        seqc_descriptor = next(d for d in seqc_descriptors if d.name == core_id)
        simulated_waves[core_id] = _build_compatibility_output(ws, seqc_descriptor)

    return simulated_waves


def find_signal_start_times_for_result(
    runtimes: Dict[str, SimpleRuntime], threshold=1e-6, holdoff=50e-9
):
    retval = {}
    for name, runtime in runtimes.items():
        retval[name] = find_signal_start_times(runtime, threshold, holdoff)
    return retval


def find_signal_start_times(runtime: SimpleRuntime, threshold=1e-6, holdoff=50e-9):
    retval = {}
    for output_key, wave in runtime.output.items():
        found = {"samples": [], "time": [], "time_at_port": []}
        last_found = None
        for i, val in enumerate(wave):
            current_time_at_port = runtime.times_at_port[output_key][i]
            current_time = runtime.times[output_key][i]
            if np.abs(val) > threshold:
                if last_found is None or current_time - last_found > holdoff:
                    found["samples"].append(i)
                    found["time"].append(current_time)
                    found["time_at_port"].append(current_time_at_port)
                last_found = current_time
        retval[output_key] = found
    return retval
