# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

from numpy.typing import ArrayLike

from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.device.io_units.physical_channel import PhysicalChannel
from laboneq.simulator.seqc_parser import simulate
from laboneq.simulator.wave_scroller import SimTarget, WaveScroller


@dataclass
class OutputData:
    time: ArrayLike
    wave: ArrayLike = None
    trigger: ArrayLike = None
    frequency: ArrayLike = None


@dataclass
class _AWG_ID:
    is_out: bool
    prog: str
    channels: List[int]

    def __init__(self, device_setup: DeviceSetup, ch: PhysicalChannel):
        self._device_setup = device_setup
        [self._dev_uid, pch] = ch.uid.split("/")
        ch_attrs = pch.split("_")
        {
            "sigouts": self._decode_sigouts,
            "qas": self._decode_qas,
            "qachannels": self._decode_qachannels,
            "sgchannels": self._decode_sgchannels,
        }[ch_attrs[0]](ch_attrs[1:])

    def _decode_sigouts(self, chs: List[str]):
        self.is_out = True
        self.channels = [int(ch) for ch in chs]
        awg_no = self.channels[0] // 2
        self.prog = f"seq_{self._dev_uid}_{awg_no}.seqc"

    def _decode_qas(self, chs: List[str]):
        self.is_out = False
        self.channels = [int(ch) for ch in chs]
        self.prog = f"seq_{self._dev_uid}_0.seqc"

    def _is_qc(self):
        dev = self._device_setup.instrument_by_uid(self._dev_uid)
        return dev.calc_driver() == "SHFQA" and dev.is_qc

    def _decode_qachannels(self, chs: List[str]):
        self.is_out = chs[1] == "output"
        self.channels = [0]
        self.prog = f"seq_{self._dev_uid}_{chs[0]}.seqc"

    def _decode_sgchannels(self, chs: List[str]):
        self.is_out = True
        self.channels = [0, 1]
        if self._is_qc():
            self.prog = f"seq_{self._dev_uid}_sg_{chs[0]}.seqc"
        else:
            self.prog = f"seq_{self._dev_uid}_{chs[0]}.seqc"


class OutputSimulator:
    """Interface to the output simulator.

    .. highlight:: python
    .. code-block:: python

        # Usage:

        # Given compiled_experiment
        compiled_experiment = session.compile(exp)

        # Create an output simulation object
        output_simulator = OutputSimulator(compiled_experiment)

        # By default, simulation is stopped after 1ms, but it can be explicitly specified
        output_simulator = OutputSimulator(compiled_experiment, max_simulation_length = 10e-3)

        # Also the maximum output snippet length is configurable, defaulting to 1us
        output_simulator = OutputSimulator(compiled_experiment, max_output_length = 5e-6)

        # Maximum output length can also be set later
        output_simulator.max_output_length = 5e-6


        # As next, retrieve the actual simulated waveform
        data = output_simulator.get_snippet(
            physical_channel,
            start = 1e-6,
            output_length = 500e-9,
            get_wave=True,       # Default True
            get_trigger=True,    # Default False
            get_frequency=True,  # Default False
        )

        # Returned structure has 4 members, each is a numpy array
        data.time       # time axis
        data.wave       # waveform data
        data.trigger    # trigger values
        data.frequency  # frequency data
    """

    def __init__(
        self,
        compiled_experiment: CompiledExperiment,
        max_simulation_length: float = 10e-3,
        max_output_length: float = 10e-6,
    ) -> None:
        self._compiled_experiment = compiled_experiment
        self._max_output_length = max_output_length
        self._simulations = simulate(
            compiled_experiment, max_time=max_simulation_length
        )

    @property
    def max_output_length(self) -> float:
        return self._max_output_length

    @max_output_length.setter
    def max_output_length(self, max_output_length: float):
        self._max_output_length = max_output_length

    def _uid_to_channel(self, uid: str) -> PhysicalChannel:
        pcg = self._compiled_experiment.device_setup.physical_channel_groups
        for g in pcg.values():
            for ch in g.channels.values():
                if ch.uid == uid:
                    return ch
        raise RuntimeError(f"Can't find physical channel with uid '{uid}'")

    def get_snippet(
        self,
        physical_channel: Union[str, PhysicalChannel],
        start: float,
        output_length: float,
        get_wave: bool = True,
        get_trigger: bool = False,
        get_frequency: bool = False,
    ) -> OutputData:
        channel = (
            physical_channel
            if isinstance(physical_channel, PhysicalChannel)
            else self._uid_to_channel(physical_channel)
        )
        awg_id = _AWG_ID(self._compiled_experiment.device_setup, channel)

        sim = self._simulations[awg_id.prog]
        sim_targets = SimTarget.NONE
        if get_wave and awg_id.is_out:
            sim_targets |= SimTarget.PLAY
        if get_wave and not awg_id.is_out:
            sim_targets |= SimTarget.ACQUIRE
        if get_trigger and awg_id.is_out:
            sim_targets |= SimTarget.TRIGGER
        if get_frequency and awg_id.is_out:
            sim_targets |= SimTarget.FREQUENCY
        ws = WaveScroller(
            ch=awg_id.channels,
            sim_targets=sim_targets,
            sim=sim,
        )
        ws.calc_snippet(start, output_length)
        return OutputData(
            time=ws.time_axis,
            wave=ws.wave_snippet if awg_id.is_out else ws.acquire_snippet,
            trigger=ws.trigger_snippet,
            frequency=ws.frequency_snippet,
        )
