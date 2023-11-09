# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.data.recipe import RealtimeExecutionInit
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.device.instruments.shfqc import SHFQC
from laboneq.dsl.device.io_units.physical_channel import PhysicalChannel
from laboneq.simulator.seqc_parser import simulate
from laboneq.simulator.wave_scroller import SimTarget, WaveScroller


@dataclass
class OutputData:
    time: ArrayLike
    wave: ArrayLike = None
    trigger: ArrayLike = None
    marker: ArrayLike = None
    frequency: ArrayLike = None


@dataclass
class _AWG_ID:
    is_out: bool
    prog: str
    channels: list[int]

    def __init__(
        self,
        device_setup: DeviceSetup,
        ch: PhysicalChannel,
        realtime_inits: list[RealtimeExecutionInit],
    ):
        self._device_setup = device_setup
        [self._dev_uid, pch] = ch.uid.split("/")
        ch_attrs = pch.split("_")
        {
            "sigouts": self._decode_sigouts,
            "qas": self._decode_qas,
            "qachannels": self._decode_qachannels,
            "sgchannels": self._decode_sgchannels,
        }[ch_attrs[0]](ch_attrs[1:], realtime_inits)

    def find_seqc(
        self, device_name: str, awg_no: int, realtime_inits: list[RealtimeExecutionInit]
    ):
        self.prog = next(
            iter(
                sorted(
                    (
                        rt_init
                        for rt_init in realtime_inits
                        if rt_init.awg_id == awg_no and rt_init.device_id == device_name
                    ),
                    key=lambda rt_init: rt_init.nt_step.indices,
                )
            )
        ).seqc_ref

    def _decode_sigouts(
        self, chs: list[str], realtime_inits: list[RealtimeExecutionInit]
    ):
        self.is_out = True
        self.channels = [int(ch) for ch in chs]
        awg_no = self.channels[0] // 2
        self.find_seqc(self._dev_uid, awg_no, realtime_inits)

    def _decode_qas(self, chs: list[str], realtime_inits: list[RealtimeExecutionInit]):
        self.is_out = False
        self.channels = [int(ch) for ch in chs]
        self.find_seqc(self._dev_uid, 0, realtime_inits)

    def _is_qc(self):
        return isinstance(self._device_setup.instrument_by_uid(self._dev_uid), SHFQC)

    def _decode_qachannels(
        self, chs: list[str], realtime_inits: list[RealtimeExecutionInit]
    ):
        self.is_out = chs[1] == "output"
        self.channels = [0]
        self.find_seqc(self._dev_uid, int(chs[0]), realtime_inits)

    def _decode_sgchannels(
        self, chs: list[str], realtime_inits: list[RealtimeExecutionInit]
    ):
        internal_device_name = (
            self._dev_uid if not self._is_qc() else f"{self._dev_uid}_sg"
        )

        self.is_out = True
        self.channels = [0, 1]
        self.find_seqc(internal_device_name, int(chs[0]), realtime_inits)


class OutputSimulator:
    """Interface to the output simulator.

    Arguments:
        compiled_experiment:
            The compiled experiment to simulate.
        max_simulation_length:
            The maximum amount of time to simulate (in seconds).
        max_output_length:
            Deprecated and has no effect. Use the `output_length` argument to
            the `get_snippet` method instead.

    Attributes:
        max_output_length:
            Deprecated nad has no effect. Use the `output_length` argument to
            the `get_snippet` method instead.

    Examples:

        Example showing how to compile an experiment and make use of the
        output simulator:

        ``` py
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
        ```
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
        channel_by_uids = {
            ch.uid: ch for g in pcg.values() for ch in g.channels.values()
        }
        try:
            return channel_by_uids[uid]
        except KeyError as e:
            raise RuntimeError(
                f"Can't find physical channel with uid '{uid}'."
                f" Available channels: {', '.join(channel_by_uids.keys())}"
            ) from e

    def get_snippet(
        self,
        physical_channel: str | PhysicalChannel,
        start: float,
        output_length: float,
        get_wave: bool = True,
        get_trigger: bool = False,
        get_marker: bool = False,
        get_frequency: bool = False,
    ) -> OutputData:
        """Retrieve the simulated waveforms for a given channel and window of time.

        Arguments:
            physical_channel: The physical channel to retrieve waveforms for.
            start: The start time of the window of events to retrieve (in seconds).
            output_length: The maximum length of the window to retrieve (in seconds).
            get_wave: Whether to return the waveform data.
            get_trigger: Whether to return the trigger data.
            get_marker: Whether to return the marker data.
            get_frequency: Whether to return the oscillator frequency data.

        Returns:
            The output data has the following attributes:

                - `time`: an array of the times corresponding to the returned waveform samples.
                - `wave`: an array of waveform values at the given times.
                - `trigger`: an array of trigger values at the given times.
                - `frequency`: an array of oscillator frequencies at the given times.

                The corresponding attribute is `None` if the associated data was not
                requested.
        """
        channel = (
            physical_channel
            if isinstance(physical_channel, PhysicalChannel)
            else self._uid_to_channel(physical_channel)
        )
        awg_id = _AWG_ID(
            self._compiled_experiment.device_setup,
            channel,
            self._compiled_experiment.scheduled_experiment.recipe.realtime_execution_init,
        )

        sim = self._simulations[awg_id.prog]
        sim_targets = SimTarget.NONE
        if get_wave and awg_id.is_out:
            sim_targets |= SimTarget.PLAY
        if get_wave and not awg_id.is_out:
            sim_targets |= SimTarget.ACQUIRE
        if get_trigger and awg_id.is_out:
            sim_targets |= SimTarget.TRIGGER
        if get_marker and awg_id.is_out:
            sim_targets |= SimTarget.MARKER
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
            marker=ws.marker_snippet,
            trigger=ws.trigger_snippet,
            frequency=ws.frequency_snippet,
        )
