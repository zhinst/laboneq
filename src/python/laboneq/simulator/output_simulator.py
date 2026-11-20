# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AWGSignalType, PhysicalChannelType
from laboneq.core.utilities.shfqc import VIRTUAL_SHFSG_UID_SUFFIX
from laboneq.dsl.device.io_units.physical_channel import PhysicalChannel
from laboneq.simulator.seqc_parser import simulate
from laboneq.simulator.wave_scroller import SimTarget, WaveScroller

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from laboneq.core.types.compiled_experiment import CompiledExperiment
    from laboneq.data.artifacts_qccs import (
        ArtifactsCodegen,
        Initialization,
        RealtimeExecutionInit,
    )


@dataclass
class OutputData:
    time: ArrayLike
    wave: ArrayLike | None = None
    trigger: ArrayLike | None = None
    marker: ArrayLike | None = None
    frequency: ArrayLike | None = None


def _find_seqc(
    *, device_uid: str, awg_no: int, rt_exec_inits: list[RealtimeExecutionInit]
) -> str:
    try:
        return next(
            iter(
                sorted(
                    (
                        rt_init
                        for rt_init in rt_exec_inits
                        if rt_init.awg_index == awg_no
                        and rt_init.device_id == device_uid
                    ),
                    key=lambda rt_init: rt_init.nt_step.indices,
                )
            )
        ).program_ref
    except StopIteration:
        raise LabOneQException(
            f"Unable to find SeqC program for AWG {awg_no} of device {device_uid}."
        ) from None


def _map_channel_type(
    *, device_uid: str, awg_no: int, inits: list[Initialization]
) -> PhysicalChannelType:
    for i in inits:
        if i.device_uid == device_uid:
            for awg in i.awgs:
                if awg.awg == awg_no:
                    return {
                        AWGSignalType.IQ: PhysicalChannelType.IQ_CHANNEL,
                        AWGSignalType.SINGLE: PhysicalChannelType.RF_CHANNEL,
                        AWGSignalType.DOUBLE: PhysicalChannelType.RF_CHANNEL,
                    }[awg.signal_type]

    raise LabOneQException(
        f"Unable to map channel type for AWG {awg_no} of device {device_uid}."
    )


def parse_channel_uid(channel_uid: str) -> tuple[str, str]:
    """Split a channel UID into device UID and channel name.

    A channel UID is of the format `<device uid>/<channel name>`
    where the format of `channel_name` differs between device- and
    channel types. Examples:

        device/sigouts_0
        device/sigouts_0_1
        device/qas_0
        device/qachannels_0_output
        device/qachannels_0_input
        device/sgchannels_0_output

    Args:
        channel_uid: A channel UID in the format `<device uid>/<channel name>`.

    Returns:
        A tuple containing the device UID and the channel name.
    """
    try:
        device_uid, channel_name = channel_uid.split("/")
    except ValueError:
        raise LabOneQException(
            f"Unable to parse channel UID {channel_uid}. "
            "Expected format is '<device uid>/<channel name>'."
        ) from None

    return device_uid, channel_name


@dataclass
class _AWG_ID:
    is_out: bool
    prog: str
    awg_channels: list[int]
    channel_type: PhysicalChannelType

    @classmethod
    def from_artifacts(
        cls,
        *,
        channel_uid: str,
        artifacts: ArtifactsCodegen,
    ) -> _AWG_ID:
        device_uid, channel_name = parse_channel_uid(channel_uid)
        port_type, *channel_attrs = channel_name.split("_")

        match port_type:
            case "sigouts":
                is_out = True
                awg_channels = [int(c) for c in channel_attrs]
                awg_no = awg_channels[0] // 2
            case "qas":
                is_out = False
                awg_channels = [int(c) for c in channel_attrs]
                awg_no = 0
            case "qachannels":
                is_out = channel_attrs[-1] == "output"
                awg_channels = [0]
                awg_no = int(channel_attrs[0])
            case "sgchannels":
                qc_sg_uid = f"{device_uid}{VIRTUAL_SHFSG_UID_SUFFIX}"

                device_uid = next(
                    (
                        qc_sg_uid
                        for init in artifacts.initializations
                        if init.device_uid == qc_sg_uid
                    ),
                    device_uid,
                )

                is_out = True
                awg_channels = [0, 1]
                awg_no = int(channel_attrs[0])
            case _:
                raise LabOneQException(f"Unknown port type '{port_type}'.")

        return _AWG_ID(
            is_out=is_out,
            awg_channels=awg_channels,
            prog=_find_seqc(
                device_uid=device_uid,
                awg_no=awg_no,
                rt_exec_inits=artifacts.realtime_execution_init,
            ),
            channel_type=_map_channel_type(
                device_uid=device_uid, awg_no=awg_no, inits=artifacts.initializations
            ),
        )


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
            Deprecated and has no effect. Use the `output_length` argument to
            the `get_snippet` method instead.

    Examples:
        Example showing how to compile an experiment and make use of the
        output simulator:

        ``` py
        # Given compiled_experiment
        compiled_experiment = session.compile(exp)

        # Create an output simulation object
        output_simulator = OutputSimulator(compiled_experiment)

        # By default, simulation is stopped after 10 ms, but it can be explicitly specified
        output_simulator = OutputSimulator(
            compiled_experiment, max_simulation_length=10e-3
        )

        # Also the maximum output snippet length is configurable, defaulting to 1us
        output_simulator = OutputSimulator(compiled_experiment, max_output_length=5e-6)

        # Maximum output length can also be set later
        output_simulator.max_output_length = 5e-6

        # As next, retrieve the actual simulated waveform
        data = output_simulator.get_snippet(
            physical_channel,
            start=1e-6,
            output_length=500e-9,
            get_wave=True,  # Default True
            get_trigger=True,  # Default False
            get_frequency=True,  # Default False
        )

        # Returned structure has 4 members, each is a numpy array
        data.time  # time axis
        data.wave  # waveform data
        data.trigger  # trigger values
        data.frequency  # frequency data
        ```
    """

    def __init__(
        self,
        compiled_experiment: CompiledExperiment,
        max_simulation_length: float | None = None,
        max_output_length: float = 10e-6,
    ) -> None:
        self._compiled_experiment = compiled_experiment
        self._max_output_length = max_output_length
        self._simulations = simulate(
            compiled_experiment, max_time=max_simulation_length or 10e-3
        )

    @property
    def max_output_length(self) -> float:
        return self._max_output_length

    @max_output_length.setter
    def max_output_length(self, max_output_length: float):
        self._max_output_length = max_output_length

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
            physical_channel.uid
            if isinstance(physical_channel, PhysicalChannel)
            else physical_channel
        )

        artifacts = self._compiled_experiment.scheduled_experiment.artifacts

        awg_id = _AWG_ID.from_artifacts(
            channel_uid=channel,
            artifacts=artifacts,
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
            ch=awg_id.awg_channels,
            sim_targets=sim_targets,
            sim=sim,
            channel_type=awg_id.channel_type,
        )
        ws.calc_snippet(start, output_length)
        return OutputData(
            time=ws.time_axis,
            wave=ws.wave_snippet if awg_id.is_out else ws.acquire_snippet,
            marker=ws.marker_snippet,
            trigger=ws.trigger_snippet,
            frequency=ws.frequency_snippet,
        )
