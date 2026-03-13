# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from laboneq.data.compilation_job import ParameterInfo, SignalRange

if TYPE_CHECKING:
    from laboneq.data.awg_info import AWGInfo


@dataclass(init=True, repr=True, order=True)
class SignalObj:
    """A collection of experiment signal properties relevant for code generation. The delay
    fields are in seconds and their meaning is as follows:
    - id: signal's id
    - signal_type: one of "iq" / "single" / "integration" - see SignalInfoType
    - local_oscillator_frequency: signal's local oscillator frequency, if defined via calibration; see SignalInfo.lo_frequency
    - channels: list of physical-port channels relevant to the signal
    - channel_to_port: map of signal-relevant physical-port channels to their paths; see SignalInfo.channel_to_port, PhysicalChannel.ports
    - awg: reference to the signal's awg, see AWGInfo
    - port_delay: port delay specified via calibration; realized via the device node in addition to potential on-device delays
    - mixer_type: IQ (complex modulation) or UHFQA_ENVELOPE (envelope modulation); see MixerType
    - is_qc: flag indicating signal's device type being (SHF)QC
    - automute: The signal output can be automatically muted when no waveforms are played; see SignalInfo.automute
    - signal_range: The selected signal's input / output range; see SignalInfo.signal_range
    """

    id: str
    signal_type: str
    local_oscillator_frequency: float | None = None
    channels: list[int] = field(default_factory=list)
    channel_to_port: dict[int, str] = field(default_factory=dict)
    awg: AWGInfo | None = None
    port_delay: float | ParameterInfo = 0.0
    is_qc: bool | None = None
    automute: bool = False
    signal_range: SignalRange | None = None
