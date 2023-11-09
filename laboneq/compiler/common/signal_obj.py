# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.core.types.enums.mixer_type import MixerType

if TYPE_CHECKING:
    from laboneq.compiler.common.awg_info import AWGInfo


@dataclass(init=True, repr=True, order=True)
class SignalObj:
    """A collection of a signal's properties relevant for code generation. The delay
    fields are in seconds and their meaning is as follows:
    - start_delay: the delay from the trigger to the start of the sequence (lead time),
      realized as initial playZeros; includes lead time and precompensation
    - delay_signal: user-defined additional delays, realized by adding to the initial
      playZeros; rounded to the sequencer grid (sample_multiple)
    - base_delay_signal: in case of an acquisition pulse, the delay_signal of the
      corresponding measure pulse on the same AWG
    - total_delay: the sum of the above two fields, plus delays generated during code
      generation, e.g., relative delay between a play and acquire pulse
    - on_device_delay: delay on the device, realized by delay nodes and independent
      from the sequencer, generated during code generation, e.g., relative delay between
      a play and acquire pulse; in addition to potential port delays specified via the
      calibration; a list which can contain multiple values due to the way we handle
      acquisition delays (the delay of the measure pulse channel is added to the delay
      of the acquire pulse channel)
    - port_delay: port delay specified via the calibration; realized via the device node
      in addition to potential on-device delays.
    - base_port_delay: in case of an acquisition pulse, the port_delay of the
      corresponding measure pulse on the same AWG
    """

    id: str
    start_delay: float
    delay_signal: float
    signal_type: AWGSignalType | str
    base_delay_signal: float | None = None
    oscillator_frequency: float = None  # for software modulation only
    channels: List = field(default_factory=list)
    awg: AWGInfo = None
    total_delay: float = None
    on_device_delay: float = 0
    port_delay: float = 0
    base_port_delay: float | None = None
    mixer_type: MixerType | None = None
    hw_oscillator: str | None = None
    is_qc: bool | None = None
