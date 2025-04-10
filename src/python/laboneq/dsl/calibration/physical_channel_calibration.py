# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TypeVar


from laboneq.core.types.enums import PortMode
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.output_routing import OutputRoute
from laboneq.dsl.calibration.precompensation import Precompensation
from laboneq.dsl.parameter import Parameter
import attrs

T = TypeVar("T")


@classformatter
@attrs.define(
    kw_only=True,
    on_setattr=lambda self, attr, value: self.__on_setattr_callback__(attr.name, value),
    slots=False,  # Needed for users to bypass __setattr__
)
class PhysicalChannelCalibration:
    """Calibration parameters and settings for a
    [PhysicalChannel][laboneq.dsl.device.io_units.physical_channel.PhysicalChannel].

    Attributes:
        local_oscillator (Oscillator | None):
            The local oscillator assigned to the signal.
            Sets the center frequency of the playback. `modulation_type` of the
            assigned `Oscillator` must be either `ModulationType.HARDWARE` or
            `ModulationType.AUTO`. Only supported by SHFSG, SHFQA and SHFQC signals.
            Default: `None`.
        mixer_calibration (MixerCalibration | None):
            Mixer calibration correction settings.
            Only supported by HDAWG IQ signals.
            Default: `None`.
        precompensation (Precompensation | None):
            Signal distortion precompensation settings.
            Only supported by HDAWG signals.
            Default: `None`.
        port_delay (float | Parameter | None):
            An optional delay of all output on this signal.
            Implemented by setting delay nodes on the instruments, and will
            not be visible in the pulse sheet.
            Not currently supported on SHFSG output channels.
            Units: seconds.
            Default: `None`.
        port_mode (PortMode | None):
            On SHFSG, SHFQA and SHFQC, the port mode may be set to select
            either amplified high-frequency mode (PortMode.RF, default) or
            baseband mode (PortMode.LF).
            Default: `None`.
        voltage_offset (float | Parameter | None):
            On the HDAWG lines, the voltage offset may be used to set a
            constant voltage offset on individual RF line.
        range (int | float | None):
            The output or input range setting for the signal.
        amplitude (float | Parameter | None):
            Amplitude multiplying all waveforms played on the signal line.
            Only supported by the SHFQA.
        amplifier_pump (AmplifierPump | None):
            Parametric Pump Controller settings.
        added_outputs (list[OutputRoute] | None):
            Added outputs to the signal line's physical channel port.
            Only available for SHFSG/SHFQC devices with Output Router and Adder (RTR) option enabled.
            Only viable for signals which point to 'SGCHANNELS/N/OUTPUT' physical ports.
    """

    local_oscillator: Oscillator | None = None
    mixer_calibration: MixerCalibration | None = None
    precompensation: Precompensation | None = None
    port_delay: float | Parameter | None = None
    port_mode: PortMode | None = None
    voltage_offset: float | Parameter | None = None
    range: int | float | None = None
    amplitude: float | Parameter | None = None
    amplifier_pump: AmplifierPump | None = None
    added_outputs: list[OutputRoute] | None = None

    def __on_setattr_callback__(self, attr: attrs.Attribute, value: T) -> T:
        return value
