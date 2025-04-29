# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TypeVar

from laboneq.core.types.enums import PortMode
from laboneq.core.types.enums.modulation_type import ModulationType
from laboneq.core.types.units import Quantity
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.output_routing import OutputRoute
from laboneq.dsl.calibration.precompensation import Precompensation
from laboneq.dsl.parameter import Parameter
import attrs

T = TypeVar("T")


def _check_local_oscillator_does_not_use_sw_modulation(
    inst: attrs.AttrsInstance, attr: attrs.Attribute, value: Oscillator | None
):
    if value is not None and value.modulation_type == ModulationType.SOFTWARE:
        raise ValueError(
            "Local oscillator's modulation type can not be `ModulationType.SOFTWARE`."
        )


@classformatter
@attrs.define(  # TODO: add kw_only=True
    on_setattr=lambda self, attr, value: self.__on_setattr_callback__(attr.name, value),
    slots=False,  # Needed for users to bypass __setattr__
)
class SignalCalibration:
    """Calibration parameters and settings for a
    [LogicalSignal][laboneq.dsl.device.io_units.logical_signal.LogicalSignal].

    Attributes:
        oscillator (Oscillator | None):
            The oscillator assigned to the signal.
            Determines the frequency and type of modulation for any pulses
            played back on this line.
            Default: `None`.
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
        delay_signal (float | None):
            Defines an additional global delay on this signal line.
            Implemented by adjusting the waveforms and sequencer code emitted
            for this logical signal, and is thus visible in the pulse sheet.
            Default: `None`.
        voltage_offset (float | Parameter | None):
            On the HDAWG lines, the voltage offset may be used to set a
            constant voltage offset on individual RF line.
        range (int | float | Quantity | None):
            The output or input range setting for the signal.
        threshold (float | list[float] | None):
            Specify the state discrimination threshold.
            Only supported for acquisition signals on the UHFQA, SHFQA
            and SHFQC.
        amplitude (float | Parameter | None):
            Amplitude multiplying all waveforms played on the signal line.
            Only supported by the SHFQA.
        amplifier_pump (AmplifierPump | None):
            Parametric Pump Controller settings.
        added_outputs (list[OutputRoute] | None):
            Added outputs to the signal line's physical channel port.
            Only available for SHFSG/SHFQC devices with Output Router and Adder (RTR) option enabled.
            Only viable for signals which point to 'SGCHANNELS/N/OUTPUT' physical ports.
        automute (bool):
            Mute output channel when no waveform is played on it i.e for the duration of delays.
            Only available on SHF+ output channels.
    """

    amplitude: float | Parameter | None = None
    delay_signal: float | None = None
    local_oscillator: Oscillator | None = attrs.field(
        validator=_check_local_oscillator_does_not_use_sw_modulation, default=None
    )
    voltage_offset: float | Parameter | None = None
    mixer_calibration: MixerCalibration | None = None
    precompensation: Precompensation | None = None
    oscillator: Oscillator | None = None
    port_delay: float | Parameter | None = None
    port_mode: PortMode | None = None
    range: int | float | Quantity | None = None
    threshold: float | list[float] | None = None
    amplifier_pump: AmplifierPump | None = None
    added_outputs: list[OutputRoute] | None = None
    automute: bool = False

    def __on_setattr_callback__(self, attr: attrs.Attribute, value: T) -> T:
        return value
