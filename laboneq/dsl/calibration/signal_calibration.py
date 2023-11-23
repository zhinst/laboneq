# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from laboneq.core.types.enums import PortMode
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.observable import Observable
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.output_routing import OutputRoute
from laboneq.dsl.calibration.precompensation import Precompensation
from laboneq.dsl.parameter import Parameter


@classformatter
@dataclass(init=False, order=True)
class SignalCalibration(Observable):
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
            Sets the center frequency of the playback. Only supported by
            SHFSG, SHFQA and SHFQC signals.
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
        voltage_offset (float | None):
            On the HDAWG lines, the voltage offset may be used to set a
            constant voltage offset on individual RF line.
        range (int | float | None):
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
    """

    oscillator: Oscillator | None = None
    local_oscillator: Oscillator | None = None
    mixer_calibration: MixerCalibration | None = None
    precompensation: Precompensation | None = None
    port_delay: float | Parameter | None = None
    port_mode: PortMode | None = None
    delay_signal: float | None = None
    voltage_offset: float | None = None
    range: int | float | None = None
    threshold: float | list[float] | None = None
    amplitude: float | Parameter | None = None
    amplifier_pump: AmplifierPump | None = None
    added_outputs: list[OutputRoute] | None = None

    def __init__(
        self,
        amplitude=None,
        delay_signal=None,
        local_oscillator=None,
        voltage_offset=None,
        mixer_calibration=None,
        precompensation=None,
        oscillator=None,
        port_delay=None,
        port_mode=None,
        range=None,
        threshold=None,
        amplifier_pump=None,
        added_outputs=None,
    ):
        super().__init__()
        self.amplitude = amplitude
        self.delay_signal = delay_signal
        self.local_oscillator = local_oscillator
        self.voltage_offset = voltage_offset
        self._mixer_calibration = mixer_calibration
        self._precompensation = precompensation
        self.oscillator = oscillator
        self.port_delay = port_delay
        self.port_mode = port_mode
        self.range = range
        self.threshold = threshold
        self.amplifier_pump = amplifier_pump
        self.added_outputs = added_outputs

        super().__post_init__()
        if self._mixer_calibration is not None:
            self._mixer_calibration.has_changed().connect(
                self._mixer_calibration_changed_callback
            )
        if self._precompensation is not None:
            self._precompensation.has_changed().connect(
                self._precompensation_changed_callback
            )

    @property
    def mixer_calibration(self) -> MixerCalibration:
        return self._mixer_calibration

    @mixer_calibration.setter
    def mixer_calibration(self, value: MixerCalibration | None):
        if value is self._mixer_calibration:
            return

        if self._mixer_calibration is not None:
            self._mixer_calibration.has_changed().disconnect(
                self._mixer_calibration_changed_callback
            )
        self._mixer_calibration = value
        if self._mixer_calibration is not None:
            self._mixer_calibration.has_changed().connect(
                self._mixer_calibration_changed_callback
            )

    def _mixer_calibration_changed_callback(
        self, calibration: MixerCalibration, key: str, value
    ):
        self.has_changed().fire("mixer_calibration", calibration)

    @property
    def precompensation(self) -> Precompensation:
        return self._precompensation

    @precompensation.setter
    def precompensation(self, value: Precompensation | None):
        if value is self._precompensation:
            return

        if self._precompensation is not None:
            self._precompensation.has_changed().disconnect(
                self._precompensation_changed_callback
            )
        self._precompensation = value
        if self._precompensation is not None:
            self._precompensation.has_changed().connect(
                self._precompensation_changed_callback
            )

    def _precompensation_changed_callback(
        self, calibration: Precompensation, key: str, value
    ):
        self.has_changed().fire("precompensation", calibration)
