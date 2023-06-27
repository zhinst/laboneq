# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from laboneq.core.types.enums import PortMode
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.observable import Observable
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.precompensation import Precompensation
from laboneq.dsl.dsl_dataclass_decorator import classformatter
from laboneq.dsl.parameter import Parameter


@classformatter
@dataclass(init=False, order=True)
class SignalCalibration(Observable):
    """Dataclass containing all calibration parameters
    and settings related to a :class:`~.LogicalSignal`.
    """

    #: The oscillator assigned to the :class:`~.LogicalSignal`
    #: - determines the frequency and type of modulation for any pulses played back on this line.
    oscillator: Oscillator | None
    #: The local oscillator assigned to the :class:`~.LogicalSignal`
    #: - sets the center frequency of the playback - only relevant on SHFSG, SHFQA and SHFQC
    local_oscillator: Oscillator | None
    #: Settings to enable the optional mixer calibration correction
    #: - only applies to IQ signals on HDAWG
    mixer_calibration: MixerCalibration | None
    #: Settings to enable signal distortion precomensation
    #: - only applies to HDAWG instruments
    precompensation: Precompensation | None
    #: An optional delay of all output on this signal.
    #: Works by setting delay nodes on the instruments, and will not be visible in the pulse sheet.
    #: Not currently available on SHFSG output channels.
    port_delay: float | Parameter | None
    #: Allows to switch between amplified high-frequency mode (PortMode.RF)
    #: and direct low_frequency mode (PortMode.LF) on SHFSG output channels.
    port_mode: PortMode | None
    #: Defines an additional global delay on this signal line.
    #: Will be mapped to the waveforms and sequencer code emitted for this logical signal,
    #: and thus visible in the pulse sheet viewer.
    delay_signal: float | None
    #: Allows to set a constant voltage offset on individual rf lines on the HDAWG.
    voltage_offset: float | None
    #: The output or input range setting for the logical signal
    range: int | float | None
    #: The state discrimination threshold
    #: - only relevant for acquisition type signals on UHFQA and SHFQA/SHFQC
    threshold: float | None
    #: (Not Implemented) Amplitude multiplying all waveforms played on a signal line
    amplitude: float | Parameter | None
    #: Parametric Pump Controller settings
    amplifier_pump: AmplifierPump | None

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
