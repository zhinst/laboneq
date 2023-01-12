# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Union

from laboneq.core.types.enums import PortMode
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.observable import Observable
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.precompensation import Precompensation


@dataclass(init=False, order=True)
class SignalCalibration(Observable):
    amplitude: Optional[float]
    delay_signal: Optional[float]
    local_oscillator: Optional[Oscillator]
    voltage_offset: Optional[float]
    mixer_calibration: Optional[MixerCalibration]
    precompensation: Optional[Precompensation]
    oscillator: Optional[Oscillator]
    port_delay: Optional[float]
    port_mode: Optional[PortMode]
    range: Union[int, float, None]
    threshold: Optional[float]

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
    def mixer_calibration(self, value: Optional[MixerCalibration]):
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
    def precompensation(self, value: Optional[Precompensation]):
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
