# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.calibration import MixerCalibration, SignalCalibration

experiment_signal_id = 0


def experiment_signal_id_generator():
    global experiment_signal_id
    retval = f"s{experiment_signal_id}"
    experiment_signal_id += 1
    return retval


@dataclass(init=False, repr=True, order=True)
class ExperimentSignal:
    """Class representing a signal within an experiment. Experiment signals are connected to logical signals."""

    uid: str
    calibration: Optional[SignalCalibration]
    mapped_logical_signal_path: Optional[str]

    def __init__(
        self,
        uid=None,
        map_to=None,
        calibration=None,
        oscillator=None,
        amplitude: float = None,
        port_delay: float = None,
        delay_signal: float = None,
        mixer_calibration=None,
        precompensation=None,
        local_oscillator=None,
        range: float = None,
        port_mode=None,
        threshold: float = None,
        mapped_logical_signal_path=None,
    ):
        if uid is None:
            self.uid = experiment_signal_id_generator()
        else:
            self.uid = uid

        if (calibration is None) and (
            oscillator
            or amplitude is not None
            or port_delay is not None
            or delay_signal is not None
            or mixer_calibration is not None
            or precompensation is not None
            or local_oscillator is not None
            or range is not None
            or port_mode is not None
            or threshold is not None
        ):
            self.calibration = SignalCalibration(
                oscillator=oscillator,
                amplitude=amplitude,
                port_delay=port_delay,
                delay_signal=delay_signal,
                mixer_calibration=mixer_calibration,
                precompensation=precompensation,
                local_oscillator=local_oscillator,
                range=range,
                port_mode=port_mode,
                threshold=threshold,
            )
        else:
            self.calibration = calibration
        self.mapped_logical_signal_path = None
        self.map(map_to)
        if mapped_logical_signal_path is not None:
            self.map(mapped_logical_signal_path)

    def is_mapped(self):
        return self.mapped_logical_signal_path is not None

    def map(self, to):
        if isinstance(to, str):
            self.mapped_logical_signal_path = to
        else:
            if to is not None and (not hasattr(to, "path") or to.path is None):
                raise LabOneQException(
                    "Invalid LogicalSignal: Seems like the logical signal is not part of a qubit setup. Make sure the object is retrieved from a device setup."
                )
            self.mapped_logical_signal_path = to.path if to is not None else None

    def disconnect(self):
        """Disconnect the experiment signal from the logical signal."""
        self.mapped_logical_signal_path = None

    @property
    def mixer_calibration(self):
        return self.calibration.mixer_calibration if self.is_calibrated() else None

    @mixer_calibration.setter
    def mixer_calibration(self, value):
        if self.is_calibrated():
            self.calibration.mixer_calibration = value
        else:
            self.calibration = SignalCalibration(mixer_calibration=value)

    @property
    def precompensation(self):
        return self.calibration.precompensation if self.is_calibrated() else None

    @precompensation.setter
    def precompensation(self, value):
        if self.is_calibrated():
            self.calibration.precompensation = value
        else:
            self.calibration = SignalCalibration(precompensation=value)

    @property
    def oscillator(self):
        return self.calibration.oscillator if self.is_calibrated() else None

    @oscillator.setter
    def oscillator(self, value):
        if self.is_calibrated():
            self.calibration.oscillator = value
        else:
            self.calibration = SignalCalibration(oscillator=value)

    @property
    def amplitude(self):
        return self.calibration.amplitude if self.is_calibrated() else None

    @amplitude.setter
    def amplitude(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(amplitude=value)
        else:
            self.calibration.amplitude = value

    @property
    def port_delay(self):
        return self.calibration.port_delay if self.is_calibrated() else None

    @port_delay.setter
    def port_delay(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(port_delay=value)
        else:
            self.calibration.port_delay = value

    @property
    def delay_signal(self):
        return self.calibration.delay_signal if self.is_calibrated() else None

    @delay_signal.setter
    def delay_signal(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(delay_signal=value)
        else:
            self.calibration.delay_signal = value

    @property
    def voltage_offset(self):
        return self.calibration.voltage_offset if self.is_calibrated() else None

    @voltage_offset.setter
    def voltage_offset(self, value):
        if self.is_calibrated():
            self.calibration.voltage_offset = value
        else:
            self.calibration = SignalCalibration(voltage_offset=value)

    @property
    def voltage_offsets(self):
        return self.calibration.voltage_offsets if self.is_calibrated() else None

    @voltage_offsets.setter
    def voltage_offsets(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(
                mixer_calibration=MixerCalibration(voltage_offsets=value)
            )
        else:
            self.calibration.voltage_offsets = value

    @property
    def correction_matrix(self):
        return self.calibration.correction_matrix if self.is_calibrated() else None

    @correction_matrix.setter
    def correction_matrix(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(
                mixer_calibration=MixerCalibration(correction_matrix=value)
            )
        else:
            self.calibration.correction_matrix = value

    @property
    def local_oscillator(self):
        return self.calibration.local_oscillator if self.is_calibrated() else None

    @local_oscillator.setter
    def local_oscillator(self, value):
        if self.is_calibrated():
            self.calibration.local_oscillator = value
        else:
            self.calibration = SignalCalibration(local_oscillator=value)

    @property
    def range(self):
        return self.calibration.range if self.is_calibrated() else None

    @range.setter
    def range(self, value):
        if self.is_calibrated():
            self.calibration.range = value
        else:
            self.calibration = SignalCalibration(range=value)

    @property
    def port_mode(self):
        return self.calibration.port_mode if self.is_calibrated() else None

    @port_mode.setter
    def port_mode(self, value):
        if self.is_calibrated():
            self.calibration.port_mode = value
        else:
            self.calibration = SignalCalibration(port_mode=value)

    @property
    def threshold(self):
        return self.calibration.threshold if self.is_calibrated() else None

    @threshold.setter
    def threshold(self, value: float):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(threshold=value)
        else:
            self.calibration.threshold = value

    def is_calibrated(self):
        return self.calibration is not None

    def reset_calibration(self, calibration=None):
        self.calibration = calibration
