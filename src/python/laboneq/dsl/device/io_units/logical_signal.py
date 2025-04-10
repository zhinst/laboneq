# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
from typing import Optional, Union

from laboneq.core.exceptions.laboneq_exception import LabOneQException
from laboneq.core.types.enums import IODirection
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.signal_calibration import (
    SignalCalibration,
)
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.calibratable import Calibratable
from laboneq.dsl.device.io_units.physical_channel import (
    PhysicalChannel,
)


@classformatter
@attrs.define(repr=False, slots=False)
class LogicalSignal(Calibratable):
    uid: str
    direction: Optional[IODirection] = None
    name: str | None = None
    calibration: SignalCalibration | None = attrs.field(
        default=None,
        on_setattr=lambda self, attr, value: self._set_calibration(attr, value),
    )
    path: str | None = None
    _physical_channel: Optional[PhysicalChannel] = None

    def _set_calibration(self, attr, value):
        return value

    def __hash__(self):
        # By default, dataclass does not generate a __hash__() method for LogicalSignal,
        # because it is mutable, and thus cannot safely be used as a key in a dict (the
        # hash might change while it is stored). Assuming that both the uid and the path
        # are indeed unique and permanent (at least among those instances used as keys
        # in a dict), we can use this implementation safely.
        return hash((self.uid, self.path))

    def __repr__(self):
        field_values = []
        for field in attrs.fields(LogicalSignal):
            value = getattr(self, field.name)
            if field.name == "_calibration":
                field_values.append(f"calibration={value!r}")
            elif field.name == "_physical_channel":
                field_values.append(f"physical_channel={value!r}")
            else:
                field_values.append(f"{field.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(field_values)})"

    def __rich_repr__(self):
        for field in attrs.fields(LogicalSignal):
            value = getattr(self, field.name)
            if field.name == "_calibration":
                yield "calibration", value
            elif field.name == "_physical_channel":
                yield "physical_channel", value
            else:
                yield f"{field.name}", value

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
        if self.is_calibrated() and self.calibration.mixer_calibration is not None:
            return self.calibration.mixer_calibration.voltage_offsets
        return None

    @voltage_offsets.setter
    def voltage_offsets(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(
                mixer_calibration=MixerCalibration(voltage_offsets=value)
            )
        else:
            self.calibration.mixer_calibration.voltage_offsets = value

    @property
    def correction_matrix(self):
        if self.is_calibrated() and self.calibration.mixer_calibration is not None:
            return self.calibration.mixer_calibration.correction_matrix
        return None

    @correction_matrix.setter
    def correction_matrix(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(
                mixer_calibration=MixerCalibration(correction_matrix=value)
            )
        else:
            self.calibration.mixer_calibration.correction_matrix = value

    @property
    def local_oscillator(self):
        return self.calibration.local_oscillator if self.is_calibrated() else None

    @local_oscillator.setter
    def local_oscillator(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(local_oscillator=value)
        else:
            self.calibration.local_oscillator = value

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
        if not self.is_calibrated():
            self.calibration = SignalCalibration(port_mode=value)
        else:
            self.calibration.port_mode = value

    @property
    def threshold(self):
        return self.calibration.threshold if self.is_calibrated() else None

    @threshold.setter
    def threshold(self, value: float | list[float]):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(threshold=value)
        else:
            self.calibration.threshold = value

    @property
    def amplifier_pump(self) -> AmplifierPump | None:
        return self.calibration.amplifier_pump if self.is_calibrated() else None

    @amplifier_pump.setter
    def amplifier_pump(self, value: AmplifierPump):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(amplifier_pump=value)
        else:
            self.calibration.amplifier_pump = value

    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def reset_calibration(self):
        self.calibration = None

    @property
    def physical_channel(self):
        return self._physical_channel


LogicalSignalRef = Union[LogicalSignal, str]


def resolve_logical_signal_ref(logical_signal_ref: LogicalSignalRef) -> str:
    if logical_signal_ref is None:
        return None
    if isinstance(logical_signal_ref, str):
        return logical_signal_ref
    if (
        not isinstance(logical_signal_ref, LogicalSignal)
        or logical_signal_ref.path is None
    ):
        raise LabOneQException(
            "Invalid LogicalSignal: Seems like the logical signal is not part of a qubit setup. "
            "Make sure the object is retrieved from a device setup."
        )
    return logical_signal_ref.path
