# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from laboneq.core.types.enums import IODirection
from laboneq.dsl.calibration import SignalCalibration, MixerCalibration
from laboneq.dsl.calibration.calibratable import Calibratable
from laboneq.dsl.device.io_units.physical_channel import (
    PhysicalChannel,
    PHYSICAL_CHANNEL_CALIBRATION_FIELDS,
)


@dataclass(init=False, repr=False, order=True)
class LogicalSignal(Calibratable):
    uid: str
    name: Optional[str]
    _calibration: Optional[SignalCalibration]
    _physical_channel: Optional[PhysicalChannel]
    path: Optional[str]
    direction: Optional[IODirection]

    def __init__(
        self,
        uid,
        direction=None,
        name=None,
        calibration=None,
        path=None,
        physical_channel: PhysicalChannel = None,
    ):
        super().__init__()
        self.uid = uid
        self.name = name
        self.direction = direction
        self._calibration = calibration
        self.path = path
        self._physical_channel = physical_channel
        if self.is_calibrated():
            self.calibration.has_changed().connect(self._on_calibration_changed)
        if physical_channel is not None:
            physical_channel.calibration_has_changed().connect(
                self._on_physical_channel_calibration_changed
            )

    def __hash__(self):
        # By default, dataclass does not generate a __hash__() method for LogicalSignal,
        # because it is mutable, and thus cannot safely be used as a key in a dict (the
        # hash might change while it is stored). Assuming that both the uid and the path
        # are indeed unique and permanent (at least among those instances used as keys
        # in a dict), we can use this implementation safely.
        return hash((self.uid, self.path))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"uid={repr(self.uid)}, "
            f"direction={repr(self.direction)}, "
            f"name={repr(self.name)}, "
            f"calibration={repr(self.calibration)}, "
            f"path={repr(self.path)}, "
            f"physical_channel={repr(self.physical_channel)}"
            f")"
        )

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
    def threshold(self, value: float):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(threshold=value)
        else:
            self.calibration.threshold = value

    @property
    def calibration(self) -> SignalCalibration:
        return self._calibration

    @calibration.setter
    def calibration(self, new_calib: Optional[SignalCalibration]):
        if new_calib == self._calibration:
            return

        # Ignore the physical channel properties that are None in the new object
        if self.is_calibrated() and new_calib is not None:
            for field in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
                if getattr(new_calib, field) is None:
                    old_value = getattr(self._calibration, field)
                    setattr(new_calib, field, old_value)
        elif self.is_calibrated() and new_calib is None:
            self.reset_calibration()
            return

        with self._suspend_callback():  # disconnect old callback, reconnect new one
            self._calibration = new_calib

        if self.is_calibrated():
            for key in self.calibration.observed_fields():
                new_calib_attribute = getattr(self.calibration, key)
                self._on_calibration_changed(self.calibration, key, new_calib_attribute)

    def _on_calibration_changed(self, _: SignalCalibration, key: str, value):
        """Called when the user modifies `LogicalSignal.calibration`"""
        if self.physical_channel is None:
            return

        pc = self.physical_channel

        if (
            key in PHYSICAL_CHANNEL_CALIBRATION_FIELDS
            and key != "mixer_calibration"
            and value is not None
        ):
            with self._suspend_callback():
                if pc.calibration is None:
                    pc.calibration = SignalCalibration(**{key: value})
                else:
                    setattr(pc.calibration, key, value)

        elif key == "mixer_calibration" and value is not None:
            with self._suspend_callback():
                if pc.calibration is None:
                    pc.calibration = SignalCalibration(mixer_calibration=value)
                else:
                    pc.calibration.mixer_calibration = value

    def _on_physical_channel_calibration_changed(
        self, _: PhysicalChannel, key: str, value
    ):
        """Called when the user modifies `LogicalSignal.physical_channel.calibration`"""
        if key == "calibration":  # Entire calibration object was replaced
            calibration = value
            with self._suspend_callback():
                if not self.is_calibrated():
                    self._calibration = SignalCalibration()
                for field in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
                    setattr(self._calibration, field, getattr(calibration, field, None))
        else:
            assert self.physical_channel.is_calibrated()
            with self._suspend_callback():
                if self._calibration is None:
                    self._calibration = SignalCalibration(**{key: value})
                else:
                    setattr(self._calibration, key, value)

    @contextmanager
    def _suspend_callback(self):
        if self.is_calibrated():
            self._calibration.has_changed().disconnect(self._on_calibration_changed)
        if self.physical_channel is not None:
            self.physical_channel.calibration_has_changed().disconnect(
                self._on_physical_channel_calibration_changed
            )

        yield

        # By now any of the observables might have changed so test (for None) again.
        if self.is_calibrated():
            self._calibration.has_changed().connect(self._on_calibration_changed)
        if self.physical_channel is not None:
            self.physical_channel.calibration_has_changed().connect(
                self._on_physical_channel_calibration_changed
            )

    def is_calibrated(self):
        return self._calibration is not None

    def reset_calibration(self):
        if not self.is_calibrated():
            return
        physical_channel_is_calibrated = any(
            getattr(self._calibration, field) is not None
            for field in PHYSICAL_CHANNEL_CALIBRATION_FIELDS
        )
        if physical_channel_is_calibrated:
            for field in self.calibration.observed_fields():
                if field in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
                    continue
                with self._suspend_callback():
                    setattr(self.calibration, field, None)

        if not physical_channel_is_calibrated:
            # calibration is empty, no need to hold on to it
            with self._suspend_callback():
                self._calibration = None

    @property
    def physical_channel(self):
        return self._physical_channel
