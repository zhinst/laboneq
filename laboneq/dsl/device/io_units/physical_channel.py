# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from laboneq.dsl.calibration import Calibratable, SignalCalibration
from laboneq.dsl.calibration.observable import Signal


class PhysicalChannelType(Enum):
    IQ_CHANNEL = "iq_channel"
    RF_CHANNEL = "rf_channel"


PHYSICAL_CHANNEL_CALIBRATION_FIELDS = (
    "local_oscillator",
    "port_delay",
    "port_mode",
    "range",
    "voltage_offset",
    "mixer_calibration",
    "precompensation",
)


@dataclass(init=False, repr=False, order=True)
class PhysicalChannel(Calibratable):
    #: Unique identifier.
    uid: str

    #: The name of the channel.
    name: Optional[str]

    #: The type of the channel.
    type: Optional[PhysicalChannelType]

    #: Logical path to the channel. Typically of the form
    # ``/<device name>/<channel name>``.
    path: Optional[str]
    _calibration: Optional[SignalCalibration]

    def __init__(
        self,
        uid,
        name: str = None,
        type: PhysicalChannelType = None,
        path: str = None,
        calibration: SignalCalibration = None,
    ):
        self.uid = uid
        self.name = name
        self.type = type
        self.path = path
        self._calibration = calibration
        if self._calibration is not None:
            self._calibration.has_changed().connect(self._on_calibration_changed)
        self._signal_calibration_changed = Signal(self)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(uid={self.uid}, type={self.type}, "
            f"name={self.name}, path={self.path}, calibration={self.calibration})"
        )

    def __hash__(self):
        # By default, dataclass does not generate a __hash__() method for
        # PhysicalChannel, because it is mutable, and thus cannot safely be used as a
        # key in a dict (the hash might change while it is stored). Assuming that both
        # the uid and the path are indeed unique and permanent (at least among those
        # instances used as keys in a dict), we can use this implementation safely.
        return hash((self.uid, self.path))

    def is_calibrated(self):
        return self.calibration is not None

    def reset_calibration(self):
        self.calibration = None

    @property
    def calibration(self) -> SignalCalibration:
        return self._calibration

    @calibration.setter
    def calibration(self, new_calib: Optional[SignalCalibration]):
        if new_calib == self._calibration:
            return

        if self._calibration is not None:
            self._calibration.has_changed().disconnect(self._on_calibration_changed)
        self._calibration = new_calib
        if self._calibration is not None:
            self._calibration.has_changed().connect(self._on_calibration_changed)

        if new_calib is not None:
            for key in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
                new_calib_attribute = getattr(self.calibration, key)
                self._on_calibration_changed(self.calibration, key, new_calib_attribute)
        else:
            # signal that the entire calibration has been deleted
            self._signal_calibration_changed.fire("calibration", None)

    def _on_calibration_changed(self, _: SignalCalibration, field: str, value):
        if field in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
            self._signal_calibration_changed.fire(field, value)

    def calibration_has_changed(self):
        return self._signal_calibration_changed
