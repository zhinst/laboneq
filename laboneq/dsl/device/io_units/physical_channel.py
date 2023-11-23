# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
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
    "amplitude",
    "added_outputs",
)


@classformatter
@dataclass(init=False, repr=False, order=True)
class PhysicalChannel(Calibratable):
    #: Unique identifier. Typically of the form
    # ``<device uid>/<channel name>``.
    uid: str

    #: The name of the channel, == <channel name>.
    # Computed from the HW channel ids like:
    # [SIGOUTS/0, SIGOUTS/1] -> "sigouts_0_1"
    # [SIGOUTS/2] -> "sigouts_2"
    name: str | None

    #: The type of the channel.
    type: Optional[PhysicalChannelType]

    #: Logical path to the channel. Typically of the form
    # ``/<device uid>/<channel name>``.
    path: str | None
    _calibration: Optional[SignalCalibration]

    def __init__(
        self,
        uid,
        name: str | None = None,
        type: PhysicalChannelType | None = None,
        path: str | None = None,
        calibration: SignalCalibration | None = None,
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
        field_values = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "_calibration":
                field_values.append(f"calibration={value!r}")
            else:
                field_values.append(f"{field.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(field_values)})"

    def __rich_repr__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "_calibration":
                yield "calibration", value
            else:
                yield f"{field.name}", value

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
