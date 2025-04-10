# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
from enum import Enum
from typing import Optional

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibratable, SignalCalibration
from laboneq.dsl.calibration.physical_channel_calibration import (
    PhysicalChannelCalibration,
)


PHYSICAL_CHANNEL_CALIBRATION_FIELDS = tuple(
    attrs.fields_dict(PhysicalChannelCalibration).keys()
)


class PhysicalChannelType(Enum):
    IQ_CHANNEL = "iq_channel"
    RF_CHANNEL = "rf_channel"


def _physical_calibration_to_signal_calibration(
    value: PhysicalChannelCalibration | SignalCalibration | None,
) -> SignalCalibration | None:
    if isinstance(value, SignalCalibration):
        return value
    elif value is None:
        return None

    cal = SignalCalibration(
        **attrs.asdict(
            value,
            recurse=False,
        )
    )
    return cal


@classformatter
@attrs.define(repr=False, slots=False)
class PhysicalChannel(Calibratable):
    #: Unique identifier. Typically of the form
    # ``<device uid>/<channel name>``.
    uid: str

    #: The name of the channel, == <channel name>.
    # Computed from the HW channel ids like:
    # [SIGOUTS/0, SIGOUTS/1] -> "sigouts_0_1"
    # [SIGOUTS/2] -> "sigouts_2"
    name: str | None = None

    #: The type of the channel.
    type: Optional[PhysicalChannelType] = None

    #: Logical path to the channel. Typically of the form
    # ``/<device uid>/<channel name>``.
    path: str | None = None
    # TODO: Make the calibration type `PhysicalChannelCalibration`
    calibration: SignalCalibration | None = attrs.field(
        default=None,
        converter=_physical_calibration_to_signal_calibration,
        on_setattr=lambda self, attr, value: self._set_calibration(attr, value),
    )

    def _set_calibration(self, attr, value):
        return value

    def __repr__(self):
        field_values = []
        for field in attrs.fields(PhysicalChannel):
            value = getattr(self, field.name)
            if field.name == "calibration":
                field_values.append(f"calibration={value!r}")
            else:
                field_values.append(f"{field.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(field_values)})"

    def __rich_repr__(self):
        for field in attrs.fields(PhysicalChannel):
            value = getattr(self, field.name)
            if field.name == "calibration":
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
