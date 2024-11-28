# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from laboneq.dsl.calibration.signal_calibration import SignalCalibration


class Calibratable(ABC):
    """Abstract base class for objects that can have calibration
    attached.

    Attributes:
        calibration (Optional[SignalCalibration]):
            The calibration for this object. Implementations
            of [Calibratable][laboneq.dsl.calibration.calibratable.Calibratable]
            should document and set this attribute.
    """

    calibration: Optional[SignalCalibration]

    @abstractmethod
    def is_calibrated(self) -> bool:
        """Return True if calibration has been set on this object.

        Implementations of [Calibratable][laboneq.dsl.calibration.calibratable]
        should document and implement this method.
        """

    def create_info(self) -> dict:
        """Return the calibration type and status in a dictionary.

        Used for introspection and debugging.
        """
        return {
            "type": f"{SignalCalibration.__module__}.{SignalCalibration.__qualname__}",
            "is_calibrated": self.is_calibrated(),
        }
