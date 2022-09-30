# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

from laboneq.dsl.calibration.signal_calibration import SignalCalibration


class Calibratable(ABC):
    calibration: Optional[SignalCalibration]

    @abstractmethod
    def is_calibrated(self):
        pass

    def create_info(self):
        return {
            "type": f"{SignalCalibration.__module__}.{SignalCalibration.__qualname__}",
            "is_calibrated": self.is_calibrated(),
        }
