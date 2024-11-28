# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.data.calibration import Calibration, SignalCalibration
from laboneq.data.path import Separator
from laboneq.data.setup_description import LogicalSignal


class CalibrationHelper:
    """A helper class for Calibration."""

    def __init__(self, calibration: Calibration):
        if calibration is not None and calibration.items:
            self._items = calibration.items
        else:
            self._items = {}

    def empty(self) -> bool:
        """Check whether the calibration is empty."""
        if not self._items:
            return True
        return False

    def by_logical_signal(
        self, logical_signal: LogicalSignal
    ) -> SignalCalibration | None:
        """Return `SignalCalibration` object for the given `logical_signal`."""
        signal = Separator.join((logical_signal.group, logical_signal.name))
        return self._items.get(signal, None)
