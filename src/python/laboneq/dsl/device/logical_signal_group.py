# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs

import laboneq.core.path as qct_path
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration.calibratable import Calibratable
from laboneq.dsl.device.io_units import LogicalSignal


@classformatter
@attrs.define
class LogicalSignalGroup:
    """Group of logical signals. This could be used as a qubit representation with
    multiple logical signals driving a single qubit.
    """

    uid: str = attrs.field(default=None)
    logical_signals: dict[str, LogicalSignal] = attrs.field(factory=dict)

    def get_calibration(self):
        """Retrieve the calibration of the logical signal group."""
        calibration = dict()
        for logical_signal in self.logical_signals.values():
            calibration[self.path + qct_path.Separator + logical_signal.name] = (
                logical_signal.calibration if logical_signal.is_calibrated() else None
            )

        return calibration

    def reset_calibration(self):
        """Reset the calibration on all the logical signals of the group."""
        for logical_signal in self.logical_signals.values():
            assert isinstance(logical_signal, Calibratable)
            logical_signal.reset_calibration()

    def list_calibratables(self):
        return {
            signal.path: signal.create_info()
            for signal in self.logical_signals.values()
            if isinstance(signal, Calibratable)
        }

    @property
    def path(self):
        return qct_path.LogicalSignalGroups_Path_Abs + qct_path.Separator + self.uid
