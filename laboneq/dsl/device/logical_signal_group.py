# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

import laboneq.core.path as qct_path
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from ..calibration import Calibratable
from .io_units import LogicalSignal


@classformatter
@dataclass(init=True, repr=True, order=True)
class LogicalSignalGroup:
    """Group of logical signals. This could be used as a qubit representation with
    multiple logical signals driving a single qubit.
    """

    uid: str = field(default=None)
    logical_signals: Dict[str, LogicalSignal] = field(default_factory=dict)

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

    def list_calibratables(self, parent_path: str | None = None):
        current_path = (
            qct_path.concat(parent_path, self.uid)
            if parent_path is not None
            else self.uid
        )
        calibratables = dict()
        for signal in self.logical_signals.values():
            if isinstance(signal, Calibratable):
                calibratables[
                    qct_path.concat(current_path, signal.uid)
                ] = signal.create_info()
        return calibratables

    @property
    def path(self):
        return qct_path.LogicalSignalGroups_Path_Abs + qct_path.Separator + self.uid
