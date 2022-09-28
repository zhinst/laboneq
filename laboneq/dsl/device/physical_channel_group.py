# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Dict

import laboneq.core.path as qct_path
from laboneq.dsl.calibration import Calibratable
from laboneq.dsl.device.io_units.physical_channel import PhysicalChannel


@dataclass
class PhysicalChannelGroup:
    #: Unique identifier.
    uid: str = field(default=None)

    #: The physical channels in this group.
    channels: Dict[str, PhysicalChannel] = field(default_factory=dict)

    def get_calibration(self):
        """Retrieve the calibration of the physical channel group."""
        calibration = dict()
        for channel in self.channels.values():
            calibration[channel.path] = (
                channel.calibration if channel.is_calibrated() else None
            )

        return calibration

    def reset_calibration(self):
        """Reset the calibration on all the logical signals of the group."""
        for channel in self.channels.values():
            assert isinstance(channel, Calibratable)
            channel.reset_calibration()

    def list_calibratables(self, parent_path: str = None):
        current_path = (
            qct_path.concat(parent_path, self.uid)
            if parent_path is not None
            else self.uid
        )
        calibratables = dict()
        for signal in self.channels.values():
            if isinstance(signal, Calibratable):
                calibratables[
                    qct_path.concat(current_path, signal.uid)
                ] = signal.create_info()
        return calibratables

    @property
    def path(self):
        return qct_path.PhysicalChannelGroups_Path_Abs + qct_path.Separator + self.uid
