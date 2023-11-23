# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.communication import DaqNodeAction, DaqWrapper
from laboneq.controller.devices.device_zi import DeviceQualifier, DeviceZI


class DevicePRETTYPRINTER(DeviceZI):
    def is_leader(self):
        return False

    def is_follower(self):
        return False

    def is_standalone(self):
        return False

    def check_errors(self):
        pass

    def collect_reset_nodes(self) -> list[DaqNodeAction]:
        return []

    def __init__(self, device_qualifier: DeviceQualifier, daq: DaqWrapper):
        super().__init__(device_qualifier=device_qualifier, daq=daq)
        self._device_class = 0x1
