# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.communication import DaqNodeAction
from laboneq.controller.devices.device_zi import DeviceZI


class DeviceNonQC(DeviceZI):
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
