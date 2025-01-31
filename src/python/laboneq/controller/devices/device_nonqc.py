# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.devices.device_zi import DeviceBase


class DeviceNonQC(DeviceBase):
    def is_leader(self):
        return False

    def is_follower(self):
        return False

    def is_standalone(self):
        return False

    async def reset_to_idle(self):
        pass
