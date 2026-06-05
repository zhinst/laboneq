# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.devices.device_leader_base import DeviceLeaderBase
from laboneq.controller.devices.device_utils import NodeCollector


class DeviceQHUB(DeviceLeaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "QHUB"
        self.dev_opts = []
        self._zsyncs = 56

    async def reset_to_idle(self):
        await super().reset_to_idle()
        nc = NodeCollector(base=f"/{self.serial}/")
        # QHub does not automatically transition execution/enable to 0 (stop),
        # ensure it is on stop before we begin execution.
        nc.add("execution/enable", 0, cache=False)
        await self.set_async(nc)
