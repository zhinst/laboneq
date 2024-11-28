# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import asyncio
from laboneq.controller.communication import DaqNodeSetAction, batch_set
from laboneq.controller.devices.device_leader_base import DeviceLeaderBase
from laboneq.controller.devices.device_utils import NodeCollector


class DeviceQHUB(DeviceLeaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "QHUB"
        self.dev_opts = []

    async def qhub_reset_zsync_phy(self):
        async def _set_debug_sequential(values: list[int]):
            nc = NodeCollector(base=f"/{self.serial}/raw/debug/0/")
            for v in values:
                nc.add("value", v)
                nc.barrier()
            await batch_set(await self.maybe_async(nc))

        await asyncio.sleep(2)
        await _set_debug_sequential(
            [
                13,  # assert PHY_RST
                # 12, # de-assert EN_VTC
                11,  # assert EN_VTC
                14,  # de-assert PHY_RST
                13,  # assert PHY_RST
                14,  # de-assert PHY_RST
                13,  # assert PHY_RST
                14,  # de-assert PHY_RST
            ]
        )
        await asyncio.sleep(0.1)
        await _set_debug_sequential(
            [
                13,  # assert PHY_RST
            ]
        )
        await asyncio.sleep(0.1)
        await _set_debug_sequential(
            [
                14,  # de-assert PHY_RST
            ]
        )
        await asyncio.sleep(2)

    async def collect_reset_nodes(self) -> list[DaqNodeSetAction]:
        reset_nodes = await super().collect_reset_nodes()
        nc = NodeCollector(base=f"/{self.serial}/")
        # QHub does not automatically transition execution/enable to 0 (stop),
        # ensure it is on stop before we begin execution.
        nc.add("execution/enable", 0, cache=False)
        reset_nodes.extend(await self.maybe_async(nc))
        return reset_nodes
