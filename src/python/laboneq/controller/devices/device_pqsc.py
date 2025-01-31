# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.devices.device_leader_base import DeviceLeaderBase
from laboneq.controller.devices.node_control import (
    Command,
    WaitCondition,
    NodeControlBase,
    Response,
)


class DevicePQSC(DeviceLeaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "PQSC"
        self.dev_opts = []

    def load_factory_preset_control_nodes(self) -> list[NodeControlBase]:
        return [
            Command(f"/{self.serial}/system/preset/load", 1),
            Response(f"/{self.serial}/system/preset/busy", 0),
            # TODO(2K): Remove once https://zhinst.atlassian.net/browse/HULK-1800 is resolved
            WaitCondition(f"/{self.serial}/system/clocks/referenceclock/in/source", 0),
            WaitCondition(f"/{self.serial}/system/clocks/referenceclock/in/status", 0),
        ]
