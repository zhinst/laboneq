# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.util import LabOneQControllerException

from laboneq.controller.devices.device_base import DeviceBase, DeviceQualifier
from laboneq.controller.devices.device_hdawg import DeviceHDAWG
from laboneq.controller.devices.device_pqsc import DevicePQSC
from laboneq.controller.devices.device_uhfqa import DeviceUHFQA
from laboneq.controller.devices.device_shfqa import DeviceSHFQA
from laboneq.controller.devices.device_shfsg import DeviceSHFSG


class DeviceFactory:
    @staticmethod
    def create(device_qualifier: DeviceQualifier) -> DeviceBase:
        if device_qualifier.driver == "HDAWG":
            return DeviceHDAWG(device_qualifier)
        elif device_qualifier.driver == "UHFQA":
            return DeviceUHFQA(device_qualifier)
        elif device_qualifier.driver == "SHFQA":
            return DeviceSHFQA(device_qualifier)
        elif device_qualifier.driver == "SHFSG":
            return DeviceSHFSG(device_qualifier)
        elif device_qualifier.driver == "PQSC":
            return DevicePQSC(device_qualifier)
        else:
            raise LabOneQControllerException(
                f"Unknown device driver {device_qualifier.driver}"
            )
