# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.communication import DaqWrapper
from laboneq.controller.devices.device_hdawg import DeviceHDAWG
from laboneq.controller.devices.device_nonqc import DeviceNonQC
from laboneq.controller.devices.device_pqsc import DevicePQSC
from laboneq.controller.devices.device_shfppc import DeviceSHFPPC
from laboneq.controller.devices.device_shfqa import DeviceSHFQA
from laboneq.controller.devices.device_shfsg import DeviceSHFSG
from laboneq.controller.devices.device_uhfqa import DeviceUHFQA
from laboneq.controller.devices.device_pretty_printer import DevicePRETTYPRINTER
from laboneq.controller.devices.device_zi import DeviceQualifier, DeviceZI
from laboneq.controller.util import LabOneQControllerException


class DeviceFactory:
    @staticmethod
    def create(device_qualifier: DeviceQualifier, daq: DaqWrapper) -> DeviceZI:
        dev_class = {
            "HDAWG": DeviceHDAWG,
            "UHFQA": DeviceUHFQA,
            "SHFQA": DeviceSHFQA,
            "SHFSG": DeviceSHFSG,
            "SHFPPC": DeviceSHFPPC,
            "PQSC": DevicePQSC,
            "NONQC": DeviceNonQC,
            "PRETTYPRINTERDEVICE": DevicePRETTYPRINTER,
        }.get(device_qualifier.driver.upper())
        if dev_class is None:
            raise LabOneQControllerException(
                f"Unknown device driver {device_qualifier.driver}"
            )
        return dev_class(device_qualifier, daq)
