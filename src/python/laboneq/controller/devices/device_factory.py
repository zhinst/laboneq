# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.devices.device_setup_dao import ServerQualifier, DeviceQualifier
from laboneq.controller.devices.device_hdawg import DeviceHDAWG
from laboneq.controller.devices.device_nonqc import DeviceNonQC
from laboneq.controller.devices.device_pqsc import DevicePQSC
from laboneq.controller.devices.device_qhub import DeviceQHUB
from laboneq.controller.devices.device_shfppc import DeviceSHFPPC
from laboneq.controller.devices.device_shfqa import DeviceSHFQA
from laboneq.controller.devices.device_shfsg import DeviceSHFSG
from laboneq.controller.devices.device_uhfqa import DeviceUHFQA
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.versioning import SetupCaps


class DeviceFactory:
    _registered_devices: dict[str, type[DeviceZI]] = {}

    @classmethod
    def register_device(cls, driver: str, dev_class: type[DeviceZI]):
        assert driver not in cls._registered_devices
        cls._registered_devices[driver] = dev_class

    @classmethod
    def create(
        cls,
        server_qualifier: ServerQualifier,
        device_qualifier: DeviceQualifier,
        setup_caps: SetupCaps,
    ) -> DeviceZI:
        dev_class = cls._registered_devices.get(device_qualifier.driver.upper())
        if dev_class is None:
            raise LabOneQControllerException(
                f"Unknown device driver {device_qualifier.driver}"
            )
        return dev_class(server_qualifier, device_qualifier, setup_caps)


DeviceFactory.register_device("HDAWG", DeviceHDAWG)
DeviceFactory.register_device("UHFQA", DeviceUHFQA)
DeviceFactory.register_device("SHFQA", DeviceSHFQA)
DeviceFactory.register_device("SHFSG", DeviceSHFSG)
DeviceFactory.register_device("SHFPPC", DeviceSHFPPC)
DeviceFactory.register_device("PQSC", DevicePQSC)
DeviceFactory.register_device("QHUB", DeviceQHUB)
DeviceFactory.register_device("NONQC", DeviceNonQC)
