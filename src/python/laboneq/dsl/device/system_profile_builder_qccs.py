# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""System profile builder for Gen2/QCCS hardware."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from zhinst.core import __version__ as zhinst_version

from laboneq.dsl.device.instruments import (
    HDAWG,
    PQSC,
    QHUB,
    SHFPPC,
    SHFQA,
    SHFQC,
    SHFSG,
    UHFQA,
)
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument
from laboneq.dsl.device.system_profile_builder import register_profile_builder

if TYPE_CHECKING:
    from laboneq.controller import Controller
    from laboneq.controller.devices.device_zi import DeviceZI
    from laboneq.dsl.device import DeviceSetup, Instrument
    from laboneq.dsl.device.system_profile_qccs import SystemProfile

_DEVICE_DEMO_DATA: dict[type, tuple[str, list[str]]] = {
    HDAWG: ("HDAWG8", ["CNT", "FF", "ME", "MF", "PC", "SKW"]),
    PQSC: ("PQSC", []),
    QHUB: ("QHUB", []),
    SHFPPC: ("SHFPPC4", []),
    SHFQA: ("SHFQA4", ["16W", "LRT"]),
    SHFSG: ("SHFSG2", ["PLUS"]),
    UHFQA: ("UHFQA", ["AWG", "DIG", "QA"]),
}

_SessionClass = TypeVar("_SessionClass")


def _build_from_devices(
    device_setup: DeviceSetup,
    controller: Controller[_SessionClass] | None,
    *,
    demo: bool = False,
) -> SystemProfile:
    """Build QCCS system profile from connected devices.

    Args:
        device_setup: Device setup to build profile for
        controller: Controller for hardware queries; required if demo is False
        demo: If True, build a demo profile without querying hardware

    Returns:
        System profile with device capabilities
    """
    from laboneq.controller.devices.device_zi import DeviceBase
    from laboneq.dsl.device.system_profile_qccs import (
        DeviceCapabilitiesQCCS,
        SystemProfileQCCS,
    )

    server = next(iter(device_setup.servers.values()))
    profile = SystemProfileQCCS(
        setup_uid=device_setup.uid,
        server_address=server.host,
        server_port=int(server.port),
    )
    if not device_setup.instruments:
        # Happens during some tests
        return profile

    if demo:
        profile.server_version = zhinst_version
        instrument: Instrument
        for instrument in device_setup.instruments:
            if not isinstance(instrument, ZIStandardInstrument):
                continue
            assert instrument.address
            address = instrument.address.upper()
            if isinstance(instrument, SHFQC):
                profile.devices[f"{address}_QA"] = DeviceCapabilitiesQCCS(
                    device_model="SHFQC",
                    device_options=["16W", "LRT"],
                )
                profile.devices[f"{address}_SG"] = DeviceCapabilitiesQCCS(
                    device_model="SHFQC",
                    device_options=["QC6CH"],
                )
            else:
                model, options = _DEVICE_DEMO_DATA[type(instrument)]
                profile.devices[address] = DeviceCapabilitiesQCCS(
                    device_model=model,
                    device_options=options,
                )
        return profile

    assert controller is not None
    devices = controller.devices

    device: DeviceZI
    for device in devices.values():
        if not isinstance(device, DeviceBase):
            continue
        assert device.server_qualifier.host == profile.server_address
        assert device.server_qualifier.port == profile.server_port
        if not profile.server_version:
            profile.server_version = str(device.setup_caps.server_version or "")
        else:
            assert profile.server_version == str(device.setup_caps.server_version)
        device_key = device.serial
        if device.options.is_qc:
            if isinstance(device, SHFQA):
                device_key += f"{device.serial}_QA"
            else:
                device_key += f"{device.serial}_SG"

        assert device.serial not in profile.devices
        profile.devices[device.serial.upper()] = DeviceCapabilitiesQCCS(
            device_model=device.options.dev_type,
            device_options=device.dev_opts,
        )

    return profile


register_profile_builder("QCCS", _build_from_devices)
