# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""System description builder for Gen2/QCCS hardware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from laboneq.dsl.device.system_description_builder import register_description_builder

if TYPE_CHECKING:
    from laboneq.dsl.device import DeviceSetup, Instrument
    from laboneq.dsl.device.system_description_qccs import (
        DeviceCapabilitiesQCCS,
        SystemDescription,
    )

_DEVICE_DEMO_DATA: dict[type, tuple[str, list[str]]] = {
    HDAWG: ("HDAWG8", ["CNT", "FF", "ME", "MF", "PC", "SKW"]),
    PQSC: ("PQSC", []),
    QHUB: ("QHUB", []),
    SHFPPC: ("SHFPPC4", []),
    SHFQA: ("SHFQA4", ["16W", "LRT"]),
    SHFSG: ("SHFSG2", ["PLUS"]),
    UHFQA: ("UHFQA", ["AWG", "DIG", "QA"]),
}


def _build_from_devices(
    device_setup: DeviceSetup,
    *,
    demo: bool = False,
    server_version: str | None = None,
    device_capabilities: dict[str, dict[str, Any] | DeviceCapabilitiesQCCS]
    | None = None,
) -> SystemDescription:
    """Build QCCS system description from pre-extracted device capabilities.

    Args:
        device_setup: Device setup to build system description for.
        demo: If True, build a demo system description with synthetic data.
        server_version: LabOne server version string (required unless demo=True).
        device_capabilities: Pre-extracted device capabilities keyed by
            uppercased serial number. Values may be
            :class:`DeviceCapabilitiesQCCS` instances or plain dicts with
            ``device_model`` and ``device_options`` keys.

    Returns:
        System description with device capabilities.
    """
    from laboneq.dsl.device.system_description_qccs import (
        DeviceCapabilitiesQCCS,
        SystemDescriptionQCCS,
    )

    server = next(iter(device_setup.servers.values()))
    description = SystemDescriptionQCCS(
        setup_uid=device_setup.uid,
        server_address=server.host,
        server_port=int(server.port),
    )
    if not device_setup.instruments:
        # Happens during some tests
        return description

    if demo:
        description.server_version = zhinst_version
        instrument: Instrument
        for instrument in device_setup.instruments:
            if not isinstance(instrument, ZIStandardInstrument):
                continue
            assert instrument.address
            address = instrument.address.upper()
            if isinstance(instrument, SHFQC):
                description.devices[f"{address}_QA"] = DeviceCapabilitiesQCCS(
                    device_model="SHFQC",
                    device_options=["16W", "LRT"],
                )
                description.devices[f"{address}_SG"] = DeviceCapabilitiesQCCS(
                    device_model="SHFQC",
                    device_options=["QC6CH"],
                )
            else:
                model, options = _DEVICE_DEMO_DATA[type(instrument)]
                description.devices[address] = DeviceCapabilitiesQCCS(
                    device_model=model,
                    device_options=options,
                )
        return description

    if server_version:
        description.server_version = server_version
    if device_capabilities:
        for serial, caps in device_capabilities.items():
            if isinstance(caps, dict):
                caps = DeviceCapabilitiesQCCS(**caps)
            description.devices[serial] = caps

    return description


register_description_builder("QCCS", _build_from_devices)
