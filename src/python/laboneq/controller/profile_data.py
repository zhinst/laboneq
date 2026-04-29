# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Extract hardware data from a connected controller for system profile building.

The profile builders (``system_profile_builder_*.py``) are pure data
transformations that never touch the controller.  This module bridges the
gap: it reads device-level attributes and packages them as plain dicts
that the session layer forwards to the builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from laboneq.controller.devices.device_zi import DeviceBase

if TYPE_CHECKING:
    from laboneq.controller import Controller
    from laboneq.dsl.device.device_setup import DeviceSetup

# device_class 0x0 = QCCS (default), 0x1 = ZQCS
_ZQCS_DEVICE_CLASS = 0x1


def extract_profile_data(
    device_setup: DeviceSetup,
    controller: Controller[Any],
) -> tuple[str, dict[str, Any]]:
    """Determine the system type and extract the matching hardware data.

    Returns ``(system_type, hw_kwargs)`` where *hw_kwargs* can be
    unpacked directly into :func:`build_profile`.
    """
    if (
        len(device_setup.instruments) == 1
        and getattr(device_setup.instruments[0], "device_class", 0)
        == _ZQCS_DEVICE_CLASS
    ):
        return "ZQCS", _extract_zqcs_profile_data(device_setup, controller)
    return "QCCS", _extract_qccs_profile_data(controller)


def _extract_zqcs_profile_data(
    device_setup: DeviceSetup,
    controller: Controller[Any],
) -> dict[str, Any]:
    """Extract ZQCS hardware data from a connected controller.

    Returns a dict with ``scm_version`` and ``setup_description``,
    suitable for unpacking into :func:`build_profile`.
    """
    [device] = device_setup.instruments
    controller_device = controller.devices[device.uid]
    return {
        "scm_version": getattr(controller_device, "scm_version", None),
        "setup_description": getattr(controller_device, "setup_description", None),
    }


def _extract_qccs_profile_data(
    controller: Controller[Any],
) -> dict[str, Any]:
    """Extract QCCS device capabilities from a connected controller.

    Returns a dict with ``server_version`` and ``device_capabilities``,
    suitable for unpacking into :func:`build_profile`.
    """
    server_version: str | None = None
    device_capabilities: dict[str, dict[str, Any]] = {}

    for device in controller.devices.values():
        if not isinstance(device, DeviceBase):
            continue
        if server_version is None:
            server_version = str(device.setup_caps.server_version or "")
        device_capabilities[device.serial.upper()] = {
            "device_model": device.options.dev_type,
            "device_options": device.dev_opts,
        }

    return {
        "server_version": server_version,
        "device_capabilities": device_capabilities,
    }
