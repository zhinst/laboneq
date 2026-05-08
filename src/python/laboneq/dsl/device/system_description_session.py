# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Session integration helpers for system descriptions.

This module provides helper functions for loading, resolving, and updating
system descriptions during session lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from laboneq.dsl.device import SystemDescription
from laboneq.dsl.device.system_description_builder import build_system_description

if TYPE_CHECKING:
    from laboneq.controller import Controller
    from laboneq.dsl.device.device_setup import DeviceSetup


_logger = logging.getLogger(__name__)


def resolve_system_description(
    system_description: SystemDescription | Path | str | None,
) -> SystemDescription | None:
    """Resolve system_description parameter to a SystemDescription object.

    Args:
        system_description: SystemDescription object, path to YAML file, or None

    Returns:
        Resolved SystemDescription object or None
    """
    from laboneq.serializers import load
    from laboneq.serializers.core import SerializerFormat

    if system_description is None:
        return None
    if isinstance(system_description, SystemDescription):
        return system_description

    path = (
        Path(system_description)
        if isinstance(system_description, str)
        else system_description
    )
    data = load(path, format=SerializerFormat.YAML)
    assert isinstance(data, SystemDescription)
    return data


_SessionClass = TypeVar("_SessionClass")


def update_system_description(
    device_setup: DeviceSetup,
    controller: Controller[_SessionClass],
) -> SystemDescription | None:
    """Query hardware and update cached system description.

    Extracts hardware data from the controller (via
    :mod:`laboneq.controller.description_data`) and delegates to the
    appropriate builder.

    Args:
        device_setup: Device setup to use for description generation.
        controller: Connected controller with devices.

    Returns:
        Generated system description or None on failure.
    """
    from laboneq.controller.system_description_data import extract_description_data
    from laboneq.core import system_description_cache

    if not device_setup.instruments:
        # Happens during some tests
        return None

    system_type, hw_data = extract_description_data(device_setup, controller)

    try:
        description = build_system_description(
            system_type=system_type,
            device_setup=device_setup,
            **hw_data,
        )

        cache_path = system_description_cache.save(description)
        _logger.info("System description saved to %s", cache_path)
        return description
    except Exception as e:
        raise RuntimeError(f"Failed to update system description: {e}") from e
