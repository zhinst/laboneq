# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Session integration helpers for system profiles.

This module provides helper functions for loading, resolving, and updating
system profiles during session lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from laboneq.dsl.device import SystemProfile
from laboneq.dsl.device.instruments import PRETTYPRINTERDEVICE
from laboneq.dsl.device.system_profile_builder import build_profile

if TYPE_CHECKING:
    from laboneq.controller import Controller
    from laboneq.dsl.device.device_setup import DeviceSetup


_logger = logging.getLogger(__name__)


def resolve_system_profile(
    system_profile: SystemProfile | Path | str | None,
) -> SystemProfile | None:
    """Resolve system_profile parameter to a SystemProfile object.

    Args:
        system_profile: SystemProfile object, path to YAML file, or None

    Returns:
        Resolved SystemProfile object or None
    """
    from laboneq.serializers import load
    from laboneq.serializers.core import SerializerFormat

    if system_profile is None:
        return None
    if isinstance(system_profile, SystemProfile):
        return system_profile

    path = Path(system_profile) if isinstance(system_profile, str) else system_profile
    data = load(path, format=SerializerFormat.YAML)
    assert isinstance(data, SystemProfile)
    return data


_SessionClass = TypeVar("_SessionClass")


def update_system_profile(
    device_setup: DeviceSetup,
    controller: Controller[_SessionClass],
) -> SystemProfile | None:
    """Query hardware and update cached system profile.

    Args:
        controller: Connected controller with devices
        device_setup: Device setup to use for profile generation

    Returns:
        Generated system profile or None on failure
    """
    from laboneq.core import system_profile_cache

    if not device_setup.instruments:
        # Happens during some tests
        return None

    if len(device_setup.instruments) == 1 and isinstance(
        device_setup.instruments[0], PRETTYPRINTERDEVICE
    ):
        system_type = "PrettyPrinter"
    else:
        system_type = "QCCS"

    try:
        profile = build_profile(
            system_type=system_type,
            device_setup=device_setup,
            controller=controller,
        )

        cache_path = system_profile_cache.save(profile)
        _logger.info("System profile saved to %s", cache_path)
        return profile
    except Exception as e:
        raise RuntimeError(f"Failed to update system profile: {e}") from e
