# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""System profile builder facade with plugin architecture.

The main SystemProfileBuilder delegates to registered builder plugins.
Builder plugins register themselves when their modules are imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from laboneq.controller import Controller

if TYPE_CHECKING:
    from typing import Any, Callable

    from laboneq.dsl.device import DeviceSetup, SystemProfile

# Builder plugin registry
_builder_registry: dict[str, Callable[..., SystemProfile]] = {}


def register_profile_builder(
    system_type: str, builder_func: Callable[..., SystemProfile]
) -> None:
    """Register a system profile builder function.

    Args:
        system_type: Unique name for this builder (e.g., "QCCS")
        builder_func: The builder function to register
    """
    if system_type in _builder_registry:
        raise ValueError(
            f"Internal error: Builder for system type '{system_type}' already registered."
        )
    _builder_registry[system_type] = builder_func


_SessionClass = TypeVar("_SessionClass")


def build_profile(
    system_type: str,
    device_setup: DeviceSetup,
    controller: Controller[_SessionClass] | None,
    *args: Any,
    **kwargs: Any,
) -> SystemProfile:
    """Build a system profile using a registered builder.

    Args:
        system_type: The type of system profile to build (e.g., "QCCS")
        device_setup: The device setup to build the profile for
        controller: The controller for hardware queries, or None for demo profiles
        args: Additional positional arguments for the builder
        kwargs: Additional keyword arguments for the builder
    Returns:
        The built system profile
    Raises:
        ValueError: If the builder is not registered
    """
    builder = _builder_registry.get(system_type)
    if builder is None:
        raise ValueError(
            f"Internal error: System profile builder for system type '{system_type}' not registered."
        )
    return builder(device_setup, controller, *args, **kwargs)


def build_demo_profile(
    system_type: str, device_setup: DeviceSetup, *args: Any, **kwargs: Any
) -> SystemProfile:
    """Build a demo system profile using a registered builder.

    The demo profile is meant for experimenting with LabOne Q without having access to real
    hardware from which the actual data could be obtained. It returns realistic data for a
    to demonstrate typical features of a setup.

    Args:
        system_type: The type of system profile to build (e.g., "QCCS")
        device_setup: The device setup to build the profile for
        args: Additional positional arguments for the builder
        kwargs: Additional keyword arguments for the builder
    Returns:
        The built system profile
    Raises:
        ValueError: If the builder is not registered
    """
    return build_profile(system_type, device_setup, None, *args, demo=True, **kwargs)
