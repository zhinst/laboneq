# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""System description builder facade with plugin architecture.

The main SystemDescriptionBuilder delegates to registered builder plugins.
Builder plugins register themselves when their modules are imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable

    from laboneq.dsl.device import DeviceSetup, SystemDescription

# Builder plugin registry
_builder_registry: dict[str, Callable[..., SystemDescription]] = {}


def register_description_builder(
    system_type: str, builder_func: Callable[..., SystemDescription]
) -> None:
    """Register a system description builder function.

    Args:
        system_type: Unique name for this builder (e.g., "QCCS")
        builder_func: The builder function to register
    """
    if system_type in _builder_registry:
        raise ValueError(
            f"Internal error: Builder for system type '{system_type}' already registered."
        )
    _builder_registry[system_type] = builder_func


def build_system_description(
    system_type: str,
    device_setup: DeviceSetup,
    *args: Any,
    **kwargs: Any,
) -> SystemDescription:
    """Build a system description using a registered builder.

    Args:
        system_type: The type of system description to build (e.g., "QCCS")
        device_setup: The device setup to build the system description for
        args: Additional positional arguments for the builder
        kwargs: Additional keyword arguments for the builder (hardware data)
    Returns:
        The built system description
    Raises:
        ValueError: If the builder is not registered
    """
    builder = _builder_registry.get(system_type)
    if builder is None:
        raise ValueError(
            f"Internal error: System description builder for system type '{system_type}' not registered."
        )
    return builder(device_setup, *args, **kwargs)


def build_demo_system_description(
    system_type: str, device_setup: DeviceSetup, *args: Any, **kwargs: Any
) -> SystemDescription:
    """Build a demo system description using a registered builder.

    The demo system description is meant for experimenting with LabOne Q without having access to real
    hardware from which the actual data could be obtained. It returns realistic data for a
    to demonstrate typical features of a setup.

    Args:
        system_type: The type of system description to build (e.g., "QCCS")
        device_setup: The device setup to build the system description for
        args: Additional positional arguments for the builder
        kwargs: Additional keyword arguments for the builder
    Returns:
        The built system description
    Raises:
        ValueError: If the builder is not registered
    """
    return build_system_description(
        system_type, device_setup, *args, demo=True, **kwargs
    )
