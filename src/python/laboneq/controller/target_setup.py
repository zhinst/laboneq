# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Per-backend construction of the controller's `TargetSetup`.

A backend owns the mapping from a DSL `DeviceSetup` to the instruments the
controller drives, and registers a builder for it under its backend name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq._backend_plugin import resolve_backend

if TYPE_CHECKING:
    from collections.abc import Callable

    from laboneq.data.execution_payload import TargetSetup
    from laboneq.dsl.device.device_setup import DeviceSetup

    TargetSetupBuilder = Callable[[DeviceSetup], TargetSetup]

_registered_builders: dict[str, TargetSetupBuilder] = {}


def register_target_setup_builder(backend: str, builder: TargetSetupBuilder) -> None:
    """Register `builder` as the way to build `backend`'s target setup."""
    registered = _registered_builders.get(backend)
    if registered is not None and registered is not builder:
        raise ValueError(
            f"Backend {backend!r} already builds its target setup with "
            f"{registered.__module__}.{registered.__qualname__}."
        )
    _registered_builders[backend] = builder


def build_target_setup(device_setup: DeviceSetup) -> TargetSetup:
    """The controller's view of `device_setup`, built by its backend."""
    backend = resolve_backend(device_setup)
    builder = _registered_builders.get(backend)
    if builder is None:
        installed = ", ".join(sorted(_registered_builders)) or "none"
        raise ValueError(
            f"No target setup builder registered for backend {backend!r}. "
            f"Installed backends: {installed}."
        )
    return builder(device_setup)
