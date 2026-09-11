# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Backend plugin system for LabOne Q.

Backends can register themselves via entry points in the "laboneq.backends" group.
Each backend must provide a class that implements the BackendPlugin interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.dsl.device import Instrument
    from laboneq.dsl.device.device_setup import DeviceSetup

#: The backend for an instrument that does not declare one.
DEFAULT_BACKEND = "qccs"


class BackendPlugin(ABC):
    """Base class for LabOne Q backend plugins.

    Backend plugins are registered via entry points and loaded automatically
    when laboneq is imported. Each backend must implement the register() method
    to set up its device types, compiler hooks, and other integrations.
    """

    @abstractmethod
    def register(self) -> None:
        """Register the backend with LabOne Q.

        This method is called once during laboneq import. It should:
        - Register device types with ``DeviceFactory.register_device()``
        - Register compiler hooks via ``register_compiler_hooks()``
        - Register the deserializer for the backend's own compiler output via
          ``laboneq.core.compiler_output.register_deserializer()``, under the backend name its
          compiler stamps into the compiled bytes
        - Register how the device setup is translated into a controller's payload
          via ``laboneq.controller.target_setup.register_target_setup_builder()``
        - Register logger namespaces via ``register_logger_namespace()``
        - Register instrumentors via ``LabOneQInstrumentor.register_instrumentor()``
        - Register serializers or other framework extensions
        - Register backends for signal plotting
        """
        ...


def resolve_backend(device_setup: DeviceSetup) -> str:
    """The single backend that compiles and runs this setup.

    Mixing instruments of different backends in one setup is an error.
    """
    instruments_by_backend: dict[str, list[str]] = {}
    for instrument in device_setup.instruments:
        instruments_by_backend.setdefault(_eval_backend(instrument), []).append(
            instrument.uid
        )
    if len(instruments_by_backend) > 1:
        conflict = "; ".join(
            f"{backend}: {', '.join(sorted(uids))}"
            for backend, uids in sorted(instruments_by_backend.items())
        )
        raise RuntimeError(
            f"A device setup is compiled by one backend, but this one has"
            f" instruments of several ({conflict})."
        )
    return next(iter(instruments_by_backend), DEFAULT_BACKEND)


def _eval_backend(device: Instrument) -> str:
    """The backend that compiles and runs this instrument."""
    return getattr(device, "backend", DEFAULT_BACKEND)


@cache
def backend_import_prefixes() -> tuple[str, ...]:
    """Module prefixes a serializer may be imported from.

    `laboneq` itself, plus the top-level package of every installed backend
    plugin.
    """
    from importlib.metadata import entry_points

    packages = {
        ep.module.partition(".")[0]
        for ep in entry_points().select(group="laboneq.backends")
    }
    return ("laboneq.", *sorted(f"{package}." for package in packages))
