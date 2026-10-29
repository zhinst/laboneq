# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from laboneq._rust import compiler as compiler_rs
    from laboneq.dsl.device.device_setup import DeviceSetup


@dataclass
class SetupDescriptionResult:
    """Result of `build_setup_description()`: the capnp setup description object and the
    per-signal maps_to mapping (experiment signal -> channel identifier)."""

    setup_description: Any
    signal_map: dict[str, str]


class CompilerHooks(ABC):
    @staticmethod
    @abstractmethod
    def backend_name() -> str:
        """The name of the backend these hooks belong to.

        Matches what the backend's `CompilerBackend::name()` returns in Rust,
        which is also written to `DeviceSetup.backend` and
        `CompiledExperiment.metadata.createdBy`.
        """

    @staticmethod
    @abstractmethod
    def compiler_module() -> compiler_rs: ...

    @staticmethod
    @abstractmethod
    def build_setup_description(
        device_setup: DeviceSetup, signal_map: dict[str, str]
    ) -> SetupDescriptionResult:
        """Build the capnp setup description for this backend's device class.

        Converts `device_setup` into the backend-specific setup representation used
        during compiler payload serialization.

        Args:
            device_setup: The DSL device setup to compile against.
            signal_map: Experiment signal uid -> logical signal path.

        Returns:
            The setup description and the signal map that resolves experiment signal
                uids to this backend's channel identifiers.
        """


T = TypeVar("T", bound=CompilerHooks)

_registered_compiler_hooks: dict[str, type[CompilerHooks]] = {}


def register_compiler_hooks(hooks: type[T]) -> type[T]:
    name = hooks.backend_name()
    registered = _registered_compiler_hooks.get(name)
    if registered is not None and registered is not hooks:
        raise ValueError(
            f"Backend {name!r} is already registered, by "
            f"{registered.__module__}.{registered.__qualname__}. A backend name "
            f"has to be unique across the installed backends."
        )
    _registered_compiler_hooks[name] = hooks
    return hooks


def get_compiler_hooks(backend: str) -> type[CompilerHooks]:
    if backend not in _registered_compiler_hooks:
        installed = ", ".join(sorted(_registered_compiler_hooks)) or "none"
        raise ValueError(
            f"No compiler hooks registered for backend {backend!r}. "
            f"Installed backends: {installed}."
        )
    return _registered_compiler_hooks[backend]


def all_compiler_hooks() -> Iterator[type[CompilerHooks]]:
    yield from _registered_compiler_hooks.values()


def resolve_compiler_module(backend: str) -> compiler_rs:
    """Return the compiler module for the given backend."""
    return get_compiler_hooks(backend).compiler_module()
