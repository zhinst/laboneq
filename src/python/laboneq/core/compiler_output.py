# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Deserializes compiler output into Python objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeAlias

from laboneq.core.exceptions import LabOneQException

if TYPE_CHECKING:
    from laboneq.data.scheduled_experiment import ScheduledExperiment

Deserializer: TypeAlias = Callable[[bytes], "ScheduledExperiment"]


def peek_backend_name(data: bytes) -> str:
    """Read the identifier of the backend that produced the compiler output `data`."""
    from laboneq._rust import compiled_experiment

    return compiled_experiment.peek_backend_name(data)


class DeserializerRegistry:
    """Backend deserializers for compiler output, keyed by the backend identifier
    recorded in `CompiledExperiment.metadata.createdBy`."""

    def __init__(self) -> None:
        self._deserializers: dict[str, Deserializer] = {}

    def register(self, backend: str, deserializer: Deserializer) -> None:
        """Register `backend`'s deserializer for the compiler output it produces.

        `backend` must match the backend half of `CompiledExperiment.metadata.createdBy`
        in those bytes; that is what selects this deserializer.
        """
        existing = self._deserializers.get(backend)
        if existing is not None:
            if existing is deserializer:
                return
            raise LabOneQException(
                f"Backend {backend!r} already registered with "
                f"{existing!r}, cannot re-register with {deserializer!r}"
            )
        self._deserializers[backend] = deserializer

    def deserialize(self, data: bytes) -> ScheduledExperiment:
        """Deserialize compiler output into a `ScheduledExperiment`.

        Dispatches to the registered deserializer of whichever backend produced
        the bytes, as recorded in `CompiledExperiment.metadata.createdBy`.
        """
        backend = peek_backend_name(data)
        deserializer = self._deserializers.get(backend)
        if deserializer is None:
            registered = sorted(self._deserializers)
            raise LabOneQException(
                f"No deserializer registered for backend {backend!r}. "
                f"Registered backends: {registered}. Is the backend that produced these "
                "bytes installed and imported?"
            )
        return deserializer(data)


_DEFAULT_REGISTRY = DeserializerRegistry()


def register_deserializer(backend: str, deserializer: Deserializer) -> None:
    """Register `backend`'s deserializer in the process-wide registry.

    See `DeserializerRegistry.register`.
    """
    _DEFAULT_REGISTRY.register(backend, deserializer)


def deserialize(data: bytes) -> ScheduledExperiment:
    """Deserialize compiler output using the process-wide registry.

    See `DeserializerRegistry.deserialize`.
    """
    return _DEFAULT_REGISTRY.deserialize(data)
