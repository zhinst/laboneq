# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Setup description payloads for different hardware generations.

A ``DeviceSetup`` may carry a ``SetupDescription`` that is opaque to the
public API but consumed by the compiler. The base class is abstract.
``SetupDescriptionZQCS`` lives in ``laboneq_zqcs`` and is only known
once that backend has been imported. QCCS has no payload of its own.
"""

from __future__ import annotations

from typing import Any, TypeVar

import attrs

_T = TypeVar("_T", bound="SetupDescription")

_REGISTRY: dict[str, type["SetupDescription"]] = {}


def register_setup_description(cls: type[_T]) -> type[_T]:
    """Class decorator: register a `SetupDescription` subclass by name."""
    if cls.__name__ in _REGISTRY and _REGISTRY[cls.__name__] is not cls:
        raise ValueError(
            f"SetupDescription subclass {cls.__name__!r} already registered"
        )
    _REGISTRY[cls.__name__] = cls
    return cls


def get_setup_description_class(name: str) -> type["SetupDescription"]:
    """Look up a registered `SetupDescription` subclass by name.

    Raises:
        KeyError: if no subclass with that name has been registered. For
            ZQCS this typically means the `laboneq_zqcs` backend was not
            imported.
    """
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"No SetupDescription subclass named {name!r} is registered. "
            "If this is a ZQCS payload, ensure `laboneq_zqcs` is installed "
            "and imported."
        ) from e


@attrs.define
class SetupDescription:
    """Abstract base for hardware-generation-specific setup descriptions."""

    def to_dict(self) -> dict[str, Any]:
        """Encode this instance's fields as a JSON-friendly dict."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SetupDescription:
        """Reconstruct an instance from a dict produced by :meth:`to_dict`."""
        raise NotImplementedError

    def serialize(self) -> dict[str, Any]:
        """Encode with a class-name tag for registry-based dispatch."""
        return {"_type": type(self).__name__, "fields": self.to_dict()}

    @staticmethod
    def deserialize(data: dict[str, Any]) -> SetupDescription:
        """Inverse of :meth:`serialize`; dispatches via the registry."""
        return get_setup_description_class(data["_type"]).from_dict(data["fields"])
