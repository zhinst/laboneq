# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Optional dependencies for the LabOne Q package.

This module provides type hinting for the optionals and a helper function
for importing modules, which can be used for importing optionals, as it also
provides missing module error handling.
"""
from __future__ import annotations

import importlib
from types import ModuleType

from typing_extensions import TypeAlias

try:
    import xarray as xr

    HAS_XARRAY = True
except ModuleNotFoundError:
    HAS_XARRAY = False

# Type hints
XarrayDataArray: TypeAlias = "xr.DataArray"
XarrayDataset: TypeAlias = "xr.Dataset"
Xarray: TypeAlias = "xr"


def import_optional(
    name: str, message: str | None = None, allow_missing: bool = False
) -> ModuleType | None:
    """Import optional dependency.

    This is a helper function for importing optional dependencies and
    providing an informational error message if the dependency is not found.

    Args:
        name: Name of the module.
        message: Extra message for the error message.
            It will be appended to:
                'Optional dependency '<name>' is missing: <message>'
        allow_missing: Allow missing import.

    Returns:
        Module if the module is installed.
        `None` if `allow_missing` is `True` and module
        is not found.

    Raises:
        ModuleNotFoundError: When module is not found
            and `allow_missing` is `False`.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        if allow_missing:
            return
        msg = f"Optional dependency '{name}' is missing"
        if message:
            msg += f": {message}"
        raise ModuleNotFoundError(msg) from None
