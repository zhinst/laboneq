# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Lazy access to the LabOne client packages.

`zhinst.comms_schemas` is compiled and not available on every platform. Importing it at
module level would make it a requirement of `import laboneq` itself; the names below
resolve on first use instead.

Reach them as attributes of this module — a `from` import resolves at import time and
defeats the purpose.
"""

from __future__ import annotations

import importlib
from typing import Any

_ORIGINS = {
    "KernelInfo": "zhinst.comms_schemas.labone",
    "KernelSession": "zhinst.comms_schemas.labone",
    "AnnotatedValue": "zhinst.comms_schemas.labone.core",
    "ShfGeneratorWaveformVectorData": "zhinst.comms_schemas.labone.core",
    "ZIContext": "zhinst.comms_schemas.labone.core",
    "LabOneCoreError": "zhinst.comms_schemas.labone.core.errors",
    "NotFoundError": "zhinst.comms_schemas.labone.core.errors",
    "OverwhelmedError": "zhinst.comms_schemas.labone.core.errors",
}

__all__ = list(_ORIGINS)


def __getattr__(name: str) -> Any:
    try:
        origin = _ORIGINS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    resolved = getattr(importlib.import_module(origin), name)
    # `__getattr__` runs only for names the module lacks, so binding here caches the import.
    globals()[name] = resolved
    return resolved
