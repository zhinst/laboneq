# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Backwards compatibility instrumentors for LabOne Q."""

from laboneq import __version__ as laboneq_version

# Backwards compatibility for old versions without `LabOneQInstrumentor`.
try:
    from laboneq.instrumentation import LabOneQInstrumentor
except (ImportError, ModuleNotFoundError):
    from .instrumentor import LabOneQInstrumentor

# Backwards compatibility for old versions without scheduler/codegenerator instrumentation.
_USE_LEGACY_INSTRUMENTOR = laboneq_version <= "26.4.0"
if _USE_LEGACY_INSTRUMENTOR:
    from .instrumentor import LabOneQInstrumentor
else:
    from laboneq.instrumentation import LabOneQInstrumentor

__all__ = ["LabOneQInstrumentor"]
