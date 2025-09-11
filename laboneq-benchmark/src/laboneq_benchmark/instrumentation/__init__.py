# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Backwards compatibility instrumentors for LabOne Q."""

try:
    from laboneq.instrumentation import LabOneQInstrumentor
except (ImportError, ModuleNotFoundError):
    from .instrumentor import LabOneQInstrumentor

__all__ = ["LabOneQInstrumentor"]
