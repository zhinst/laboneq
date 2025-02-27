# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry instrumentation for tracing LabOne Q.

Requires `laboneq[instrumentation]` optional
dependencies to be installed.
"""

from .instrumentor import LabOneQInstrumentor

__all__ = "LabOneQInstrumentor"
