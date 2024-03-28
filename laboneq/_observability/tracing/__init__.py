# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q observability tracing module.

The module contains tracing API for LabOneQ.
"""
import os

_TRACING_AVAILABLE = bool(int(os.getenv("LABONEQ_ENABLE_OBSERVABILITY", 0)))


if _TRACING_AVAILABLE:
    from ._tracer import disable, enable, get_tracer, trace

else:
    from ._noop_tracer import disable, enable, get_tracer, trace

__all__ = ["get_tracer", "trace", "enable", "disable"]
