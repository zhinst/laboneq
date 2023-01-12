# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q observability tracing module.

The module contains tracing API for LabOneQ.
"""

_TRACING_AVAILABLE = False


try:
    from zhinst.core import _tracing

    _TRACING_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    pass


if _TRACING_AVAILABLE:
    from ._tracer import disable, enable, get_tracer, trace

else:
    from ._noop_tracer import disable, enable, get_tracer, trace

__all__ = ["get_tracer", "trace", "enable", "disable"]
