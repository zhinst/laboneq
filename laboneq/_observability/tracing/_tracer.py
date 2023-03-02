# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import typing as t
from functools import wraps

from zhinst.core import _tracing as tracing

from laboneq import __version__

from . import _noop_tracer

_TRACING_ENABLED = False


def enable():
    global _TRACING_ENABLED
    _TRACING_ENABLED = True
    tracing.enable_tracing()


def disable():
    global _TRACING_ENABLED
    _TRACING_ENABLED = False
    tracing.disable_tracing()


def get_tracer():
    if _TRACING_ENABLED:
        return tracing.get_tracer("laboneq.observability.tracing", __version__)
    return _noop_tracer._NoopObj


def trace(
    span_name: t.Optional[str] = None,
    attributes: t.Optional[t.Dict[str, t.Any]] = None,
    attribute_callback: t.Callable = None,
    disable_tracing_during=False,
):
    """Decorator to turn function to a span.

    Args:
        span_name: Name of the span. Otherwise function name is used to name the span.
        attributes: Span attributes as key-value pairs.
        attribute_callback: A callback function for getting attributes.
            The callback function must return a flat dictionary with keys as strings.
        disable_tracing_during: Disable tracing during the execution of the function.
            This is to avoid creating unnecessasry spans.
    """

    def outer_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _TRACING_ENABLED:
                attrs = {
                    "code.function": func.__qualname__,
                    "code.namespace": func.__module__,
                }
                if attributes:
                    attrs.update(attributes)
                if attribute_callback:
                    attrs.update(attribute_callback(*args, **kwargs))
                with get_tracer().start_span(
                    func.__qualname__ if not span_name else span_name,
                    attrs,
                ) as _:
                    if disable_tracing_during:
                        tracing.disable_tracing()
                    res = func(*args, **kwargs)
                    if disable_tracing_during:
                        tracing.enable_tracing()
                    return res
            return func(*args, **kwargs)

        return wrapper

    return outer_wrapper
