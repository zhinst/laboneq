# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from functools import wraps


def enable():
    pass


def disable():
    pass


class _NoopObject:
    """This class in meant to do nothing."""

    def __init__(self, *_, **__) -> None:
        pass

    def __getattr__(self, *_, **__) -> "_NoopObject":
        return self

    def __call__(self, *_, **__) -> "_NoopObject":
        return self

    def __enter__(self, *_, **__) -> "_NoopObject":
        return self

    def __exit__(self, *_, **__):
        pass


_NoopObj = _NoopObject()


def get_tracer() -> _NoopObj:
    """"""
    return _NoopObj


def trace(*_, **__):
    """Noop Decorator to turn function to a span."""

    def outer_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return outer_wrapper
