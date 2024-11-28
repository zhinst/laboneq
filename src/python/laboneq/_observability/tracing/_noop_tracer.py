# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


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


def get_tracer() -> _NoopObject:
    return _NoopObj


def trace(*_, **__):
    def outer_wrapper(func):
        return func

    return outer_wrapper
