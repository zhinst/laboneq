# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import threading
from contextlib import contextmanager
from typing import Generator, Iterable


class Context(abc.ABC):
    @abc.abstractmethod
    def add(self, section):
        raise NotImplementedError


_store = threading.local()


def get_local_contexts():
    try:
        return _store.active_contexts
    except AttributeError:
        _store.active_contexts = []
    return _store.active_contexts


def push_context(context: Context):
    get_local_contexts().append(context)


def peek_context() -> Context | None:
    contexts = get_local_contexts()
    return contexts[-1] if contexts else None


def pop_context() -> Context:
    return get_local_contexts().pop()


def iter_contexts() -> Iterable[Context]:
    return iter(get_local_contexts())


def reversed_iter_contexts() -> Iterable[Context]:
    return reversed(get_local_contexts())


@contextmanager
def isolated_contexts() -> Generator[None, None, None]:
    """Hide the active contexts for the duration of the block.

    Code within the block behaves as if no context was active, so sections
    created within it are not added to an enclosing section or experiment.
    """
    outer_contexts = get_local_contexts()
    _store.active_contexts = []
    try:
        yield
    finally:
        _store.active_contexts = outer_contexts
