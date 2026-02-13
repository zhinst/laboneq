# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import threading
from typing import Iterable


class Context(abc.ABC):
    @abc.abstractmethod
    def add(self, section):
        raise NotImplementedError


_store = threading.local()
_store.active_contexts = []


def get_local_contexts():
    try:
        return _store.active_contexts
    except AttributeError:
        _store.active_contexts = []
    return _store.active_contexts


def push_context(context: Context):
    get_local_contexts().append(context)


def peek_context() -> Context | None:
    return get_local_contexts()[-1] if len(_store.active_contexts) else None


def pop_context() -> Context:
    return get_local_contexts().pop()


def iter_contexts() -> Iterable[Context]:
    return iter(get_local_contexts())


def reversed_iter_contexts() -> Iterable[Context]:
    return reversed(get_local_contexts())
