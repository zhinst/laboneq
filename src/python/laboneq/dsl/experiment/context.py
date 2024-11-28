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


def push_context(context: Context):
    _store.active_contexts.append(context)


def peek_context() -> Context | None:
    return _store.active_contexts[-1] if len(_store.active_contexts) else None


def pop_context() -> Context:
    return _store.active_contexts.pop()


def iter_contexts() -> Iterable[Context]:
    return iter(_store.active_contexts)


def reversed_iter_contexts() -> Iterable[Context]:
    return reversed(_store.active_contexts)
