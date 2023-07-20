# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import threading


class Context(abc.ABC):
    @abc.abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, f):
        raise NotImplementedError

    @abc.abstractmethod
    def add(self, section):
        raise NotImplementedError


_store = threading.local()
_store.active_contexts = []


def push_context(context):
    _store.active_contexts.append(context)


def peek_context():
    return _store.active_contexts[-1] if len(_store.active_contexts) else None


def pop_context():
    return _store.active_contexts.pop()


def iter_contexts():
    return iter(_store.active_contexts)


def current_context() -> Context | None:
    return _store.active_contexts[-1] if len(_store.active_contexts) else None
