# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Generator


class _ContextStorage(threading.local):
    # NOTE: Subclassed for type hinting
    scopes: ClassVar[dict[str, list]] = defaultdict(list)


_contexts = _ContextStorage()


T = TypeVar("T")


class LocalContext(Generic[T]):
    """Local context."""

    _scope = "default"

    @classmethod
    @contextmanager
    def scoped(cls, obj: T | None = None) -> Generator:
        cls.enter(obj)
        try:
            yield
        finally:
            cls.exit()

    @classmethod
    def enter(cls, obj: T | None = None) -> None:
        _contexts.scopes[cls._scope].append(obj)

    @classmethod
    def exit(cls) -> T:
        try:
            return _contexts.scopes[cls._scope].pop()
        except (KeyError, IndexError) as error:
            raise RuntimeError("No active context.") from error

    @classmethod
    def get_active(cls) -> T | None:
        """Get an active context."""
        try:
            return _contexts.scopes[cls._scope][-1]
        except IndexError:
            return None

    @classmethod
    def iter_stack(cls, *, reverse: bool = False) -> Generator[T, None, None]:
        """Iterate over the existing stack.

        Iterates from top to bottom by default.

        Arguments:
            reverse: Iterate from bottom to top.
        """
        yield from (
            reversed(_contexts.scopes[cls._scope])
            if reverse
            else _contexts.scopes[cls._scope]
        )
