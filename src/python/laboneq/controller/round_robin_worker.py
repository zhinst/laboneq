# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Hashable

T = TypeVar("T")

_logger = logging.getLogger(__name__)


class RoundRobinWorker(Generic[T]):
    """Runs queued items one at a time, taking turns between several queues.

    Each queue runs its items in submission order, and the worker takes one item
    from each queue in turn.

    An item runs at its turn if `is_ready` returns True for it. Otherwise the
    worker skips its queue and starts `wait_ready` for it, which wakes the
    worker once it returns True. If `wait_ready` returns False, the item is
    dropped without running.

    The worker task runs while any queue holds an item, and starts again on the
    next `submit`. `wait_ready` must not raise, and `run_one` records the errors
    of its item. If `run_one` raises, the worker logs the exception and goes on
    with the next item.
    """

    def __init__(
        self,
        *,
        run_one: Callable[[T], Coroutine[None, None, None]],
        wait_ready: Callable[[T], Awaitable[bool]],
        is_ready: Callable[[T], bool],
    ):
        self._run_one = run_one
        self._wait_ready = wait_ready
        self._is_ready = is_ready
        # Dict order is the order in which the queues take their turns.
        self._queues: dict[Hashable, deque[_Entry[T]]] = {}
        self._changed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def submit(self, queue_id: Hashable, item: T):
        """Appends `item` to the queue `queue_id`, creating the queue if needed."""
        entry = _Entry(item)
        queue = self._queues.setdefault(queue_id, deque())
        queue.append(entry)
        self._changed.set()
        if self._task is not None and self._task.done():
            await self._finalize_task()
        if self._task is None:
            self._task = asyncio.create_task(self._work())

    def discard(self, item: T) -> bool:
        """Removes `item` from its queue if it has not started running.

        Returns whether it was removed.
        """
        for queue_id, queue in self._queues.items():
            for entry in queue:
                if entry.item is item:
                    self._remove(queue_id, entry)
                    return True
        return False

    async def stop(self):
        """Waits until every queued item has run or been dropped, then stops the worker."""
        if self._task is None:
            return
        await asyncio.wait([self._task])
        gates = [
            entry.gate
            for queue in self._queues.values()
            for entry in queue
            if entry.gate is not None
        ]
        for gate in gates:
            gate.cancel()
        if gates:
            await asyncio.wait(gates)
        self._queues.clear()
        await self._finalize_task()

    async def cancel(self):
        """Stops the worker at once.

        The item in progress stays unfinished, and the queued items stay queued.
        """
        if self._task is None:
            return
        self._task.cancel()
        await self._finalize_task()

    async def _work(self):
        while (item := await self._next()) is not None:
            await self._run_logging_errors(item)

    async def _run_logging_errors(self, item: T):
        try:
            await self._run_one(item)
        except Exception:
            _logger.exception("Running a queued item failed")

    async def _next(self) -> T | None:
        """Takes the next ready item in turn, or returns None once all queues are empty."""
        while self._queues:
            self._changed.clear()
            for queue_id, queue in list(self._queues.items()):
                entry = queue[0]
                if not self._is_ready(entry.item):
                    # An interrupt cancels a wait from outside. Such a wait is
                    # done, and starts again.
                    if entry.gate is None or entry.gate.done():
                        entry.gate = asyncio.create_task(self._gate(entry))
                    continue
                queue.popleft()
                if entry.gate is not None:
                    entry.gate.cancel()
                # The served queue takes its next turn after every other queue.
                del self._queues[queue_id]
                if queue:
                    self._queues[queue_id] = queue
                return entry.item
            await self._changed.wait()
        return None

    async def _gate(self, entry: _Entry[T]):
        ready = await self._wait_ready(entry.item)
        # The gate must not cancel itself.
        entry.gate = None
        if ready:
            self._changed.set()
            return
        for queue_id, queue in self._queues.items():
            if queue and queue[0] is entry:
                self._remove(queue_id, entry)
                return

    def _remove(self, queue_id: Hashable, entry: _Entry[T]):
        queue = self._queues[queue_id]
        queue.remove(entry)
        if entry.gate is not None:
            entry.gate.cancel()
        if not queue:
            del self._queues[queue_id]
        self._changed.set()

    async def _finalize_task(self):
        assert self._task is not None
        task, self._task = self._task, None
        with suppress(asyncio.CancelledError):
            await task


@dataclass(eq=False)
class _Entry(Generic[T]):
    item: T
    gate: asyncio.Task[None] | None = None
