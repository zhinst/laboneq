# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent
import concurrent.futures
import signal
import threading
import time
from contextlib import contextmanager, suppress
from typing import Awaitable, Callable, Coroutine, Generic, TypeVar

T = TypeVar("T")


async def wait_for(awaitable: Awaitable[T], timeout_s: float | None) -> T:
    """Await *awaitable* for at most *timeout_s* seconds (None: no limit).

    Raises `asyncio.TimeoutError` on expiry, like `asyncio.wait_for`, but
    never drops a cancellation of the caller: before Python 3.12,
    `asyncio.wait_for` returns the inner result when it completes in the same
    loop iteration as the cancellation arrives, and the caller carries on as if
    nobody had cancelled it.
    """
    inner = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({inner}, timeout=timeout_s)
    except asyncio.CancelledError:
        inner.cancel()
        with suppress(asyncio.CancelledError):
            await inner
        raise
    if not done:
        inner.cancel()
        with suppress(asyncio.CancelledError):
            await inner
        raise asyncio.TimeoutError
    return inner.result()


# Required to keep sync interface callable from Jupyter Notebooks
# See https://blog.jupyter.org/ipython-7-0-async-repl-a35ce050f7f7
class EventLoopHolder:
    """Manages a background event loop running in a daemon thread.

    This class provides a wrapper for a background event loop that operates in a
    separate daemon thread. The event loop remains active for the entire lifetime
    of the process.

    **Warning:**
    This class does not provide an automatism to terminate the event loop. It is the
    caller's responsibility to ensure that this class is instantiated only once
    per desired event loop scope, that the instance is reused as needed, or a call
    to stop() is issued.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self, func: Callable[..., Coroutine[None, None, T]], *args, **kwargs) -> T:
        self._ensure_event_loop()
        with self._override_sigint_handler():
            return self._wait_with_yielding(
                asyncio.run_coroutine_threadsafe(func(*args, **kwargs), self._loop)
            )

    def start(self) -> None:
        if not self._loop:
            self._loop = asyncio.new_event_loop()
        self._ensure_event_loop()

    def stop(self) -> None:
        if self._loop:
            self._loop.stop()
            while self._loop.is_running():
                self._loop.call_soon_threadsafe(lambda: None)  # wake up the loop
                time.sleep(0.01)
            self._loop.close()
        self._loop = None
        self._thread = None

    def _ensure_event_loop(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._event_loop_thread, daemon=True)
            self._thread.start()

    def _event_loop_thread(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @contextmanager
    def _override_sigint_handler(self):
        if threading.current_thread() is not threading.main_thread():
            # SIGINT handler can only be overridden in the main thread,
            # TODO(2K): Mechanism to interrupt the event loop running in a thread
            yield
            return
        orig_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, orig_handler)

    def _sigint_handler(self, *args):
        asyncio.run_coroutine_threadsafe(self._cancel_all_tasks(), self._loop).result()

    async def _cancel_all_tasks(self):
        this_task = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task != this_task:
                task.cancel()

    def _wait_with_yielding(self, future: concurrent.futures.Future[T]) -> T:
        while True:
            try:
                # Exit future wait every 0.1s to handle SIGINT, balancing
                # responsiveness and CPU load.
                return future.result(timeout=0.1)  # @IgnoreException
            except concurrent.futures.TimeoutError:  # noqa: PERF203
                pass


class AsyncWorker(Generic[T]):
    """Base class for async workers that process queued items sequentially.

    Pass the `run_one` coroutine that implements the processing logic.

    The worker stops automatically when the queue is empty for a specified number of cycles,
    and restarts processing when new items are submitted.

    Calling `stop` will block until all items in the queue are processed and the worker is stopped.

    Make sure no exceptions escape from `run_one` — they'll only be
    caught by the worker lifecycle logic as a last resort, and likely at an inappropriate point
    in execution, and also will prevent the worker from processing subsequent items.
    """

    def __init__(
        self,
        *,
        run_one: Callable[[T], Coroutine[None, None, None]],
        max_idle_cycles: int = 5,
        cycle_timeout: float = 1.0,
    ):
        self._run_one = run_one
        self._max_idle_cycles = max_idle_cycles
        self._cycle_timeout = cycle_timeout
        self._idle_cycles = 0
        self._task: asyncio.Task | None = None
        self._queue = asyncio.Queue[T]()

    async def _worker(self):
        try:
            while True:
                try:
                    item = await wait_for(self._queue.get(), self._cycle_timeout)
                    self._idle_cycles = 0
                    await self._run_one(item)
                    self._queue.task_done()
                except asyncio.TimeoutError:  # noqa: PERF203
                    self._idle_cycles += 1
                    if self._idle_cycles >= self._max_idle_cycles:
                        # Idle timeout reached. Stopping background task.
                        break
        except asyncio.CancelledError:
            pass
        finally:
            self._idle_cycles = 0

    async def _start_if_needed(self):
        if self._task is not None and self._task.done():
            # Catch any leftover uncaught exceptions from a previous run
            await self._finalize_task()
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def submit(self, item: T):
        await self._queue.put(item)
        await self._start_if_needed()

    async def stop(self):
        if self._task is not None:
            # Complete the queue, but avoid waiting forever if the worker
            # stops unexpectedly (e.g. due to an uncaught exception)
            join = asyncio.create_task(self._queue.join())
            try:
                await asyncio.wait(
                    [join, self._task], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                # the queue is not joined if the worker task stopped mid-item
                join.cancel()
                # `stop` must not return with a task of its own still pending.
                await asyncio.wait([join])
            self._task.cancel()
            await self._finalize_task()

    async def _finalize_task(self):
        assert self._task is not None
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
