# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import concurrent
import concurrent.futures
from contextlib import contextmanager
import signal
import threading
from typing import Coroutine, TypeVar, Callable
import asyncio


T = TypeVar("T")


# Required to keep sync interface callable from Jupyter Notebooks
# See https://blog.jupyter.org/ipython-7-0-async-repl-a35ce050f7f7
class EventLoopHolder:
    """Manages a background event loop running in a daemon thread.

    This class provides a wrapper for a background event loop that operates in a
    separate daemon thread. The event loop remains active for the entire lifetime
    of the process.

    **Warning:**
    This class does not provide a mechanism to terminate the event loop. It is the
    caller's responsibility to ensure that this class is instantiated only once
    per desired event loop scope and that the instance is reused as needed.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._loop = asyncio.new_event_loop()

    def run(self, func: Callable[..., Coroutine[None, None, T]], *args, **kwargs) -> T:
        self._ensure_event_loop()
        with self._override_sigint_handler():
            return self._wait_with_yielding(
                asyncio.run_coroutine_threadsafe(func(*args, **kwargs), self._loop)
            )

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


class EventLoopMixIn:
    _thread_local = threading.local()

    @property
    def _event_loop(self) -> EventLoopHolder:
        event_loop = getattr(self._thread_local, "laboneq_event_loop", None)
        if event_loop is None:
            event_loop = EventLoopHolder()
            self._thread_local.laboneq_event_loop = event_loop
        return event_loop
