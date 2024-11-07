# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Basic logbook classes."""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from laboneq.workflow.recorder import ExecutionRecorder

if TYPE_CHECKING:
    from laboneq.workflow import Workflow


class LogbookStore(abc.ABC):
    """Protocol for storing a collection of records of workflow execution."""

    @abc.abstractmethod
    def create_logbook(self, workflow: Workflow, start_time: datetime) -> Logbook:
        """Create a logbook for recording a single workflow execution."""

    def activate(self) -> None:
        """Activate this logbook store.

        Workflows write to all active logbook stores by default.
        """
        if self not in _active_logbook_stores:
            _active_logbook_stores.append(self)

    def deactivate(self) -> None:
        """Deactivate this logbook store.

        If this store is not active, this method does nothing.
        """
        if self in _active_logbook_stores:
            _active_logbook_stores.remove(self)


_active_logbook_stores = []


def active_logbook_stores() -> list[LogbookStore]:
    """Returns a list of active logbook stores.

    Modifying the list does not affect the active logbooks. Use
    `LogBookStore.activate()` or `LogBookStore.deactivate()` to activate
    or deactivate individual logbook stores, respectively.
    """
    return _active_logbook_stores.copy()


class Logbook(ExecutionRecorder):
    """Protocol for storing the record of a single workflow execution."""


def format_time(time: datetime) -> str:
    """Format a datetime object as a string."""
    return time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")
