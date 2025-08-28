# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Logbooks for recording the operations of workflows."""

__all__ = [
    "LogbookStore",
    "Logbook",
    "FolderStore",
    "LoggingStore",
    "SaveMode",
    "active_logbook_stores",
    "format_time",
    "DEFAULT_LOGGING_STORE",
]


from .core import (
    Logbook,
    LogbookStore,
    active_logbook_stores,
    format_time,
)
from .folder_store import FolderStore, SaveMode
from .logging_store import LoggingStore

# Add the default logging store:
DEFAULT_LOGGING_STORE = LoggingStore()
DEFAULT_LOGGING_STORE.activate()
