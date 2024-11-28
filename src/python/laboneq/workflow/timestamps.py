# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Timestamps functions for LabOne Q workflows."""

from __future__ import annotations

from datetime import datetime, timezone

from dateutil.tz import tzlocal


def utc_now(timestamp: datetime | None = None) -> datetime:
    """Returns the given timestamp or the current time.

    Uses UTC timezone.
    """
    return (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)


def local_now(dt: datetime | None = None) -> datetime:
    """Returns the given datetime in the local timezone.

    Arguments:
        dt:
            The time to convert to the local timezone. If None,
            the current time is used.

    Returns:
        The time in the local timezone.
    """
    return (dt or datetime.now(tzlocal())).astimezone(tzlocal())


def local_timestamp(dt: datetime | None = None) -> str:
    """Returns a string formatted timestamp in the local timezone.

    Arguments:
        dt:
            The time use. If None, the current time is used.

    Returns:
        The datetime formatted as '%Y%m%dT%H%M%S'.

    Note:
        This function is used by the `FolderStore` to generate the
        timestamp used in the names of the logbook folders it creates.
    """
    return local_now(dt).strftime("%Y%m%dT%H%M%S")


def local_date_stamp(dt: datetime | None = None) -> str:
    """Returns a string formatted date stamp in the local timezone.

    Arguments:
        dt:
            The time use. If None, the current time is used.

    Returns:
        THe datetime formatted as '%Y%m%d'.

    Note:
        This function is used by the `FolderStore` to generate the
        date stamp used in the names of the per-day subfolders it
        creates.
    """
    return local_now(dt).strftime("%Y%m%d")
