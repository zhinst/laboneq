# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Naming of the virtual SHFSG an SHFQC is split into.

An SHFQC is driven as two devices, an SHFQA under the instrument's own uid and
an SHFSG under a suffixed one. `laboneq_common::shfqc` derives the same uid, and
the recipe names the SG device by it.
"""

from __future__ import annotations

VIRTUAL_SHFSG_UID_SUFFIX = "_sg"


def to_sg_uid(instrument_uid: str) -> str:
    """Returns the uid of the virtual SHFSG split out of the SHFQC `instrument_uid`."""
    return f"{instrument_uid}{VIRTUAL_SHFSG_UID_SUFFIX}"
