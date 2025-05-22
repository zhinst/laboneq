# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.compiler.seqc.feedback_register_allocator import (
        FeedbackRegisterAllocator,
        FeedbackRegisterAllocation,
    )
    from laboneq.compiler.common.awg_info import AWGInfo, AwgKey


def allocate_feedback_registers(
    awgs: list[AWGInfo],
    signal_to_handle: dict[str, str],
    feedback_register_allocator: FeedbackRegisterAllocator,
) -> dict[AwgKey, FeedbackRegisterAllocation]:
    """Allocate feedback registers for each AWG.

    Returns:
        Allocated feedback registers.
    """
    register_map: dict = {}
    for awg in awgs:
        for sig in awg.signals:
            if sig.id not in signal_to_handle:
                continue
            register_map[awg.key] = feedback_register_allocator.allocate(
                awg.key, signal_to_handle[sig.id]
            )
    return register_map
