# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    import numpy as np


@attrs.frozen
class LoggedVariable:
    """Recorded history of a logged variable.

    One entry in Results.variable_results.

    Attributes:
        data: Recorded values. Shape derived statically from the enclosing
            real-time loops. dtype matches the Variable's type (rich types
            decoded to natural units; raw integers at native width).
        valid: Boolean mask, same shape as `data`. True where a value was
            recorded; False where a slot was never written (early-exit
            do_until iteration, match arm with fewer updates, etc.).
        axis_name: Axis labels, parallel to AcquiredResult.axis_name.
        axis: Axis grids, parallel to AcquiredResult.axis.
        last_nt_step: Near-time progress for partial NT runs; None otherwise.
    """

    data: np.ndarray
    valid: np.ndarray
    axis_name: list
    axis: list
    last_nt_step: list[int] | None = None
