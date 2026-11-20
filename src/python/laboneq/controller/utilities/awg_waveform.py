# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Lazy access to `zhinst.utils`' AWG waveform conversion."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


@cache
def convert_awg_waveform() -> Callable[..., np.ndarray]:
    """Resolve `zhinst.utils.convert_awg_waveform` on first use.

    Importing `zhinst.utils` pulls in `scipy.io` & co. and is a significant part of
    `import laboneq`, while only QCCS waveform upload needs this single function.
    """
    import zhinst.utils  # type: ignore[import-untyped]

    return zhinst.utils.convert_awg_waveform
