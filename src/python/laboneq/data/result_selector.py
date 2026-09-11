# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import attrs


@attrs.define(frozen=True)
class ResultSelector:
    """Selection of acquired results.

    !!! warning
        This is an experimental class and the properties may not be stable.
    """

    handles: tuple[str, ...] | None = None
    nt: Literal["all", "latest"] = "all"
