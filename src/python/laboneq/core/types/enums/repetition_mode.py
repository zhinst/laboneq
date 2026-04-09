# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class RepetitionMode(Enum):
    """Controls how the repetition interval between averaging loop iterations is determined."""

    FASTEST = "fastest"
    """Execute iterations as fast as the hardware permits."""

    CONSTANT = "constant"
    """Execute at a fixed repetition time specified by the user."""

    AUTO = "auto"
    """Execute at a constant rate determined automatically from the longest iteration.

    Incompatible with ``SectionTimingMode.STRICT``.
    """
