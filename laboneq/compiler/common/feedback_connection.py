# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

from laboneq.compiler.common.awg_info import AwgKey


@dataclass
class FeedbackConnection:
    """A dataclass representing a feedback connection between AWGs.

    Attributes:
        tx: The transmitter AWG
        rx: A set of receiver AWGs.
    """

    tx: AwgKey | None
    rx: Set[AwgKey] = field(default_factory=set)
