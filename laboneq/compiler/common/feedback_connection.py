# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class FeedbackConnection:
    """A dataclass representing a feedback connection beween signals.

    Attributes:
        acquire (Optional[str]): A string that specifies the acquisition signal (source).
        drive (Set[str]): A set of strings that specifies the reacting signals' names (targets).
    """

    acquire: str | None
    drive: Set[str] = field(default_factory=set)
