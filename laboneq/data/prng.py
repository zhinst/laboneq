# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PRNG:
    range: int
    seed: int = field(default=1)


@dataclass
class PRNGSample:
    """Representation in the LabOne Q DSL of values drawn from an on-device PRNG."""

    uid: str = None
    prng: PRNG = None
    count: int = 1
