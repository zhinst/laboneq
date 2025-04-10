# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
from typing import List

from laboneq.core.types.enums import IODirection, IOSignalType
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
@attrs.define
class Port:
    """Abstraction of a port"""

    direction: IODirection
    uid: str | None = attrs.field(default=None)
    connector_labels: list[str] = attrs.field(factory=list)
    physical_port_ids: List[str] = attrs.field(factory=list)
    signal_type: IOSignalType | None = attrs.field(default=None)
