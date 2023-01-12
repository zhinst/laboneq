# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

from laboneq.dsl.enums import IODirection, IOSignalType


@dataclass(init=True, repr=True, order=True)
class Connection:
    direction: IODirection = field(default=IODirection.OUT)
    local_path: Optional[str] = field(default=None)
    local_port: Optional[str] = field(default=None)
    remote_path: Optional[str] = field(default=None)
    remote_port: Optional[str] = field(default=None)
    signal_type: Optional[IOSignalType] = field(default=None)
