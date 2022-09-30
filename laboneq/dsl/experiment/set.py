# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from .operation import Operation


@dataclass(init=True, repr=True, order=True)
class Set(Operation):

    """Operation that sets a value at a node."""

    #: Path to the node whose value should be set.
    path: str = field(default=None)
    #: Key of the node that should be set.
    key: str = field(default=None)
    #: Value that should be set.
    value: Any = field(default=None)

    def __post_init__(self):
        if hasattr(self.value, "uid"):
            self.value = self.value.uid
