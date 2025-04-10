# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import attrs
from typing import Any

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.core.utilities.validate_path import validate_path

from .operation import Operation


@classformatter
@attrs.define
class SetNode(Operation):
    """Operation that sets a value at a node."""

    #: Path to the node whose value should be set.
    path: str | None = None
    #: Value that should be set.
    value: Any = None

    def __attrs_post_init__(self):
        if self.path is not None:
            validate_path(self.path)
        if hasattr(self.value, "uid"):
            self.value = self.value.uid
