# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class TriggerMode(Enum):
    NONE = "none"
    DIO_TRIGGER = "dio_trigger"
    DIO_WAIT = "dio_wait"
    INTERNAL_TRIGGER_WAIT = "internal_trigger_wait"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"
