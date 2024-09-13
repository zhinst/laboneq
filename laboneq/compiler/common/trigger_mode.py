# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class TriggerMode(Enum):
    """Enum used to tell the code generator how the triggering scheme should be
    configured for a single AWG.

    NONE: The AWG core is triggered via ZSync.
    DIO_TRIGGER: Used to synchronize HDAWG cores in a standalone HDAWG or
        HDAWG+UHFQA setups. The generated SeqC code will be such that:
        1. The first HDAWG core emits a DIO signal and blocks until it is
           received.
        2. The rest of the AWG cores block until DIO trigger is received.
    DIO_WAIT: Used exclusively to make UHFQA AWG block until DIO trigger is received.
    INTERNAL_TRIGGER_WAIT: Used for SHFQC internal triggering scheme.
    INTERNAL_READY_CHECK: Used for standalone HDAWGs.

    """

    NONE = "none"
    DIO_TRIGGER = "dio_trigger"
    DIO_WAIT = "dio_wait"
    INTERNAL_TRIGGER_WAIT = "internal_trigger_wait"
    INTERNAL_READY_CHECK = "internal_ready_check"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"
