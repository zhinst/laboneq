# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Dict, List, Optional

from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.common.signal_obj import SignalObj


class FeedbackRegisterAllocator:
    def __init__(self, signals: Dict[str, SignalObj], events: List[Dict[str, Any]]):
        self.signals = signals
        self.matches: Dict[str, bool] = {}  # handle -> is_global
        self.feedback_registers: Dict[AwgKey, int] = {}  # AWG -> allocated reg

        for event in events:
            if event["event_type"] == EventType.SECTION_START:
                handle = event.get("handle")
                if handle is not None:
                    self.matches[handle] = not event["local"]

    def allocate(self, signal_id: str, handle: str) -> Optional[int]:
        is_global = self.matches.get(handle, False)
        if is_global:
            awg_key = self.signals[signal_id].awg.key
            feedback_register = self.feedback_registers.get(awg_key)
            if feedback_register is None:
                allocated = set(self.feedback_registers.values())
                feedback_register = 0 if len(allocated) == 0 else max(allocated) + 1
                self.feedback_registers[awg_key] = feedback_register
            return feedback_register
        return None
