# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Literal, Optional

from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.common.signal_obj import SignalObj

PQSC_FEEDBACK_REGISTER_COUNT = 32


class FeedbackRegisterAllocator:
    """Allocate a feedback register on the PQSC.

    Each QA AWG can write to at most one feedback register. The feedback register
    corresponds to the `result_address` in the `startQA` command and is zero based.
    The bits in the register are assigned by the instrument following the integrator
    order (the actual bit field layout will be elaborated later).

    `PQSC_FEEDBACK_REGISTER_COUNT` registers are available in total."""

    def __init__(self, signals: Dict[str, SignalObj]):
        self.signals = signals
        self.feedback_path: Dict[str, bool] = {}  # handle -> is_global
        self.target_feedback_registers: Dict[
            AwgKey, int | Literal["local"]
        ] = {}  # QA AWG -> allocated reg

        self._top = 0  # next free register

    def set_feedback_path(self, handle: str, via_pqsc: bool) -> None:
        self.feedback_path[handle] = via_pqsc

    def allocate(self, signal_id: str, handle: str) -> Optional[int | Literal["local"]]:
        is_global = self.feedback_path.get(handle)
        if is_global is None:
            return None
        awg_key = self.signals[signal_id].awg.key
        if is_global:
            feedback_register = self.target_feedback_registers.get(awg_key)
            if feedback_register is None:
                feedback_register = self._top
                self._top += 1
                if feedback_register >= PQSC_FEEDBACK_REGISTER_COUNT:
                    raise RuntimeError(
                        "Cannot allocate feedback register. "
                        f"All {PQSC_FEEDBACK_REGISTER_COUNT} registers "
                        "of the PQSC are already allocated."
                    )
                self.target_feedback_registers[awg_key] = feedback_register
            return feedback_register
        else:
            if awg_key not in self.target_feedback_registers:
                self.target_feedback_registers[awg_key] = "local"
        return None
