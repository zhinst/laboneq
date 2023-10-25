# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from laboneq.compiler.ir.interval_ir import IntervalIR


class ReserveIR(IntervalIR):
    @classmethod
    def create(cls, signal, grid):
        return cls(grid=grid, signals={signal})

    def generate_event_list(self, *_, **__) -> List[Dict]:
        assert self.length is not None
        return []

    def __hash__(self):
        return super().__hash__()
