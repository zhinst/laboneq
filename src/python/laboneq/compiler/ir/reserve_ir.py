# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from laboneq.compiler.ir.interval_ir import IntervalIR


class ReserveIR(IntervalIR):
    @classmethod
    def create(cls, signal, grid):
        return cls(grid=grid, signals={signal})
