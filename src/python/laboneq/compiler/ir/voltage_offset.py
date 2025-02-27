# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from attrs import define

from laboneq.compiler.ir import IntervalIR


@define(kw_only=True, slots=True)
class InitialOffsetVoltageIR(IntervalIR):
    value: float
