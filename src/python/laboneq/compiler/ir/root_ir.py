# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from attrs import define
from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.core.types.enums import AcquisitionType


@define(kw_only=True, slots=True)
class RootScheduleIR(IntervalIR):
    acquisition_type: AcquisitionType | None = None
