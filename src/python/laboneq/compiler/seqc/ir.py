# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines IR objects relevant for code generator."""

from __future__ import annotations
from attr import define

from laboneq.compiler import ir
from laboneq.compiler.common import awg_info


@define(kw_only=True, slots=True)
class SingleAwgIR(ir.IntervalIR):
    awg: awg_info.AWGInfo
