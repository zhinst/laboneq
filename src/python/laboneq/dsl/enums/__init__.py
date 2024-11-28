# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.core.types.enums import (
    AcquisitionType,
    AveragingMode,
    CarrierType,
    DSLVersion,
    ExecutionType,
    HighPassCompensationClearing,
    IODirection,
    IOSignalType,
    ModulationType,
    PortMode,
    RepetitionMode,
    SectionAlignment,
)


def enum_repr(self):
    cls_name = self.__class__.__name__
    return f"{cls_name}.{self.name}"


for e in [
    CarrierType,
    IODirection,
    IOSignalType,
    IODirection,
    DSLVersion,
    ExecutionType,
    AcquisitionType,
    SectionAlignment,
    AveragingMode,
    RepetitionMode,
    PortMode,
    ModulationType,
]:
    e.__repr__ = enum_repr
