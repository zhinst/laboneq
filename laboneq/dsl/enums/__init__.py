# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.core.types.enums import CarrierType
from laboneq.core.types.enums import IODirection
from laboneq.core.types.enums import IOSignalType
from laboneq.core.types.enums import ModulationType
from laboneq.core.types.enums import DSLVersion
from laboneq.core.types.enums import ExecutionType
from laboneq.core.types.enums import PulseType
from laboneq.core.types.enums import AcquisitionType
from laboneq.core.types.enums import SectionAlignment
from laboneq.core.types.enums import AveragingMode
from laboneq.core.types.enums import RepetitionMode
from laboneq.core.types.enums import PortMode


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
    PulseType,
    AcquisitionType,
    SectionAlignment,
    AveragingMode,
    RepetitionMode,
    PortMode,
    ModulationType,
]:
    e.__repr__ = enum_repr
