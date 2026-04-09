# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common.iface_compiler_output import RTCompilerOutput


class ICodeGenerator(abc.ABC):
    @abstractmethod
    def __init__(
        self,
        experiment_ir: compiler_rs.ExperimentIr,
    ): ...

    @abstractmethod
    def generate_code(self): ...

    @abstractmethod
    def get_output(self) -> RTCompilerOutput: ...
