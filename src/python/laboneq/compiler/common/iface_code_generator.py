# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from abc import abstractmethod

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.iface_compiler_output import RTCompilerOutput
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.feedback_router.feedback_router import FeedbackRegisterLayout


class ICodeGenerator(abc.ABC):
    @abstractmethod
    def __init__(
        self,
        ir,
        signals: list[SignalObj],
        feedback_register_layout: FeedbackRegisterLayout,
        settings: CompilerSettings | dict | None = None,
    ): ...

    @abstractmethod
    def generate_code(self): ...

    @abstractmethod
    def get_output(self) -> RTCompilerOutput: ...
