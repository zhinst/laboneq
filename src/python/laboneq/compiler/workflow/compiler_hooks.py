# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from laboneq.compiler.common.awg_info import AWGInfo
    from laboneq.compiler.common.iface_compiler_output import CombinedOutput
    from laboneq.compiler.common.iface_code_generator import ICodeGenerator
    from laboneq.compiler.common.iface_linker import ILinker
    from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
    from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
    from laboneq.compiler.workflow.compiler import (
        AWGMapping,
        IntegrationUnitAllocation,
        LeaderProperties,
    )
    from laboneq.compiler.workflow.on_device_delays import OnDeviceDelayCompensation
    from laboneq.data.compilation_job import PrecompensationInfo
    from laboneq.data.recipe import Recipe


@dataclass
class GenerateRecipeArgs:
    awgs: list[AWGInfo]
    experiment_dao: ExperimentDAO
    leader_properties: LeaderProperties
    clock_settings: dict[str, Any]
    sampling_rate_tracker: SamplingRateTracker
    integration_unit_allocation: dict[str, IntegrationUnitAllocation]
    delays_by_signal: dict[str, OnDeviceDelayCompensation]
    precompensations: dict[str, PrecompensationInfo]
    combined_compiler_output: CombinedOutput


class CompilerHooks(ABC):
    @staticmethod
    @abstractmethod
    def device_class() -> int: ...

    @staticmethod
    @abstractmethod
    def linker() -> type[ILinker]: ...

    @staticmethod
    @abstractmethod
    def code_generator() -> type[ICodeGenerator]: ...

    @staticmethod
    @abstractmethod
    def generate_recipe(args: GenerateRecipeArgs) -> Recipe: ...

    @staticmethod
    @abstractmethod
    def calc_awgs(dao: ExperimentDAO) -> AWGMapping: ...


T = TypeVar("T", bound=CompilerHooks)

_registered_compiler_hooks: dict[int, type[CompilerHooks]] = {}


def register_compiler_hooks(hooks: type[T]) -> type[T]:
    _registered_compiler_hooks[hooks.device_class()] = hooks
    return hooks


def get_compiler_hooks(device_class: int) -> type[CompilerHooks]:
    if device_class not in _registered_compiler_hooks:
        raise ValueError(
            f"No compiler hooks registered for device class {device_class}"
        )
    return _registered_compiler_hooks[device_class]


def all_compiler_hooks() -> Iterator[type[CompilerHooks]]:
    yield from _registered_compiler_hooks.values()


@register_compiler_hooks
class CompilerHooksSeqC(CompilerHooks):
    @staticmethod
    def device_class() -> int:
        return 0

    @staticmethod
    def linker() -> type[ILinker]:
        from laboneq.compiler.seqc.linker import SeqCLinker

        return SeqCLinker

    @staticmethod
    def code_generator() -> type[ICodeGenerator]:
        from laboneq.compiler.seqc.code_generator import CodeGenerator

        return CodeGenerator

    @staticmethod
    def generate_recipe(args: GenerateRecipeArgs) -> Recipe:
        from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC
        from laboneq.compiler.seqc.recipe_generator import generate_recipe

        assert isinstance(args.combined_compiler_output, CombinedRTOutputSeqC)
        return generate_recipe(
            args.awgs,
            args.experiment_dao,
            args.leader_properties,
            args.clock_settings,
            args.sampling_rate_tracker,
            args.integration_unit_allocation,
            args.delays_by_signal,
            args.precompensations,
            args.combined_compiler_output,
        )

    @staticmethod
    def calc_awgs(dao: ExperimentDAO) -> AWGMapping:
        from laboneq.compiler.workflow.compiler import calc_awgs

        return calc_awgs(dao)
