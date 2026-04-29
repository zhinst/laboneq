# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common.iface_code_generator import ICodeGenerator
    from laboneq.compiler.common.iface_compiler_output import CombinedOutput
    from laboneq.compiler.common.iface_linker import ILinker
    from laboneq.data.recipe import Recipe


@dataclass
class GenerateRecipeArgs:
    experiment_rs: compiler_rs.ExperimentInfo
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
    def compiler_module() -> compiler_rs: ...


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


def resolve_compiler_module(device_classes: set[int]) -> compiler_rs:
    """Return the compiler module for the given device classes.

    Exactly one device class must be present; mixed device classes are not
    supported. Falls back to the default (SeqC) backend for empty experiments.
    """
    if not device_classes:
        device_classes = {0}
    if len(device_classes) != 1:
        raise ValueError(f"Expected exactly one device class, got {device_classes}")
    (device_class,) = device_classes
    return get_compiler_hooks(device_class).compiler_module()


@register_compiler_hooks
class CompilerHooksSeqC(CompilerHooks):
    @staticmethod
    def device_class() -> int:
        return 0

    @staticmethod
    def compiler_module() -> compiler_rs:
        from laboneq._rust import compiler

        return compiler

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
            args.experiment_rs,
            args.combined_compiler_output,
        )
