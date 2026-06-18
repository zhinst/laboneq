# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common.iface_compiler_output import CombinedOutput
    from laboneq.compiler.common.iface_linker import ILinker
    from laboneq.data.compilation_job import DeviceInfo, ExperimentInfo
    from laboneq.data.recipe import Recipe


@dataclass
class GenerateRecipeArgs:
    experiment_rs: compiler_rs.ProcessedExperiment
    combined_compiler_output: CombinedOutput


@dataclass
class SetupDescriptionResult:
    """Result of `build_setup_description()`: the capnp setup description object and the
    per-signal maps_to mapping (experiment signal -> channel identifier)."""

    setup_description: SimpleNamespace
    signal_map: dict[str, str]


class CompilerHooks(ABC):
    @staticmethod
    @abstractmethod
    def device_class() -> int: ...

    @staticmethod
    @abstractmethod
    def linker() -> type[ILinker]: ...

    @staticmethod
    @abstractmethod
    def generate_recipe(args: GenerateRecipeArgs) -> Recipe: ...

    @staticmethod
    @abstractmethod
    def compiler_module() -> compiler_rs: ...

    @staticmethod
    @abstractmethod
    def build_setup_description(experiment: ExperimentInfo) -> SetupDescriptionResult:
        """Build the capnp setup description for this backend's device class.

        Converts `experiment` into the backend-specific setup representation used
        during compiler payload serialization, and returns the signal map that
        resolves experiment signal UIDs to physical channel identifiers.
        """


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


def resolve_compiler_module(device_class: int) -> compiler_rs:
    """Return the compiler module for the given device class."""
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
    def generate_recipe(args: GenerateRecipeArgs) -> Recipe:
        from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC
        from laboneq.compiler.seqc.recipe_generator import generate_recipe

        assert isinstance(args.combined_compiler_output, CombinedRTOutputSeqC)
        return generate_recipe(
            args.experiment_rs,
            args.combined_compiler_output,
        )

    @staticmethod
    def build_setup_description(experiment: ExperimentInfo) -> SetupDescriptionResult:
        signals = []
        for physical_channel in experiment.physical_channels:
            sig = SimpleNamespace(
                uid=physical_channel.uid,
                ports=physical_channel.ports,
                instrument_uid=physical_channel.device_uid,
            )
            signals.append(sig)

        return SetupDescriptionResult(
            setup_description=SimpleNamespace(
                instruments=[_to_instrument(d) for d in experiment.devices],
                signals=signals,
                internal_connections=experiment.internal_connections,
            ),
            signal_map=experiment.signal_map,
        )


def _to_instrument(device: DeviceInfo) -> SimpleNamespace:
    return SimpleNamespace(
        uid=device.uid,
        device_type=device.device_type.name,
        options=device.options.upper().split("/") if device.options else [],
        reference_clock_source=device.reference_clock_source.name
        if device.reference_clock_source
        else None,
    )
