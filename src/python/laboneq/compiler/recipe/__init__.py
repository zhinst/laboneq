# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from typing_extensions import TypeAlias

from laboneq.compiler.common.awg_info import AWGInfo
from laboneq.compiler.seqc.recipe_generator import generate_recipe
from laboneq.data.compilation_job import PrecompensationInfo
from laboneq.data.recipe import Recipe

if TYPE_CHECKING:
    from laboneq.compiler.common.iface_compiler_output import CombinedOutput
    from laboneq.compiler.experiment_access import ExperimentDAO
    from laboneq.compiler.workflow.compiler import (
        LeaderProperties,
        IntegrationUnitAllocation,
    )
    from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
    from laboneq.compiler.workflow.on_device_delays import OnDeviceDelayCompensation
    from laboneq.compiler.workflow.compiler_output import (
        CombinedRTCompilerOutputContainer,
    )

    Callback: TypeAlias = Callable[
        [
            list[AWGInfo],
            ExperimentDAO,
            LeaderProperties,
            dict[str, Any],
            SamplingRateTracker,
            dict[str, IntegrationUnitAllocation],
            dict[str, OnDeviceDelayCompensation],
            dict[str, PrecompensationInfo],
            CombinedOutput,
        ],
        Recipe,
    ]

_registered_hooks: dict[int, Callback] = {}


def register_recipe_hook(device_class: int, hook: Callback):
    _registered_hooks[device_class] = hook


register_recipe_hook(0, generate_recipe)


def generate_recipe_combined(
    awgs: list[AWGInfo],
    experiment_dao: ExperimentDAO,
    leader_properties: LeaderProperties,
    clock_settings: dict[str, Any],
    sampling_rate_tracker: SamplingRateTracker,
    integration_unit_allocation: dict[str, IntegrationUnitAllocation],
    delays_by_signal: dict[str, OnDeviceDelayCompensation],
    precompensations: dict[str, PrecompensationInfo],
    combined_compiler_output: CombinedRTCompilerOutputContainer,
) -> Recipe:
    for device_class, output in combined_compiler_output.combined_output.items():
        return _registered_hooks[device_class](
            awgs,
            experiment_dao,
            leader_properties,
            clock_settings,
            sampling_rate_tracker,
            integration_unit_allocation,
            delays_by_signal,
            precompensations,
            output,
        )
    else:
        return Recipe()
