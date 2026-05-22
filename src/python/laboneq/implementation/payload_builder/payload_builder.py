# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.compiler import Compiler
from laboneq.implementation.payload_builder.experiment_info_builder.experiment_info_builder import (
    ExperimentInfoBuilder,
)

if TYPE_CHECKING:
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.data.setup_description import Setup
    from laboneq.dsl.experiment import Experiment


def compile_experiment(
    device_setup: Setup,
    experiment: Experiment,
    signal_mappings: dict[str, str],
    compiler_settings: dict | None = None,
) -> ScheduledExperiment:
    experiment_info = ExperimentInfoBuilder(
        experiment, device_setup, signal_mappings
    ).load_experiment()
    return Compiler(compiler_settings).run(experiment_info)
