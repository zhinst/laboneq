# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.application_management.application_manager import ApplicationManager
from laboneq.core.types import CompiledExperiment
from laboneq.implementation.legacy_adapters.converters_experiment_description import (
    convert_Experiment,
    convert_signal_map,
)
from laboneq.implementation.legacy_adapters.device_setup_converter import (
    convert_device_setup_to_setup,
)

if TYPE_CHECKING:
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment.experiment import Experiment


def laboneq_compile(
    device_setup: DeviceSetup,
    experiment: Experiment,
    compiler_settings: dict | None = None,
) -> CompiledExperiment:
    new_setup = convert_device_setup_to_setup(device_setup)
    new_experiment = convert_Experiment(experiment)
    signal_mapping = convert_signal_map(experiment)

    payload_builder = ApplicationManager.instance().payload_builder()
    payload = payload_builder.build_payload(
        new_setup,
        new_experiment,
        signal_mapping,
        compiler_settings,
    )

    compiled_experiment = CompiledExperiment(
        device_setup=device_setup,
        experiment=experiment,
        compiler_settings=compiler_settings,
        experiment_dict=None,  # deprecated
        scheduled_experiment=payload.scheduled_experiment,
    )
    return compiled_experiment
