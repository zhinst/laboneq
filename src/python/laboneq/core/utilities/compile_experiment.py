# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import deprecated

from laboneq.core.types import CompiledExperiment
from laboneq.implementation.payload_builder.payload_builder import (
    compile_experiment as _compile_experiment,
)

if TYPE_CHECKING:
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment.experiment import Experiment


def compile_experiment(
    device_setup: DeviceSetup,
    experiment: Experiment,
    compiler_settings: dict | None = None,
) -> CompiledExperiment:
    """Compile a LabOne Q experiment."""
    output = _compile_experiment(
        device_setup=device_setup,
        experiment=experiment,
        compiler_settings=compiler_settings,
    )
    compiled_experiment = CompiledExperiment(
        device_setup=device_setup,
        experiment=experiment,
        compiler_settings=compiler_settings,
        _compiler_output=output,
    )
    return compiled_experiment


@deprecated(
    "`laboneq_compile()` is deprecated. Use `compile_experiment()` instead.",
    category=FutureWarning,
)
def laboneq_compile(
    device_setup: DeviceSetup,
    experiment: Experiment,
    compiler_settings: dict | None = None,
) -> CompiledExperiment:
    """Compile a LabOne Q experiment.

    !!! version-changed "Deprecated in version 26.4"
        Use `compile_experiment()` instead.
    """
    return compile_experiment(
        device_setup=device_setup,
        experiment=experiment,
        compiler_settings=compiler_settings,
    )
