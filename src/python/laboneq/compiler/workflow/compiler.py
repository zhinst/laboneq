# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING

# reporter import is required to register the CompilationReportGenerator hook
import laboneq.compiler.workflow.reporter  # noqa: F401
from laboneq.compiler.common import compiler_settings
from laboneq.compiler.workflow.compiler_hooks import (
    GenerateRecipeArgs,
    get_compiler_hooks,
    resolve_compiler_module,
)
from laboneq.compiler.workflow.neartime_execution import (
    NtCompilerExecutor,
)
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.compilation_job import (
    DeviceInfoType,
)
from laboneq.data.recipe import Recipe
from laboneq.data.scheduled_experiment import (
    HandleResultShape,
    ResultShapeInfo,
    RtLoopProperties,
    ScheduledExperiment,
)

from . import compat

if TYPE_CHECKING:
    import types

    from laboneq._rust import compiler as compiler_rs
    from laboneq.data.compilation_job import ExperimentInfo
    from laboneq.executor.executor import Statement

_logger = logging.getLogger(__name__)


class Compiler:
    def __init__(self, settings: dict | None = None):
        self._compiler_settings = settings or {}
        self._settings = compiler_settings.from_dict(settings)

        _logger.info("Starting LabOne Q Compiler run...")

    def run(self, experiment: ExperimentInfo) -> ScheduledExperiment:
        _logger.debug("Start LabOne Q Compiler run")

        compiler_module: compiler_rs = _resolve_compiler_module_device_class(experiment)
        capnp_bytes = compat.serialize_capnp(
            experiment_info=experiment,
            compiler_module=compiler_module,
        )
        scheduled_experiment: ScheduledExperiment = compiler_module.compile_experiment(
            capnp_bytes,
            packed=compat.use_packed_capnp(),
            compiler_settings=compat.sanitize_compiler_settings(
                self._compiler_settings
            ),
        )
        scheduled_experiment.device_setup_fingerprint = (
            experiment.device_setup_fingerprint
        )
        _logger.info("Finished LabOne Q Compiler run.")
        return scheduled_experiment


def _resolve_compiler_module_device_class(
    experiment: ExperimentInfo,
) -> types.ModuleType:
    device_classes = {
        _eval_device_class(info.device_type) for info in experiment.devices
    }
    if len(device_classes) > 1:
        raise RuntimeError(
            f"Multiple device classes {device_classes} found in experiment, but only one is supported"
        )
    device_class = next(iter(device_classes), 0)
    return resolve_compiler_module({device_class})


def _eval_device_class(device: DeviceInfoType) -> int:
    if device == DeviceInfoType.ZQCS:
        return 1
    return 0


def compile_whole_or_with_chunks(
    experiment: compiler_rs.Experiment,
    execution: Statement,
    chunk_count: int | None,
    device_class: int,
    compiler_settings: compiler_settings.CompilerSettings,
) -> ScheduledExperiment:
    if chunk_count == 1:
        chunk_count = None
    executor = NtCompilerExecutor(
        experiment, compiler_settings, chunk_count, device_class
    )
    executor.run(execution)

    combined_compiler_output = executor.combined_compiler_output()
    assert combined_compiler_output is not None, (
        "Internal error: missing real-time compiler output"
    )

    executor.finalize()

    combined_output = combined_compiler_output.get_first_combined_output()
    if combined_output is None:
        recipe = Recipe()
        result_shape_info = ResultShapeInfo({}, {})
    else:
        generate_recipe_args = GenerateRecipeArgs(
            experiment_rs=experiment,
            combined_compiler_output=combined_output,
        )
        recipe = get_compiler_hooks(device_class).generate_recipe(generate_recipe_args)
        shapes = {
            shape.handle: HandleResultShape(
                signal=shape.signal,
                shape=tuple(shape.shape),
                axis_names=[
                    axis_names[0] if len(axis_names) == 1 else axis_names
                    for axis_names in shape.axis_names
                ],
                axis_values=[
                    axis_values[0] if len(axis_values) == 1 else axis_values
                    for axis_values in shape.axis_values
                ],
                chunked_axis_index=shape.chunked_axis_index,
                match_case_mask=shape.match_case_mask or None,
            )
            for shape in experiment.get_result_shapes(combined_output)
        }
        result_shape_info = ResultShapeInfo(
            shapes=shapes, result_handle_maps=combined_output.result_handle_maps
        )

    rt_loop_properties = experiment.rt_loop_properties()
    return ScheduledExperiment(
        device_setup_fingerprint="",  # NOTE: This will be set later.
        recipe=recipe,
        artifacts=combined_compiler_output.get_artifacts(),
        schedule=combined_compiler_output.schedule,
        execution=execution,
        rt_loop_properties=RtLoopProperties(
            uid=rt_loop_properties.uid,
            acquisition_type=AcquisitionType[rt_loop_properties.acquisition_type],
            averaging_mode=AveragingMode[rt_loop_properties.averaging_mode],
            shots=rt_loop_properties.count,
            chunk_count=chunk_count,
        ),
        result_shape_info=result_shape_info,
    )
