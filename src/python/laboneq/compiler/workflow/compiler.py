# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import types
from bisect import bisect_left
from typing import TYPE_CHECKING

# reporter import is required to register the CompilationReportGenerator hook
import laboneq.compiler.workflow.reporter  # noqa: F401
from laboneq.compiler.common import compiler_settings
from laboneq.compiler.common.resource_usage import ResourceLimitationError
from laboneq.compiler.workflow.compiler_hooks import (
    GenerateRecipeArgs,
    get_compiler_hooks,
    resolve_compiler_module,
)
from laboneq.compiler.workflow.neartime_execution import (
    NtCompilerExecutor,
)
from laboneq.compiler.workflow.realtime_compiler import RealtimeCompiler
from laboneq.core.exceptions import LabOneQException
from laboneq.data.compilation_job import (
    ChunkingInfo,
    CompilationJob,
    DeviceInfoType,
    ExperimentInfo,
)
from laboneq.data.recipe import Recipe
from laboneq.data.scheduled_experiment import (
    HandleResultShape,
    ResultShapeInfo,
    ScheduledExperiment,
)
from laboneq.executor.executor import Statement

from .compat import build_rs_experiment

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs

_logger = logging.getLogger(__name__)


def _divisors(n: int) -> list[int]:
    """Return all integer divisors of n."""
    assert n > 0
    return [i for i in range(1, n // 2 + 1) if n % i == 0] + [n]


def _chunk_count_trial(requested: int, candidates: list[int]) -> int:
    """Return valid chunk count, close to the requested one but not less than it.

    Args:
        requested: Desired chunk count.
        candidates: Sorted list of possible chunk counts.
    """
    idx = bisect_left(candidates, requested)
    return candidates[min(idx, len(candidates) - 1)]


class Compiler:
    def __init__(self, settings: dict | None = None):
        self._device_setup_fingerprint: str = ""
        self._execution: Statement = None
        self._chunking_info: ChunkingInfo | None = None
        self._compiler_settings = settings or {}
        self._settings = compiler_settings.from_dict(settings)

        _logger.info("Starting LabOne Q Compiler run...")

    def _compile_whole_or_with_chunks(
        self,
        experiment: compiler_rs.Experiment,
        chunk_count: int | None,
        device_class: int,
        compiler_module: compiler_rs,
    ) -> ScheduledExperiment:
        rt_compiler = RealtimeCompiler(
            experiment,
            compiler_module,
            device_class,
            self._settings,
        )
        if chunk_count == 1:
            chunk_count = None
        executor = NtCompilerExecutor(rt_compiler, self._settings, chunk_count)
        executor.run(self._execution)

        combined_compiler_output = executor.combined_compiler_output()
        assert combined_compiler_output is not None, (
            "Internal error: missing real-time compiler output"
        )

        rt_loop_properties = executor.rt_loop_properties()

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
            recipe = get_compiler_hooks(device_class).generate_recipe(
                generate_recipe_args
            )
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

        return ScheduledExperiment(
            device_setup_fingerprint=self._device_setup_fingerprint,
            recipe=recipe,
            artifacts=combined_compiler_output.get_artifacts(),
            schedule=combined_compiler_output.schedule,
            execution=self._execution,
            rt_loop_properties=rt_loop_properties,
            result_shape_info=result_shape_info,
        )

    def _process_experiment(
        self,
        experiment_info: ExperimentInfo,
        compiler_module: compiler_rs,
        device_class: int,
    ) -> ScheduledExperiment:
        experiment = build_rs_experiment(
            experiment_info=experiment_info,
            compiler_module=compiler_module,
            compiler_settings=self._compiler_settings,
        )
        chunking = self._chunking_info
        if chunking is None or not chunking.auto:
            if chunking is None:
                chunk_count = None
            else:
                chunk_count = chunking.chunk_count

                if chunk_count > chunking.sweep_iterations:
                    _logger.warning(
                        "Provided chunk count (%s) is larger than the sweep length (%s). Using %s instead.",
                        chunk_count,
                        chunking.sweep_iterations,
                        chunking.sweep_iterations,
                    )
                    chunk_count = chunking.sweep_iterations
                if chunking.sweep_iterations % chunk_count != 0:
                    raise LabOneQException(
                        f"Chunk count ({chunk_count}) does not evenly divide sweep length ({chunking.sweep_iterations})"
                    )

            try:
                return self._compile_whole_or_with_chunks(
                    experiment=experiment,
                    chunk_count=chunk_count,
                    device_class=device_class,
                    compiler_module=compiler_module,
                )
            except ResourceLimitationError as err:
                msg = (
                    "Compilation error - resource limitation exceeded.\n"
                    "To circumvent this, try one or more of the following:\n"
                    "- Double check the integrity of your experiment (look for unexpectedly long pulses, large number of sweep steps, etc.)\n"
                    "- Reduce the number of sweep steps\n"
                    "- Reduce the number of variations in the pulses that are being played\n"
                    "- Enable chunking for a sweep\n"
                    "- If chunking is already enabled, increase the chunk count or switch to automatic chunking"
                )
                raise LabOneQException(msg) from err
        else:  # auto chunking
            divisors = _divisors(chunking.sweep_iterations)
            chunk_count = _chunk_count_trial(
                chunking.chunk_count, divisors
            )  # initial guess by user
            while True:
                try:
                    _logger.debug("Attempting to compile with %s chunks", chunk_count)
                    compiler_output = self._compile_whole_or_with_chunks(
                        experiment=experiment,
                        chunk_count=chunk_count,
                        device_class=device_class,
                        compiler_module=compiler_module,
                    )
                    _logger.info(
                        "Auto-chunked sweep divided into %s chunks", chunk_count
                    )
                    return compiler_output
                except ResourceLimitationError as err:  # noqa: PERF203
                    _logger.debug(
                        "The attempt to compile with %s chunks failed with %s",
                        chunk_count,
                        err,
                    )
                    if chunk_count == chunking.sweep_iterations:
                        msg = (
                            "Automatic chunking was not able to find a chunk count to circumvent resource limitations.\n"
                            "This means that one iteration of a sweep is too large and cannot be executed.\n"
                            "To circumvent this, try one or more of the following:\n"
                            "- Chunking another sweep (e.g. in case of nested sweeps, enable chunking for the inner one)\n"
                            "- Find ways suitable for your use case to reduce the size of the program in one iteration\n"
                        )
                        raise LabOneQException(msg) from err
                    chunk_count = _chunk_count_trial(
                        requested=chunk_count * math.ceil(err.usage or 2),
                        candidates=divisors,
                    )

    def run(self, job: CompilationJob) -> ScheduledExperiment:
        _logger.debug("Start LabOne Q Compiler run")

        compiler_module, device_class = _resolve_compiler_module_device_class(job)

        self._device_setup_fingerprint = job.experiment_info.device_setup_fingerprint
        self._execution = job.execution
        self._chunking_info = job.experiment_info.chunking

        compiler_output = self._process_experiment(
            job.experiment_info, compiler_module, device_class
        )

        assert compiler_output is not None
        _logger.info("Finished LabOne Q Compiler run.")
        return compiler_output


def _resolve_compiler_module_device_class(
    job: CompilationJob,
) -> tuple[types.ModuleType, int]:
    device_classes = {
        _eval_device_class(info.device_type) for info in job.experiment_info.devices
    }
    if len(device_classes) > 1:
        raise RuntimeError(
            f"Multiple device classes {device_classes} found in experiment, but only one is supported"
        )
    device_class = next(iter(device_classes), 0)
    return resolve_compiler_module({device_class}), device_class


def _eval_device_class(device: DeviceInfoType) -> int:
    if device == DeviceInfoType.ZQCS:
        return 1
    return 0
