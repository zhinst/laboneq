# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from bisect import bisect_left
from collections import Counter
from typing import TYPE_CHECKING

# reporter import is required to register the CompilationReportGenerator hook
import laboneq.compiler.workflow.reporter  # noqa: F401
from laboneq.compiler.common import compiler_settings
from laboneq.compiler.common.resource_usage import ResourceLimitationError
from laboneq.compiler.common.result_shape import construct_result_shape_info
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
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
    DeviceInfo,
    DeviceInfoType,
)
from laboneq.data.recipe import Recipe
from laboneq.data.scheduled_experiment import (
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
        self._experiment_dao: ExperimentDAO = None
        self._execution: Statement = None
        self._chunking_info: ChunkingInfo | None = None
        self._compiler_settings = settings or {}
        self._settings = compiler_settings.from_dict(settings)
        self._is_desktop_setup: bool = False

        _logger.info("Starting LabOne Q Compiler run...")

    @staticmethod
    def _get_first_instr_of(device_infos: list[DeviceInfo], type: str) -> DeviceInfo:
        return next(
            (instr for instr in device_infos if instr.device_type.value == type)
        )

    def _analyze_setup(self, experiment_dao: ExperimentDAO):
        device_infos = experiment_dao.device_infos()
        device_type_list = [i.device_type.value for i in device_infos]
        type_counter = Counter(device_type_list)
        has_pqsc = type_counter["pqsc"] > 0
        has_qhub = type_counter["qhub"] > 0
        has_hdawg = type_counter["hdawg"] > 0
        shf_types = {"shfsg", "shfqa", "shfqc"}

        # Basic validity checks
        signal_infos = [
            experiment_dao.signal_info(signal_id)
            for signal_id in experiment_dao.signals()
        ]
        used_devices = set(info.device.device_type.value for info in signal_infos)
        if (
            "hdawg" in used_devices
            and "uhfqa" in used_devices
            and bool(shf_types.intersection(used_devices))
        ):
            raise RuntimeError(
                "Setups with signals on each of HDAWG, UHFQA and SHF type "
                + "instruments are not supported"
            )

        device_infos_without_ppc = [
            d for d in device_infos if d and d.device_type != DeviceInfoType.SHFPPC
        ]
        standalone_qc = len(device_infos_without_ppc) <= 2 and all(
            dev.is_qc for dev in device_infos_without_ppc
        )
        if "zqcs" in used_devices:
            self._is_desktop_setup = True
        else:
            self._is_desktop_setup = (
                not has_pqsc
                and not has_qhub
                and (
                    used_devices == {"hdawg"}
                    or used_devices == {"shfsg"}
                    or used_devices == {"shfqa"}
                    or (used_devices == {"shfqa", "shfsg"} and standalone_qc)
                    or used_devices == {"hdawg", "uhfqa"}
                    or (used_devices == {"uhfqa"} and has_hdawg)  # No signal on leader
                )
            )
        if (
            not has_pqsc
            and not has_qhub
            and not self._is_desktop_setup
            and used_devices != {"uhfqa"}
            and bool(used_devices)  # Allow empty experiment (used in tests)
            and "zqcs" not in used_devices
        ):
            raise RuntimeError(
                f"Unsupported device combination {used_devices} for small setup"
            )

    def _compile_whole_or_with_chunks(
        self,
        experiment: compiler_rs.ExperimentInfo,
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
            result_shape_info = ResultShapeInfo({}, {}, {})
        else:
            generate_recipe_args = GenerateRecipeArgs(
                experiment_rs=experiment,
                combined_compiler_output=combined_output,
            )
            recipe = get_compiler_hooks(device_class).generate_recipe(
                generate_recipe_args
            )
            result_shape_info = construct_result_shape_info(
                self._execution,
                rt_loop_properties,
                combined_output,
                self._experiment_dao.device_infos(),
                self._settings,
                recipe.integrator_allocations,
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
        self, compiler_module: compiler_rs, device_class: int
    ) -> ScheduledExperiment:
        experiment = build_rs_experiment(
            experiment_dao=self._experiment_dao,
            compiler_module=compiler_module,
            desktop_setup=self._is_desktop_setup,
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

    def run(self, data: CompilationJob) -> ScheduledExperiment:
        _logger.debug("Start LabOne Q Compiler run")

        compiler_module, device_class = _resolve_compiler_module_device_class(data)

        self._device_setup_fingerprint = data.experiment_info.device_setup_fingerprint
        self._experiment_dao = ExperimentDAO(data.experiment_info)
        self._execution = data.execution
        self._chunking_info = data.experiment_info.chunking

        self._analyze_setup(self._experiment_dao)
        compiler_output = self._process_experiment(compiler_module, device_class)

        assert compiler_output is not None
        _logger.info("Finished LabOne Q Compiler run.")
        return compiler_output


def _resolve_compiler_module_device_class(
    job: CompilationJob,
) -> tuple[compiler_rs, int]:
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
