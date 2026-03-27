# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import math
from bisect import bisect_left
from collections import Counter
from typing import TYPE_CHECKING

# reporter import is required to register the CompilationReportGenerator hook
import laboneq.compiler.workflow.reporter  # noqa: F401
from laboneq.compiler.common import compiler_settings
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.resource_usage import ResourceLimitationError
from laboneq.compiler.common.result_shape import construct_result_shape_info
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
from laboneq.compiler.workflow.compiler_hooks import (
    GenerateRecipeArgs,
    all_compiler_hooks,
    get_compiler_hooks,
    resolve_compiler_module,
)
from laboneq.compiler.workflow.neartime_execution import (
    NtCompilerExecutor,
)
from laboneq.compiler.workflow.realtime_compiler import RealtimeCompiler
from laboneq.core.exceptions import LabOneQException
from laboneq.data.awg_info import AWGInfo, AwgKey
from laboneq.data.compilation_job import (
    ChunkingInfo,
    CompilationJob,
    DeviceInfo,
    DeviceInfoType,
    SignalInfo,
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


AWGMapping = list[AWGInfo]

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


def calc_awgs(dao: ExperimentDAO) -> AWGMapping:
    awgs: dict[AwgKey, AWGInfo] = {}
    for signal_id in dao.signals():
        signal_info = dao.signal_info(signal_id)
        device_id = signal_info.device.uid
        device_type = DeviceType.from_device_info_type(signal_info.device.device_type)
        if device_type.device_class != 0:
            continue
        for channel in sorted(signal_info.channels):
            awg_id = calc_awg_number(channel, device_type)
            key = AwgKey(device_id, awg_id)
            awg = awgs.get(key)
            if awg is None:
                awg = AWGInfo(
                    device_id=device_id,
                    awg_id=awg_id,
                    device_type=device_type,
                    signal_type=None,
                    device_class=device_type.device_class,
                    awg_allocation=[awg_id],
                )
                awgs[key] = awg

            awg.signal_channels.append((signal_id, channel))
    return list(awgs.values())


def calc_awg_number(channel, device_type: DeviceType):
    if device_type == DeviceType.UHFQA:
        return 0
    return int(math.floor(channel / device_type.channels_per_awg))


class Compiler:
    def __init__(self, settings: dict | None = None):
        self._device_setup_fingerprint: str = ""
        self._experiment_dao: ExperimentDAO = None
        self._execution: Statement = None
        self._chunking_info: ChunkingInfo | None = None
        self._settings = compiler_settings.from_dict(settings)
        self._is_desktop_setup: bool = False

        self._awgs: AWGMapping = []
        self._signal_objects: dict[str, SignalObj] = {}
        self._has_uhfqa: bool = False

        _logger.info("Starting LabOne Q Compiler run...")

    @classmethod
    def from_user_settings(cls, settings: dict) -> Compiler:
        return cls(compiler_settings.filter_user_settings(settings))

    def use_experiment(self, experiment: CompilationJob):
        if isinstance(experiment, CompilationJob):
            self._device_setup_fingerprint = (
                experiment.experiment_info.device_setup_fingerprint
            )
            self._experiment_dao = ExperimentDAO(experiment.experiment_info)
            self._execution = experiment.execution
            self._chunking_info = experiment.experiment_info.chunking
        else:
            raise ValueError("Invalid experiment format")

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
        has_uhfqa = type_counter["uhfqa"] > 0
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

        if has_hdawg and self._is_desktop_setup:
            triggering_hdawg = self._get_first_instr_of(device_infos, "hdawg").uid

            has_signal_on_awg_0_of_triggering_hdawg = False
            for signal_id in experiment_dao.signals():
                signal_info = experiment_dao.signal_info(signal_id)
                if signal_info.device.uid == triggering_hdawg and (
                    0 in signal_info.channels or 1 in signal_info.channels
                ):
                    has_signal_on_awg_0_of_triggering_hdawg = True
                    break

            if not has_signal_on_awg_0_of_triggering_hdawg:
                signal_id = "__small_system_trigger__"
                device_id = triggering_hdawg
                signal_type = "iq"
                channels = [0, 1]
                experiment_dao.add_signal(device_id, channels, signal_id, signal_type)
                _logger.debug(
                    "No pulses played on channels 1 or 2 of %s, adding dummy signal %s to ensure triggering of the setup",
                    triggering_hdawg,
                    signal_id,
                )

        self._has_uhfqa = has_uhfqa

    def _compile_whole_or_with_chunks(
        self, experiment: compiler_rs.ExperimentInfo, chunk_count: int | None
    ) -> ScheduledExperiment:
        rt_compiler = RealtimeCompiler(
            experiment,
            self._signal_objects,
            self._settings,
        )
        executor = NtCompilerExecutor(rt_compiler, self._settings, chunk_count)
        executor.run(self._execution)

        combined_compiler_output = executor.combined_compiler_output()
        assert combined_compiler_output is not None, (
            "Internal error: missing real-time compiler output"
        )

        rt_loop_properties = executor.rt_loop_properties()

        executor.finalize()

        awgs: list[AWGInfo] = sorted(self._awgs, key=lambda awg: awg.key)

        device_class, combined_output = (
            combined_compiler_output.get_first_combined_output()
        )
        if combined_output is None:
            recipe = Recipe()
            result_shape_info = ResultShapeInfo({}, {}, {})
        else:
            generate_recipe_args = GenerateRecipeArgs(
                awgs=awgs,
                experiment_dao=self._experiment_dao,
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

    def _process_experiment(self) -> ScheduledExperiment:
        dao = self._experiment_dao
        self._awgs = self._calc_awgs(dao)
        self._signal_objects = self._generate_signal_objects()
        compiler_module = resolve_compiler_module(
            {s.awg.device_class for s in self._signal_objects.values()}
        )
        experiment = build_rs_experiment(
            experiment_dao=dao,
            signal_objects=self._signal_objects,
            compiler_module=compiler_module,
            desktop_setup=self._is_desktop_setup,
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
                    experiment=experiment, chunk_count=chunk_count
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
                        experiment=experiment, chunk_count=chunk_count
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

    @staticmethod
    def _calc_awgs(dao: ExperimentDAO) -> AWGMapping:
        d: AWGMapping = []
        for compiler_hooks in all_compiler_hooks():
            d.extend(compiler_hooks.calc_awgs(dao))
        return d

    def _generate_signal_objects(self):
        signal_objects: dict[str, SignalObj] = {}
        awgs_by_signal_id = {
            signal_id: awg for awg in self._awgs for signal_id, _ in awg.signal_channels
        }

        for signal_id in self._experiment_dao.signals():
            signal_info: SignalInfo = self._experiment_dao.signal_info(signal_id)
            device_id = signal_info.device.uid

            awg = awgs_by_signal_id[signal_id]
            device_info = self._experiment_dao.device_info(device_id)

            signal_type = signal_info.type.value

            _logger.debug(
                "Adding signal %s with signal type %s", signal_id, signal_type
            )

            channels = copy.deepcopy(signal_info.channels)
            port_delay = self._experiment_dao.port_delay(signal_id)
            local_oscillator_frequency = self._experiment_dao.lo_frequency(signal_id)

            signal_obj = SignalObj(
                id=signal_id,
                signal_type=signal_type,
                awg=awg,
                channels=channels,
                channel_to_port={
                    int(c): p for c, p in signal_info.channel_to_port.items()
                },
                port_delay=port_delay or 0.0,
                is_qc=device_info.is_qc,
                automute=signal_info.automute,
                local_oscillator_frequency=local_oscillator_frequency,
                signal_range=signal_info.signal_range,
            )
            signal_objects[signal_id] = signal_obj
            awg.signals.append(signal_obj)
        return signal_objects

    def run(self, data: CompilationJob) -> ScheduledExperiment:
        _logger.debug("Start LabOne Q Compiler run")

        self.use_experiment(data)
        self._analyze_setup(self._experiment_dao)
        compiler_output = self._process_experiment()

        assert compiler_output is not None
        _logger.info("Finished LabOne Q Compiler run.")
        return compiler_output
