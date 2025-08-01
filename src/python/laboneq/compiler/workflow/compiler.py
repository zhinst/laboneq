# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from bisect import bisect_left
import copy
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from laboneq.compiler.common.compiler_settings import TINYSAMPLE
from laboneq.compiler.feedback_router.feedback_router import (
    FeedbackRegisterLayout,
    calculate_feedback_register_layout,
)
from laboneq.compiler.common import compiler_settings
from laboneq.compiler.common.awg_info import AWGInfo, AwgKey
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.resource_usage import ResourceLimitationError
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.common.trigger_mode import TriggerMode
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.compiler.workflow.compiler_hooks import (
    GenerateRecipeArgs,
    all_compiler_hooks,
    get_compiler_hooks,
)
from laboneq.compiler.workflow.neartime_execution import (
    NtCompilerExecutor,
    legacy_execution_program,
)
from laboneq.compiler.workflow import on_device_delays
from laboneq.compiler.workflow.precompensation_helpers import (
    compute_precompensations_and_delays,
)
from laboneq.compiler.workflow.realtime_compiler import RealtimeCompiler
from laboneq.compiler.workflow.rt_linker import CombinedRTCompilerOutputContainer

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.core.types.enums.acquisition_type import AcquisitionType, is_spectroscopy
from laboneq.core.types.enums.mixer_type import MixerType
from laboneq.data.compilation_job import (
    ChunkingInfo,
    CompilationJob,
    DeviceInfo,
    OscillatorInfo,
    PrecompensationInfo,
    ReferenceClockSourceInfo,
    SignalInfo,
    SignalInfoType,
    DeviceInfoType,
    ParameterInfo,
)
from laboneq.data.recipe import Recipe
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.executor.executor import Statement

# reporter import is required to register the CompilationReportGenerator hook
import laboneq.compiler.workflow.reporter  # noqa: F401
import numpy as np

if TYPE_CHECKING:
    from laboneq.compiler.workflow.on_device_delays import OnDeviceDelayCompensation


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


def _adjust_awg_signal_type(awg: AWGInfo):
    occupied_channels = {sc[1] for sc in awg.signal_channels}
    if len(occupied_channels) == 2 and awg.signal_type not in [
        AWGSignalType.IQ,
        AWGSignalType.MULTI,
    ]:
        awg.signal_type = AWGSignalType.DOUBLE


def _verify_rf_signal_delays(awg: AWGInfo, dao: ExperimentDAO):
    # For each awg of a HDAWG, retrieve the delay of all of its rf_signals (for
    # playZeros and check whether they are the same:
    if awg.signal_type == AWGSignalType.IQ:
        return

    signal_ids = set(sc[0] for sc in awg.signal_channels)
    signal_delays = {
        dao.signal_info(signal_id).delay_signal or 0.0 for signal_id in signal_ids
    }
    if any(isinstance(d, ParameterInfo) for d in signal_delays):
        raise LabOneQException("Cannot sweep delay on RF channel")
    if len(signal_delays) > 1:
        delay_strings = ", ".join([f"{d * 1e9:.2f} ns" for d in signal_delays])
        raise RuntimeError(
            "Delays {" + str(delay_strings) + "} on awg "
            f"{awg.device_id}:{awg.awg_id} with signals "
            f"{signal_ids} differ."
        )


def _awg_oscs(device: DeviceInfo, awg_index: int) -> tuple[list[int], str | None]:
    if device.device_type == DeviceInfoType.UHFQA:
        return [0], None
    if device.device_type == DeviceInfoType.HDAWG:
        if "MF" in device.dev_opts:
            return list(range(awg_index * 4, awg_index * 4 + 4)), None
        else:
            return [awg_index], "Missing MF option?"
    if device.device_type == DeviceInfoType.SHFQA:
        if "LRT" in device.dev_opts:
            return list(range(6)), None
        else:
            return [0], "Missing LRT option?"
    if device.device_type == DeviceInfoType.SHFSG:
        return list(range(8)), None
    return [], None


def _allocate_oscillators(awg: AWGInfo, dao: ExperimentDAO):
    assert isinstance(awg.awg_id, int)
    available_oscs: list[int] | None = None
    opt_msg: str | None = None
    oscs: dict[str, OscillatorInfo] = {}
    for signal_id in set(sc[0] for sc in awg.signal_channels):
        signal_info = dao.signal_info(signal_id)
        if available_oscs is None:
            available_oscs, opt_msg = _awg_oscs(signal_info.device, awg.awg_id)
        if (osc_info := signal_info.oscillator) is not None:
            if osc_info.is_hardware is not True:
                continue
            if osc_info.frequency is None:
                # TODO(2K): What does None frequency represent?
                # Ignore such oscillators for now.
                continue
            if osc_info.uid not in oscs:
                oscs[osc_info.uid] = osc_info

    awg_osc_map: dict[str, tuple[int, Any]] = {}
    # Ensure stable order of oscillator ids for allocation
    for osc_uid in sorted(oscs.keys()):
        osc_info = oscs[osc_uid]
        assert available_oscs is not None
        if len(available_oscs) == 0 and len(awg_osc_map) == 1:
            known_osc, known_freq = next(iter(awg_osc_map.values()))
            if (
                isinstance(known_freq, float)
                and isinstance(osc_info.frequency, float)
                and np.isclose(known_freq, osc_info.frequency)
            ):
                # TODO(2K): This is a workaround for the case where measure and
                # acquire signals on the same QA AWG, or two RF signals on the
                # same HD AWG have different oscillators, but the same fixed
                # frequency. In principle, this shouldn't be allowed, but previous
                # code did allow it, and there are test cases relying on it.
                awg_osc_map[osc_info.uid] = (known_osc, known_freq)
                continue
        if len(available_oscs) == 0:
            msg = f"No free HW osc available for oscillator '{osc_info.uid}' on device '{awg.device_id}', AWG {awg.awg_id}."
            if opt_msg is not None:
                msg += " " + opt_msg
            raise LabOneQException(msg)
        awg_osc_map[osc_info.uid] = (available_oscs.pop(0), osc_info.frequency)
    awg.oscs = {k: v[0] for k, v in awg_osc_map.items()}


def calc_awgs(dao: ExperimentDAO) -> AWGMapping:
    awgs: dict[AwgKey, AWGInfo] = {}
    signals_by_channel_and_awg: dict[
        tuple[str, int, int], dict[str, set[str] | AWGInfo]
    ] = {}
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
                signal_type = signal_info.type.value
                # Treat "integration" signal type same as "iq" at AWG level
                if signal_type == "integration":
                    signal_type = "iq"
                awg = AWGInfo(
                    device_id=device_id,
                    signal_type=AWGSignalType(signal_type),
                    awg_id=awg_id,
                    device_type=device_type,
                    dev_type=signal_info.device.seqc_dev_type,
                    dev_opts=signal_info.device.seqc_dev_opts,
                    sampling_rate=None,
                    device_class=device_type.device_class,
                )
                awgs[key] = awg

            awg.signal_channels.append((signal_id, channel))

            if signal_info.type == SignalInfoType.IQ:
                assert isinstance(awg.awg_id, int)
                signal_channel_awg_key = (device_id, awg.awg_id, channel)
                if signal_channel_awg_key in signals_by_channel_and_awg:
                    signals_by_channel_and_awg[signal_channel_awg_key]["signals"].add(
                        signal_id
                    )
                else:
                    signals_by_channel_and_awg[signal_channel_awg_key] = {
                        "awg": awg,
                        "signals": {signal_id},
                    }

    for v in signals_by_channel_and_awg.values():
        if len(v["signals"]) > 1 and v["awg"].device_type != DeviceType.SHFQA:
            awg = v["awg"]
            awg.signal_type = AWGSignalType.MULTI
            _logger.debug("Changing signal type to multi: %s", awg)

    for awg in awgs.values():
        _adjust_awg_signal_type(awg)
        _verify_rf_signal_delays(awg, dao)
        _allocate_oscillators(awg, dao)

    return list(awgs.values())


def calc_awg_number(channel, device_type: DeviceType):
    if device_type == DeviceType.UHFQA:
        return 0
    return int(math.floor(channel / device_type.channels_per_awg))


@dataclass
class LeaderProperties:
    global_leader: str | None = None
    is_desktop_setup: bool = False
    internal_followers: list[str] = field(default_factory=list)


@dataclass
class _ShfqaGeneratorAllocation:
    device_id: str
    awg_nr: int
    channels: list[int]


@dataclass
class IntegrationUnitAllocation:
    device_id: str
    awg_nr: int
    channels: list[int]
    kernel_count: int
    has_local_bus: bool


def get_total_rounded_delay(delay, signal_id, device_type, sampling_rate):
    if delay < 0:
        raise RuntimeError(f"Negative signal delay for signal {signal_id} specified.")
    # Quantize to granularity and round ties towards zero
    samples = delay * sampling_rate
    samples_rounded = (
        math.ceil(samples / device_type.sample_multiple + 0.5) - 1
    ) * device_type.sample_multiple
    delay_rounded = samples_rounded / sampling_rate
    if abs(samples - samples_rounded) > 1:
        _logger.debug(
            "Signal delay %.2f ns of %s on a %s will be rounded to "
            + "%.2f ns, a multiple of %d samples.",
            delay * 1e9,
            signal_id,
            device_type.name,
            delay_rounded * 1e9,
            device_type.sample_multiple,
        )
    return delay_rounded


def calc_integration_unit_allocation(
    dao: ExperimentDAO,
) -> dict[str, IntegrationUnitAllocation]:
    integration_unit_allocation: dict[str, IntegrationUnitAllocation] = {}

    integration_signals: list[SignalInfo] = [
        signal_info
        for signal in dao.signals()
        if (signal_info := dao.signal_info(signal)).type == SignalInfoType.INTEGRATION
    ]

    # For alignment in feedback register, place qudits before qubits
    integration_signals.sort(key=lambda s: (s.kernel_count or 0) <= 1)

    for signal_info in integration_signals:
        device_type = DeviceType.from_device_info_type(signal_info.device.device_type)
        if device_type.device_class != 0:
            continue
        awg_nr = calc_awg_number(signal_info.channels[0], device_type)
        num_acquire_signals = len(
            [
                x
                for x in integration_unit_allocation.values()
                if x.device_id == signal_info.device.uid and x.awg_nr == awg_nr
            ]
        )
        if dao.acquisition_type == AcquisitionType.SPECTROSCOPY_PSD:
            if device_type == device_type.UHFQA:
                raise LabOneQException(
                    "`AcquisitionType` `SPECTROSCOPY_PSD` not allowed on UHFQA"
                )
        integrators_per_signal = (
            device_type.num_integration_units_per_acquire_signal
            if dao.acquisition_type
            in [
                AcquisitionType.RAW,
                AcquisitionType.INTEGRATION,
            ]
            or is_spectroscopy(dao.acquisition_type)
            else 1
        )
        integration_unit_allocation[signal_info.uid] = IntegrationUnitAllocation(
            device_id=signal_info.device.uid,
            awg_nr=awg_nr,
            channels=[
                integrators_per_signal * num_acquire_signals + i
                for i in range(integrators_per_signal)
            ],
            kernel_count=signal_info.kernel_count,
            has_local_bus=signal_info.device.is_qc,
        )
    return integration_unit_allocation


def calc_shfqa_generator_allocation(
    dao: ExperimentDAO,
) -> dict[str, _ShfqaGeneratorAllocation]:
    shfqa_generator_allocation: dict[str, _ShfqaGeneratorAllocation] = {}
    for signal_id in dao.signals():
        signal_info = dao.signal_info(signal_id)
        device_type = DeviceType.from_device_info_type(signal_info.device.device_type)
        if signal_info.type != SignalInfoType.IQ or device_type != DeviceType.SHFQA:
            continue
        _logger.debug(
            "_shfqa_generator_allocation: found SHFQA iq signal %s", signal_info
        )
        device_id = signal_info.device.uid
        awg_nr = calc_awg_number(signal_info.channels[0], device_type)
        num_generator_signals = len(
            [
                x
                for x in shfqa_generator_allocation.values()
                if x.device_id == device_id and x.awg_nr == awg_nr
            ]
        )

        shfqa_generator_allocation[signal_id] = _ShfqaGeneratorAllocation(
            device_id=device_id,
            awg_nr=awg_nr,
            channels=[num_generator_signals],
        )

    return shfqa_generator_allocation


class Compiler:
    def __init__(self, settings: dict | None = None):
        self._experiment_dao: ExperimentDAO = None
        self._execution: Statement = None
        self._chunking_info: ChunkingInfo | None = None
        self._final_chunk_count: int | None = None
        self._settings = compiler_settings.from_dict(settings)
        self._sampling_rate_tracker: SamplingRateTracker = None
        self._scheduler: Scheduler = None
        self._combined_compiler_output: CombinedRTCompilerOutputContainer = None

        self._leader_properties = LeaderProperties()
        self._clock_settings: dict[str, Any] = {}
        self._shfqa_generator_allocation: dict[str, _ShfqaGeneratorAllocation] = {}
        self._integration_unit_allocation: dict[str, IntegrationUnitAllocation] = {}
        self._awgs: AWGMapping = []
        self._delays_by_signal: dict[str, OnDeviceDelayCompensation] = {}
        self._precompensations: dict[str, PrecompensationInfo] = {}
        self._signal_objects: dict[str, SignalObj] = {}
        self._feedback_register_layout: FeedbackRegisterLayout = {}
        self._has_uhfqa: bool = False
        self._recipe = Recipe()

        _logger.info("Starting LabOne Q Compiler run...")
        self._check_tinysamples()

    @classmethod
    def from_user_settings(cls, settings: dict) -> Compiler:
        return cls(compiler_settings.filter_user_settings(settings))

    def _check_tinysamples(self):
        for t in DeviceType:
            num_tinysamples_per_sample = (1 / t.sampling_rate) / TINYSAMPLE
            delta = abs(round(num_tinysamples_per_sample) - num_tinysamples_per_sample)
            if delta > 1e-11:
                raise RuntimeError(
                    f"TINYSAMPLE is not commensurable with sampling rate of {t}, has {num_tinysamples_per_sample} tinysamples per sample, which is not an integer"
                )

    def use_experiment(self, experiment):
        if isinstance(experiment, CompilationJob):
            self._experiment_dao = ExperimentDAO(experiment.experiment_info)
            self._execution = experiment.execution
            self._chunking_info = experiment.experiment_info.chunking
        else:  # legacy JSON
            self._experiment_dao = ExperimentDAO(experiment)
            self._execution = legacy_execution_program()

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
        has_shfsg = type_counter["shfsg"] > 0
        has_shfqa = type_counter["shfqa"] > 0
        shf_types = {"shfsg", "shfqa", "shfqc"}
        has_shf = bool(shf_types.intersection(set(device_type_list)))

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
        if "prettyprinterdevice" in used_devices:
            self._leader_properties.is_desktop_setup = True
        else:
            self._leader_properties.is_desktop_setup = (
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
            and not self._leader_properties.is_desktop_setup
            and used_devices != {"uhfqa"}
            and bool(used_devices)  # Allow empty experiment (used in tests)
            and "prettyprinterdevice" not in used_devices
        ):
            raise RuntimeError(
                f"Unsupported device combination {used_devices} for small setup"
            )

        leader = experiment_dao.global_leader_device()
        if self._leader_properties.is_desktop_setup:
            if leader is None:
                if has_hdawg:
                    leader = self._get_first_instr_of(device_infos, "hdawg").uid
                elif has_shfqa:
                    leader = self._get_first_instr_of(device_infos, "shfqa").uid
                    if has_shfsg:  # SHFQC
                        self._leader_properties.internal_followers = [
                            self._get_first_instr_of(device_infos, "shfsg").uid
                        ]
                elif has_shfsg:
                    leader = self._get_first_instr_of(device_infos, "shfsg").uid

            _logger.debug("Using desktop setup configuration with leader %s", leader)

            # TODO: Check if this is needed for standalone QC, where only SG part is used
            if has_hdawg or (standalone_qc is True and has_shfsg and not has_shfqa):
                has_signal_on_awg_0_of_leader = False
                for signal_id in experiment_dao.signals():
                    signal_info = experiment_dao.signal_info(signal_id)
                    if signal_info.device.uid == leader and (
                        0 in signal_info.channels or 1 in signal_info.channels
                    ):
                        has_signal_on_awg_0_of_leader = True
                        break

                if not has_signal_on_awg_0_of_leader:
                    signal_id = "__small_system_trigger__"
                    device_id = leader
                    signal_type = "iq"
                    channels = [0, 1]
                    experiment_dao.add_signal(
                        device_id, channels, signal_id, signal_type
                    )
                    _logger.debug(
                        "No pulses played on channels 1 or 2 of %s, adding dummy signal %s to ensure triggering of the setup",
                        leader,
                        signal_id,
                    )

            is_hdawg_solo = type_counter["hdawg"] == 1 and not has_shf and not has_uhfqa
            if is_hdawg_solo:
                first_hdawg = self._get_first_instr_of(device_infos, "hdawg")
                if first_hdawg.reference_clock_source is None:
                    self._clock_settings[first_hdawg.uid] = (
                        ReferenceClockSourceInfo.INTERNAL
                    )
            else:
                if not has_hdawg and has_shfsg:  # SHFSG or SHFQC solo
                    first_shfsg = self._get_first_instr_of(device_infos, "shfsg")
                    if first_shfsg.reference_clock_source is None:
                        self._clock_settings[first_shfsg.uid] = (
                            ReferenceClockSourceInfo.INTERNAL
                        )
                if not has_hdawg and has_shfqa:  # SHFQA or SHFQC solo
                    first_shfqa = self._get_first_instr_of(device_infos, "shfqa")
                    if first_shfqa.reference_clock_source is None:
                        self._clock_settings[first_shfqa.uid] = (
                            ReferenceClockSourceInfo.INTERNAL
                        )

        self._clock_settings["use_2GHz_for_HDAWG"] = has_shf
        self._leader_properties.global_leader = leader
        self._has_uhfqa = has_uhfqa

    def _compile_whole_or_with_chunks(self, chunk_count: int | None):
        rt_compiler = RealtimeCompiler(
            self._experiment_dao,
            self._sampling_rate_tracker,
            self._signal_objects,
            self._feedback_register_layout,
            self._settings,
        )
        executor = NtCompilerExecutor(rt_compiler, self._settings, chunk_count)
        executor.run(self._execution)

        combined_compiler_output = executor.combined_compiler_output()
        if combined_compiler_output is None:
            raise LabOneQException("Experiment has no real-time averaging loop")

        executor.finalize()

        return combined_compiler_output

    def _process_experiment(self):
        dao = self._experiment_dao
        self._sampling_rate_tracker = SamplingRateTracker(dao, self._clock_settings)

        self._awgs = self._calc_awgs(dao)
        self._shfqa_generator_allocation = calc_shfqa_generator_allocation(dao)
        self._integration_unit_allocation = calc_integration_unit_allocation(dao)
        self._precompensations = compute_precompensations_and_delays(dao)
        self._delays_by_signal = self._adjust_signals_for_on_device_delays(
            signal_infos=[dao.signal_info(uid) for uid in dao.signals()],
            use_2ghz_for_hdawg=self._clock_settings["use_2GHz_for_HDAWG"],
        )
        self._signal_objects = self._generate_signal_objects()

        self._feedback_register_layout = calculate_feedback_register_layout(
            self._integration_unit_allocation
        )

        chunking = self._chunking_info
        if chunking is None or not chunking.auto:
            chunk_count = None if chunking is None else chunking.chunk_count
            try:
                self._combined_compiler_output = self._compile_whole_or_with_chunks(
                    chunk_count=chunk_count
                )
                self._final_chunk_count = chunk_count
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
                    self._combined_compiler_output = self._compile_whole_or_with_chunks(
                        chunk_count=chunk_count
                    )
                    self._final_chunk_count = chunk_count
                    _logger.info(
                        "Auto-chunked sweep divided into %s chunks", chunk_count
                    )
                    break
                except ResourceLimitationError as err:
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
                        requested=chunk_count * math.ceil(err.hint or 2),
                        candidates=divisors,
                    )

        for (
            device_class,
            output,
        ) in self._combined_compiler_output.combined_output.items():
            get_compiler_hooks(device_class).assign_feedback_registers(output)

    @staticmethod
    def _calc_awgs(dao: ExperimentDAO) -> AWGMapping:
        d: AWGMapping = []
        for compiler_hooks in all_compiler_hooks():
            d.extend(compiler_hooks.calc_awgs(dao))
        return d

    def _adjust_signals_for_on_device_delays(
        self, signal_infos: list[SignalInfo], use_2ghz_for_hdawg: bool
    ) -> dict[str, OnDeviceDelayCompensation]:
        """Adjust signals for on device delays.

        Returns:
            Signal and device port delays which are adjusted to delays on device.
        """
        signal_infos: dict[str, SignalInfo] = {info.uid: info for info in signal_infos}
        delay_from_output_router = on_device_delays.calculate_output_router_delays(
            {uid: sig.output_routing for uid, sig in signal_infos.items()}
        )
        signal_grid = {}
        for info in signal_infos.values():
            devtype = DeviceType.from_device_info_type(info.device.device_type)
            sampling_rate = (
                devtype.sampling_rate_2GHz
                if use_2ghz_for_hdawg and devtype == DeviceType.HDAWG
                else devtype.sampling_rate
            )
            initial_delay = 0
            initial_delay += (
                self._precompensations[info.uid].computed_delay_samples or 0
            )
            initial_delay += delay_from_output_router[info.uid]
            signal_grid[info.uid] = on_device_delays.OnDeviceDelayInfo(
                sampling_rate=sampling_rate,
                sample_multiple=devtype.sample_multiple,
                delay_samples=initial_delay,
            )
        compensated_values = on_device_delays.compensate_on_device_delays(signal_grid)
        for key, values in compensated_values.items():
            if signal_infos[key].device.device_type == DeviceInfoType.UHFQA:
                assert values.on_port == 0
        return compensated_values

    def _generate_signal_objects(self):
        signal_objects: dict[str, SignalObj] = {}

        @dataclass
        class DelayInfo:
            port_delay_gen: float | None = None
            delay_signal_gen: float | None = None

        delay_measure_acquire: dict[AwgKey, DelayInfo] = {}

        awgs_by_signal_id = {
            signal_id: awg for awg in self._awgs for signal_id, _ in awg.signal_channels
        }

        for signal_id in self._experiment_dao.signals():
            signal_info: SignalInfo = self._experiment_dao.signal_info(signal_id)
            delay_signal = signal_info.delay_signal

            device_type = DeviceType.from_device_info_type(
                signal_info.device.device_type
            )
            device_id = signal_info.device.uid

            sampling_rate = self._sampling_rate_tracker.sampling_rate_for_device(
                device_id
            )
            start_delay = get_lead_delay(
                self._settings,
                device_type,
                self._leader_properties.is_desktop_setup,
                self._clock_settings["use_2GHz_for_HDAWG"],
            )
            start_delay += self._delays_by_signal[signal_id].on_signal

            if delay_signal is not None:
                delay_signal = get_total_rounded_delay(
                    delay_signal, signal_id, device_type, sampling_rate
                )
            else:
                delay_signal = 0

            awg = awgs_by_signal_id[signal_id]
            awg.trigger_mode = TriggerMode.NONE
            device_info = self._experiment_dao.device_info(device_id)

            ref_clk_src = self._clock_settings.get(
                device_id, device_info.reference_clock_source
            )
            ref_clk_str: str | None = None
            if ref_clk_src == ReferenceClockSourceInfo.INTERNAL:
                ref_clk_str = "internal"
            if ref_clk_src == ReferenceClockSourceInfo.EXTERNAL:
                ref_clk_str = "external"
            awg.reference_clock_source = ref_clk_str

            if self._leader_properties.is_desktop_setup:
                awg.trigger_mode = {
                    DeviceType.HDAWG: TriggerMode.DIO_TRIGGER
                    if self._has_uhfqa
                    else TriggerMode.INTERNAL_READY_CHECK,
                    DeviceType.SHFSG: TriggerMode.INTERNAL_TRIGGER_WAIT,
                    DeviceType.SHFQA: TriggerMode.INTERNAL_TRIGGER_WAIT,
                    DeviceType.UHFQA: TriggerMode.DIO_WAIT,
                }.get(device_type, TriggerMode.NONE)
            awg.sampling_rate = sampling_rate

            signal_type = signal_info.type.value

            _logger.debug(
                "Adding signal %s with signal type %s", signal_id, signal_type
            )

            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            channels = copy.deepcopy(signal_info.channels)
            if signal_id in self._integration_unit_allocation:
                channels = copy.deepcopy(
                    self._integration_unit_allocation[signal_id].channels
                )
            elif signal_id in self._shfqa_generator_allocation:
                channels = copy.deepcopy(
                    self._shfqa_generator_allocation[signal_id].channels
                )
            hw_oscillator = (
                oscillator_info.uid
                if oscillator_info is not None and oscillator_info.is_hardware
                else None
            )

            mixer_type = MixerType.IQ
            if (
                device_type == DeviceType.UHFQA
                and oscillator_info
                and oscillator_info.is_hardware
            ):
                mixer_type = MixerType.UHFQA_ENVELOPE
            elif signal_type in ("single",):
                mixer_type = None

            port_delay = self._experiment_dao.port_delay(signal_id)
            if isinstance(port_delay, str):  # NT sweep param
                port_delay = math.nan

            local_oscillator_frequency = self._experiment_dao.lo_frequency(signal_id)

            if signal_type != "integration":
                delay_info = delay_measure_acquire.setdefault(awg.key, DelayInfo())
                delay_info.port_delay_gen = port_delay
                delay_info.delay_signal_gen = delay_signal

            signal_obj = SignalObj(
                id=signal_id,
                start_delay=start_delay,
                delay_signal=delay_signal,
                signal_type=signal_type,
                awg=awg,
                channels=channels,
                channel_to_port={
                    int(c): p for c, p in signal_info.channel_to_port.items()
                },
                port_delay=port_delay,
                mixer_type=mixer_type,
                hw_oscillator=hw_oscillator,
                is_qc=device_info.is_qc,
                automute=signal_info.automute,
                local_oscillator_frequency=local_oscillator_frequency,
                signal_range=signal_info.signal_range,
            )
            signal_objects[signal_id] = signal_obj
            awg.signals.append(signal_obj)

        for s in signal_objects.values():
            delay_info = delay_measure_acquire.get(s.awg.key, None)
            if delay_info is None:
                _logger.debug("No measurement pulse signal for acquire signal %s", s.id)
                continue
            s.base_port_delay = delay_info.port_delay_gen
            s.base_delay_signal = delay_info.delay_signal_gen
        return signal_objects

    def compiler_output(self) -> CompiledExperiment:
        return CompiledExperiment(
            experiment_dict=None,
            scheduled_experiment=ScheduledExperiment(
                recipe=self._recipe,
                artifacts=self._combined_compiler_output.get_artifacts(),
                execution=self._execution,
                schedule=self._combined_compiler_output.schedule,
                chunk_count=self._final_chunk_count,
            ),
        )

    def dump_src(self, info=False):
        for src in self.compiler_output().scheduled_experiment.src:
            if info:
                _logger.info("*** %s", src["filename"])
            else:
                _logger.debug("*** %s", src["filename"])
            for line in src["text"].splitlines():
                if info:
                    _logger.info(line)
                else:
                    _logger.debug(line)
        if info:
            _logger.info("END %s", src["filename"])
        else:
            _logger.debug("END %s", src["filename"])

    def run(self, data) -> CompiledExperiment:
        _logger.debug("ES Compiler run")

        self.use_experiment(data)
        self._analyze_setup(self._experiment_dao)
        self._process_experiment()

        awgs: list[AWGInfo] = sorted(self._awgs, key=lambda awg: awg.key)

        device_class, combined_output = (
            self._combined_compiler_output.get_first_combined_output()
        )
        if combined_output is None:
            self._recipe = Recipe()
        else:
            generate_recipe_args = GenerateRecipeArgs(
                awgs=awgs,
                experiment_dao=self._experiment_dao,
                leader_properties=self._leader_properties,
                clock_settings=self._clock_settings,
                sampling_rate_tracker=self._sampling_rate_tracker,
                integration_unit_allocation=self._integration_unit_allocation,
                delays_by_signal=self._delays_by_signal,
                precompensations=self._precompensations,
                combined_compiler_output=combined_output,
            )
            self._recipe = get_compiler_hooks(device_class).generate_recipe(
                generate_recipe_args
            )

        retval = self.compiler_output()

        _logger.info("Finished LabOne Q Compiler run.")

        return retval


def get_lead_delay(
    settings: compiler_settings.CompilerSettings,
    device_type: DeviceType,
    desktop_setup: bool,
    hdawg_uses_2GHz: bool,
):
    assert isinstance(device_type, DeviceType)
    if device_type == DeviceType.HDAWG:
        if not desktop_setup:
            if hdawg_uses_2GHz:
                return settings.HDAWG_LEAD_PQSC_2GHz
            else:
                return settings.HDAWG_LEAD_PQSC
        else:
            if hdawg_uses_2GHz:
                return settings.HDAWG_LEAD_DESKTOP_SETUP_2GHz
            else:
                return settings.HDAWG_LEAD_DESKTOP_SETUP
    if device_type == DeviceType.PRETTYPRINTERDEVICE:
        return settings.PRETTYPRINTERDEVICE_LEAD
    elif device_type == DeviceType.UHFQA:
        return settings.UHFQA_LEAD_PQSC
    elif device_type == DeviceType.SHFQA:
        return settings.SHFQA_LEAD_PQSC
    elif device_type == DeviceType.SHFSG:
        return settings.SHFSG_LEAD_PQSC
    else:
        raise RuntimeError(f"Unsupported device type {device_type}")
