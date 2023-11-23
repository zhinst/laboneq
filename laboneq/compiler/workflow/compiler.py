# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from functools import singledispatchmethod
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Union, TYPE_CHECKING

from sortedcollections import SortedDict

from laboneq._observability.tracing import trace
from laboneq.compiler.code_generator.measurement_calculator import (
    IntegrationTimes,
    SignalDelays,
)
from laboneq.compiler.common import compiler_settings
from laboneq.compiler.common.awg_info import AWGInfo, AwgKey
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import (
    DeviceType,
    validate_local_oscillator_frequency,
)
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.common.trigger_mode import TriggerMode
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
from laboneq.compiler.feedback_router.feedback_router import compute_feedback_routing
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.compiler.workflow import rt_linker
from laboneq.compiler.workflow.compiler_output import (
    CombinedRealtimeCompilerOutputCode,
    CombinedRealtimeCompilerOutputPrettyPrinter,
)
from laboneq.compiler.workflow.neartime_execution import (
    NtCompilerExecutor,
    legacy_execution_program,
)
from laboneq.compiler.workflow import on_device_delays
from laboneq.compiler.workflow.precompensation_helpers import (
    compute_precompensations_and_delays,
    precompensation_is_nonzero,
    verify_precompensation_parameters,
)
from laboneq.compiler.workflow.realtime_compiler import RealtimeCompiler
from laboneq.compiler.workflow.recipe_generator import RecipeGenerator
from laboneq.compiler.workflow.rt_linker import CombinedRealtimeCompilerOutput
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.core.types.enums.acquisition_type import AcquisitionType, is_spectroscopy
from laboneq.core.types.enums.mixer_type import MixerType
from laboneq.data.calibration import PortMode
from laboneq.data.compilation_job import (
    CompilationJob,
    DeviceInfo,
    ExperimentInfo,
    ParameterInfo,
    PrecompensationInfo,
    SignalInfo,
    SignalInfoType,
    DeviceInfoType,
)
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.executor.executor import Statement

if TYPE_CHECKING:
    from laboneq.compiler.workflow.on_device_delays import OnDeviceDelayCompensation

_logger = logging.getLogger(__name__)


@dataclass
class LeaderProperties:
    global_leader: str | None = None
    is_desktop_setup: bool = False
    internal_followers: List[str] = field(default_factory=list)


_AWGMapping = Dict[str, Dict[int, AWGInfo]]


class Compiler:
    def __init__(self, settings: dict | None = None):
        self._experiment_dao: ExperimentDAO = None
        self._experiment_info: ExperimentInfo = None
        self._execution: Statement = None
        self._settings = compiler_settings.from_dict(settings)
        self._sampling_rate_tracker: SamplingRateTracker = None
        self._scheduler: Scheduler = None
        self._combined_compiler_output: CombinedRealtimeCompilerOutput = None

        self._leader_properties = LeaderProperties()
        self._clock_settings: Dict[str, Any] = {}
        self._integration_unit_allocation = None
        self._shfqa_generator_allocation = None
        self._awgs: _AWGMapping = {}
        self._delays_by_signal: dict[str, OnDeviceDelayCompensation] = {}
        self._precompensations: dict[str, PrecompensationInfo] | None = None
        self._signal_objects: Dict[str, SignalObj] = {}

        _logger.info("Starting LabOne Q Compiler run...")
        self._check_tinysamples()

    @classmethod
    def from_user_settings(cls, settings: dict) -> "Compiler":
        return cls(compiler_settings.filter_user_settings(settings))

    def _check_tinysamples(self):
        for t in DeviceType:
            num_tinysamples_per_sample = (
                1 / t.sampling_rate
            ) / self._settings.TINYSAMPLE
            delta = abs(round(num_tinysamples_per_sample) - num_tinysamples_per_sample)
            if delta > 1e-11:
                raise RuntimeError(
                    f"TINYSAMPLE is not commensurable with sampling rate of {t}, has {num_tinysamples_per_sample} tinysamples per sample, which is not an integer"
                )

    def use_experiment(self, experiment):
        if isinstance(experiment, CompilationJob):
            self._experiment_dao = ExperimentDAO(experiment.experiment_info)
            self._experiment_info = experiment.experiment_info
            self._execution = experiment.execution
        else:  # legacy JSON
            self._experiment_dao = ExperimentDAO(experiment)
            # NOTE(mr) work around for legacy JSON tests
            self._experiment_info = self._experiment_dao.to_experiment_info()
            self._execution = legacy_execution_program()

    @staticmethod
    def _get_first_instr_of(device_infos: List[DeviceInfo], type: str) -> DeviceInfo:
        return next(
            (instr for instr in device_infos if instr.device_type.value == type)
        )

    def _analyze_setup(self):
        device_infos = self._experiment_dao.device_infos()
        device_type_list = [i.device_type.value for i in device_infos]
        type_counter = Counter(device_type_list)
        has_pqsc = type_counter["pqsc"] > 0
        has_hdawg = type_counter["hdawg"] > 0
        has_shfsg = type_counter["shfsg"] > 0
        has_shfqa = type_counter["shfqa"] > 0
        shf_types = {"shfsg", "shfqa", "shfqc"}
        has_shf = bool(shf_types.intersection(set(device_type_list)))

        # Basic validity checks
        signal_infos = [
            self._experiment_dao.signal_info(signal_id)
            for signal_id in self._experiment_dao.signals()
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

        standalone_qc = len(self._experiment_dao.devices()) <= 2 and all(
            dev.is_qc for dev in device_infos
        )
        self._leader_properties.is_desktop_setup = not has_pqsc and (
            used_devices == {"hdawg"}
            or used_devices == {"prettyprinterdevice"}
            or used_devices == {"shfsg"}
            or used_devices == {"shfqa"}
            or used_devices == {"shfqa", "shfsg"}
            or standalone_qc
            or used_devices == {"hdawg", "uhfqa"}
            or (used_devices == {"uhfqa"} and has_hdawg)  # No signal on leader
        )
        if (
            not has_pqsc
            and not self._leader_properties.is_desktop_setup
            and used_devices != {"uhfqa"}
            and bool(used_devices)  # Allow empty experiment (used in tests)
            and "prettyprinterdevice" not in used_devices
        ):
            raise RuntimeError(
                f"Unsupported device combination {used_devices} for small setup"
            )

        leader = self._experiment_dao.global_leader_device()
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

            if has_hdawg or has_shfsg and not has_shfqa:
                has_signal_on_awg_0_of_leader = False
                for signal_id in self._experiment_dao.signals():
                    signal_info = self._experiment_dao.signal_info(signal_id)
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
                    self._experiment_dao.add_signal(
                        device_id, channels, signal_id, signal_type
                    )
                    _logger.debug(
                        "No pulses played on channels 1 or 2 of %s, adding dummy signal %s to ensure triggering of the setup",
                        leader,
                        signal_id,
                    )

            has_qa = type_counter["shfqa"] > 0 or type_counter["uhfqa"] > 0
            is_hdawg_solo = type_counter["hdawg"] == 1 and not has_shf and not has_qa
            if is_hdawg_solo:
                first_hdawg = self._get_first_instr_of(device_infos, "hdawg")
                if first_hdawg.reference_clock_source is None:
                    self._clock_settings[first_hdawg.uid] = "internal"
            else:
                if not has_hdawg and has_shfsg:  # SHFSG or SHFQC solo
                    first_shfsg = self._get_first_instr_of(device_infos, "shfsg")
                    if first_shfsg.reference_clock_source is None:
                        self._clock_settings[first_shfsg.uid] = "internal"
                if not has_hdawg and has_shfqa:  # SHFQA or SHFQC solo
                    first_shfqa = self._get_first_instr_of(device_infos, "shfqa")
                    if first_shfqa.reference_clock_source is None:
                        self._clock_settings[first_shfqa.uid] = "internal"

        self._clock_settings["use_2GHz_for_HDAWG"] = has_shf
        self._leader_properties.global_leader = leader

    def _process_experiment(self):
        dao = self._experiment_dao
        self._sampling_rate_tracker = SamplingRateTracker(dao, self._clock_settings)

        self._awgs = self._calc_awgs(dao)
        self._shfqa_generator_allocation = self._calc_shfqa_generator_allocation(dao)
        self._integration_unit_allocation = self._calc_integration_unit_allocation(dao)
        self._precompensations = compute_precompensations_and_delays(dao)
        self._delays_by_signal = self._adjust_signals_for_on_device_delays(
            signal_infos=[dao.signal_info(uid) for uid in dao.signals()],
            use_2ghz_for_hdawg=self._clock_settings["use_2GHz_for_HDAWG"],
        )
        self._signal_objects = self._generate_signal_objects()

        rt_compiler = RealtimeCompiler(
            self._experiment_dao,
            Scheduler(
                self._experiment_dao,
                self._sampling_rate_tracker,
                self._signal_objects,
                self._settings,
            ),
            self._sampling_rate_tracker,
            self._signal_objects,
            self._settings,
        )
        executor = NtCompilerExecutor(rt_compiler)
        executor.run(self._execution)
        self._combined_compiler_output = executor.combined_compiler_output()
        if self._combined_compiler_output is None:
            # Some of our tests do not have an RT averaging loop, so the RT compiler will
            # not have been run. For backwards compatibility, we still run it once.
            _logger.warning("Experiment has no real-time averaging loop")
            rt_compiler_output = rt_compiler.run()

            self._combined_compiler_output = rt_linker.from_single_run(
                rt_compiler_output, [0]
            )

        compute_feedback_routing(
            signal_objs=self._signal_objects,
            integration_unit_allocation=self._integration_unit_allocation,
            combined_compiler_output=self._combined_compiler_output,
        )

        if self._settings.LOG_REPORT:
            executor.report()

    @staticmethod
    def _get_total_rounded_delay(delay, signal_id, device_type, sampling_rate):
        if delay < 0:
            raise RuntimeError(
                f"Negative signal delay for signal {signal_id} specified."
            )
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

    @staticmethod
    def _calc_integration_unit_allocation(dao: ExperimentDAO):
        integration_unit_allocation: dict[str, dict] = {}

        integration_signals = [
            signal_info
            for signal in dao.signals()
            if (signal_info := dao.signal_info(signal)).type
            == SignalInfoType.INTEGRATION
        ]

        # For alignment in feedback register, place qudits before qubits
        integration_signals.sort(key=lambda s: (s.kernel_count or 0) <= 1)

        for signal_info in integration_signals:
            device_type = DeviceType.from_device_info_type(
                signal_info.device.device_type
            )
            awg_nr = Compiler.calc_awg_number(signal_info.channels[0], device_type)
            num_acquire_signals = len(
                [
                    x
                    for x in integration_unit_allocation.values()
                    if x["device_id"] == signal_info.device.uid
                    and x["awg_nr"] == awg_nr
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

            integration_unit_allocation[signal_info.uid] = {
                "device_id": signal_info.device.uid,
                "awg_nr": awg_nr,
                "channels": [
                    integrators_per_signal * num_acquire_signals + i
                    for i in range(integrators_per_signal)
                ],
                "is_msd": (signal_info.kernel_count or 0) > 1,
            }
        return integration_unit_allocation

    @staticmethod
    def _calc_shfqa_generator_allocation(dao: ExperimentDAO):
        shfqa_generator_allocation: dict[str, dict] = {}
        for signal_id in dao.signals():
            signal_info = dao.signal_info(signal_id)
            device_type = DeviceType.from_device_info_type(
                signal_info.device.device_type
            )
            if signal_info.type != SignalInfoType.IQ or device_type != DeviceType.SHFQA:
                continue
            _logger.debug(
                "_shfqa_generator_allocation: found SHFQA iq signal %s", signal_info
            )
            device_id = signal_info.device.uid
            awg_nr = Compiler.calc_awg_number(signal_info.channels[0], device_type)
            num_generator_signals = len(
                [
                    x
                    for x in shfqa_generator_allocation.values()
                    if x["device_id"] == device_id and x["awg_nr"] == awg_nr
                ]
            )

            shfqa_generator_allocation[signal_id] = {
                "device_id": device_id,
                "awg_nr": awg_nr,
                "channels": [num_generator_signals],
            }

        return shfqa_generator_allocation

    @staticmethod
    def calc_awg_number(channel, device_type: DeviceType):
        if device_type == DeviceType.UHFQA:
            return 0
        return int(math.floor(channel / device_type.channels_per_awg))

    @staticmethod
    def _calc_awgs(dao: ExperimentDAO):
        awgs: _AWGMapping = {}
        signals_by_channel_and_awg: Dict[
            Tuple[str, int, int], Dict[str, Union[Set, AWGInfo]]
        ] = {}
        for signal_id in dao.signals():
            signal_info = dao.signal_info(signal_id)
            device_id = signal_info.device.uid
            device_type = DeviceType.from_device_info_type(
                signal_info.device.device_type
            )
            for channel in sorted(signal_info.channels):
                awg_number = Compiler.calc_awg_number(channel, device_type)
                device_awgs = awgs.setdefault(device_id, SortedDict())
                awg = device_awgs.get(awg_number)
                if awg is None:
                    signal_type = signal_info.type.value
                    # Treat "integration" signal type same as "iq" at AWG level
                    if signal_type == "integration":
                        signal_type = "iq"
                    awg = AWGInfo(
                        device_id=device_id,
                        signal_type=AWGSignalType(signal_type),
                        awg_number=awg_number,
                        device_type=device_type,
                        sampling_rate=None,
                        device_class=device_type.device_class,
                    )
                    device_awgs[awg_number] = awg

                awg.signal_channels.append((signal_id, channel))

                if signal_info.type == SignalInfoType.IQ:
                    signal_channel_awg_key = (device_id, awg.awg_number, channel)
                    if signal_channel_awg_key in signals_by_channel_and_awg:
                        signals_by_channel_and_awg[signal_channel_awg_key][
                            "signals"
                        ].add(signal_id)
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

        for dev_awgs in awgs.values():
            for awg in dev_awgs.values():
                if len(awg.signal_channels) > 1 and awg.signal_type not in [
                    AWGSignalType.IQ,
                    AWGSignalType.MULTI,
                ]:
                    awg.signal_type = AWGSignalType.DOUBLE

                # For each awg of a HDAWG, retrieve the delay of all of its rf_signals (for
                # playZeros and check whether they are the same:
                if awg.signal_type == AWGSignalType.IQ:
                    continue
                signal_ids = set(sc[0] for sc in awg.signal_channels)
                signal_delays = {
                    dao.signal_info(signal_id).delay_signal or 0.0
                    for signal_id in signal_ids
                }
                if len(signal_delays) > 1:
                    delay_strings = ", ".join(
                        [f"{d * 1e9:.2f} ns" for d in signal_delays]
                    )
                    raise RuntimeError(
                        "Delays {" + str(delay_strings) + "} on awg "
                        f"{awg.device_id}:{awg.awg_number} with signals "
                        f"{signal_ids} differ."
                    )

        return awgs

    def get_awg(self, signal_id) -> AWGInfo:
        signal_info = self._experiment_dao.signal_info(signal_id)

        device_id = signal_info.device.uid
        device_type = DeviceType.from_device_info_type(signal_info.device.device_type)
        awg_number = Compiler.calc_awg_number(signal_info.channels[0], device_type)
        if (
            signal_info.type == SignalInfoType.INTEGRATION
            and device_type != DeviceType.SHFQA
        ):
            awg_number = 0
        return self._awgs[device_id][awg_number]

    def _adjust_signals_for_on_device_delays(
        self, signal_infos: list[SignalInfo], use_2ghz_for_hdawg: bool
    ) -> dict[str, OnDeviceDelayCompensation]:
        """Adjust signals for on device delays.

        Returns:
            Signal and device port delays which are adjusted to delays on device.
        """
        signal_infos: list[SignalInfo] = {info.uid: info for info in signal_infos}
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

        delay_measure_acquire: Dict[AwgKey, DelayInfo] = {}

        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
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
                delay_signal = self._get_total_rounded_delay(
                    delay_signal, signal_id, device_type, sampling_rate
                )
            else:
                delay_signal = 0

            awg = self.get_awg(signal_id)
            awg.trigger_mode = TriggerMode.NONE
            device_info = self._experiment_dao.device_info(device_id)
            try:
                awg.reference_clock_source = self._clock_settings[device_id]
            except KeyError:
                awg.reference_clock_source = device_info.reference_clock_source
            if self._leader_properties.is_desktop_setup:
                awg.trigger_mode = {
                    DeviceType.HDAWG: TriggerMode.DIO_TRIGGER,
                    DeviceType.SHFSG: TriggerMode.INTERNAL_TRIGGER_WAIT,
                    DeviceType.SHFQA: TriggerMode.INTERNAL_TRIGGER_WAIT,
                    DeviceType.UHFQA: TriggerMode.DIO_WAIT,
                }.get(device_type, TriggerMode.NONE)
            awg.sampling_rate = sampling_rate

            signal_type = signal_info.type.value

            _logger.debug(
                "Adding signal %s with signal type %s", signal_id, signal_type
            )

            oscillator_frequency = None

            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            if oscillator_info is not None and not oscillator_info.is_hardware:
                oscillator_frequency = oscillator_info.frequency
            channels = copy.deepcopy(signal_info.channels)
            if signal_id in self._integration_unit_allocation:
                channels = copy.deepcopy(
                    self._integration_unit_allocation[signal_id]["channels"]
                )
            elif signal_id in self._shfqa_generator_allocation:
                channels = copy.deepcopy(
                    self._shfqa_generator_allocation[signal_id]["channels"]
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
                oscillator_frequency=oscillator_frequency,
                channels=channels,
                port_delay=port_delay,
                mixer_type=mixer_type,
                hw_oscillator=hw_oscillator,
                is_qc=device_info.is_qc,
            )
            signal_objects[signal_id] = signal_obj
        for s in signal_objects.values():
            delay_info = delay_measure_acquire.get(s.awg.key, None)
            if delay_info is None:
                _logger.debug("No measurement pulse signal for acquire signal %s", s.id)
                continue
            s.base_port_delay = delay_info.port_delay_gen
            s.base_delay_signal = delay_info.delay_signal_gen
        return signal_objects

    def calc_outputs(self, signal_delays: SignalDelays):
        all_channels = {}

        flipper = [1, 0]

        for signal_id in self._experiment_dao.signals():
            signal_info: SignalInfo = self._experiment_dao.signal_info(signal_id)
            if signal_info.type == SignalInfoType.INTEGRATION:
                continue
            oscillator_frequency = None

            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            oscillator_is_hardware = (
                oscillator_info is not None and oscillator_info.is_hardware
            )
            if oscillator_is_hardware:
                oscillator_frequency = oscillator_info.frequency

            voltage_offset = self._experiment_dao.voltage_offset(signal_id)
            mixer_calibration = self._experiment_dao.mixer_calibration(signal_id)
            lo_frequency = self._experiment_dao.lo_frequency(signal_id)
            port_mode = self._experiment_dao.port_mode(signal_id)
            signal_range = self._experiment_dao.signal_range(signal_id)
            device_type = DeviceType.from_device_info_type(
                signal_info.device.device_type
            )
            port_delay = self._experiment_dao.port_delay(signal_id)

            scheduler_port_delay: float = 0.0
            if signal_id in signal_delays:
                scheduler_port_delay += signal_delays[signal_id].on_device
            scheduler_port_delay += self._delays_by_signal[signal_id].on_port

            base_channel = min(signal_info.channels)

            markers = self._experiment_dao.markers_on_signal(signal_id)

            triggers = self._experiment_dao.triggers_on_signal(signal_id)
            if (
                lo_frequency is not None
                and not isinstance(lo_frequency, ParameterInfo)
                and port_mode != PortMode.LF
            ):
                # TODO(2K): This validation had to be implemented in the controller
                # to support swept lo_frequency
                try:
                    validate_local_oscillator_frequency(lo_frequency, device_type)
                except ValueError as error:
                    raise LabOneQException(
                        f"Error on signal line '{signal_id}': {error}"
                    ) from error
            precompensation = self._precompensations[signal_id]
            for channel in signal_info.channels:
                output = {
                    "device_id": signal_info.device.uid,
                    "channel": channel,
                    "lo_frequency": lo_frequency,
                    "port_mode": port_mode.value if port_mode is not None else None,
                    "range": signal_range.value if signal_range is not None else None,
                    "range_unit": signal_range.unit
                    if signal_range is not None
                    else None,
                    "port_delay": port_delay,
                    "scheduler_port_delay": scheduler_port_delay,
                    "amplitude": self._experiment_dao.amplitude(signal_id),
                    "output_routers": [
                        router
                        for router in signal_info.output_routing
                        if router.to_channel == channel
                    ],
                }
                signal_is_modulated = signal_info.oscillator is not None

                if oscillator_is_hardware and signal_is_modulated:
                    output["modulation"] = True
                    if isinstance(oscillator_frequency, ParameterInfo):
                        oscillator_frequency = 0
                else:
                    output["modulation"] = False

                # default mixer calib
                if (
                    device_type == DeviceType.HDAWG
                ):  # for hdawgs, we add default values to the recipe
                    output["offset"] = 0.0
                    output["diagonal"] = 1.0
                    output["off_diagonal"] = 0.0
                else:  # others get no mixer calib values
                    output["offset"] = 0.0
                    output["diagonal"] = None
                    output["off_diagonal"] = None

                if signal_info.type == SignalInfoType.RF and voltage_offset is not None:
                    output["offset"] = voltage_offset

                if (
                    signal_info.type == SignalInfoType.IQ
                    and mixer_calibration is not None
                ):
                    if mixer_calibration.voltage_offsets is not None:
                        output["offset"] = mixer_calibration.voltage_offsets[
                            channel - base_channel
                        ]
                    if mixer_calibration.correction_matrix is not None:
                        output["diagonal"] = mixer_calibration.correction_matrix[
                            channel - base_channel
                        ][channel - base_channel]
                        output["off_diagonal"] = mixer_calibration.correction_matrix[
                            flipper[channel - base_channel]
                        ][channel - base_channel]

                device_type = DeviceType.from_device_info_type(
                    signal_info.device.device_type
                )
                if precompensation_is_nonzero(precompensation):
                    if not device_type.supports_precompensation:
                        raise RuntimeError(
                            f"Device {signal_info.device.uid} does not"
                            + " support precompensation"
                        )
                    warnings = verify_precompensation_parameters(
                        precompensation,
                        self._sampling_rate_tracker.sampling_rate_for_device(
                            signal_info.device.uid
                        ),
                        signal_id,
                    )
                    if warnings:
                        _logger.warning(warnings)
                    output["precompensation"] = precompensation
                if markers is not None:
                    if signal_info.type == SignalInfoType.IQ:
                        if device_type == DeviceType.HDAWG:
                            marker_key = channel % 2 + 1
                            if f"marker{marker_key}" in markers:
                                output["marker_mode"] = "MARKER"
                        elif device_type == DeviceType.SHFSG:
                            if "marker1" in markers:
                                output["marker_mode"] = "MARKER"
                            if "marker2" in markers:
                                raise RuntimeError("Only marker1 supported on SHFSG")
                if triggers is not None:
                    if signal_info.type == SignalInfoType.IQ:
                        if device_type == DeviceType.HDAWG:
                            trigger_bit = 2 ** (channel % 2)
                            if triggers & trigger_bit:
                                if (
                                    "marker_mode" in output
                                    and output["marker_mode"] == "MARKER"
                                ):
                                    raise RuntimeError(
                                        f"Trying to use marker and trigger on the same output channel {channel} with signal {signal_id} on device {signal_info.device.uid}"
                                    )
                                else:
                                    output["marker_mode"] = "TRIGGER"
                        elif device_type == DeviceType.SHFSG:
                            if triggers & 2:
                                raise RuntimeError("Only trigger 1 supported on SHFSG")
                            if triggers & 1:
                                if (
                                    "marker_mode" in output
                                    and output["marker_mode"] == "MARKER"
                                ):
                                    raise RuntimeError(
                                        f"Trying to use marker and trigger on the same SG output channel {channel} with signal {signal_id} on device {signal_info.device.uid}"
                                    )
                                else:
                                    output["marker_mode"] = "TRIGGER"

                output["oscillator_frequency"] = oscillator_frequency
                channel_key = (signal_info.device.uid, channel)
                # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
                all_channels[channel_key] = output
        retval = sorted(
            all_channels.values(),
            key=lambda output: output["device_id"] + str(output["channel"]),
        )
        return retval

    def calc_inputs(self, signal_delays: SignalDelays):
        all_channels = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            if signal_info.type != SignalInfoType.INTEGRATION:
                continue

            lo_frequency = self._experiment_dao.lo_frequency(signal_id)
            signal_range = self._experiment_dao.signal_range(signal_id)

            port_delay = self._experiment_dao.port_delay(signal_id)

            port_mode = self._experiment_dao.port_mode(signal_id)

            scheduler_port_delay: float = 0.0
            if signal_id in signal_delays:
                scheduler_port_delay += signal_delays[signal_id].on_device

            for channel in signal_info.channels:
                input = {
                    "device_id": signal_info.device.uid,
                    "channel": channel,
                    "lo_frequency": lo_frequency,
                    "range": signal_range.value if signal_range is not None else None,
                    "range_unit": signal_range.unit
                    if signal_range is not None
                    else None,
                    "port_delay": port_delay,
                    "scheduler_port_delay": scheduler_port_delay,
                    "port_mode": port_mode.value if port_mode is not None else None,
                }
                channel_key = (signal_info.device.uid, channel)
                # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
                all_channels[channel_key] = input
        retval = sorted(
            all_channels.values(),
            key=lambda input: input["device_id"] + str(input["channel"]),
        )
        return retval

    def calc_measurement_map(self, integration_times: IntegrationTimes):
        measurement_sections: List[str] = []

        for section_name in self._experiment_dao.sections():
            section_info = self._experiment_dao.section_info(section_name)
            if section_info.acquisition_type is not None:
                measurement_sections.append(section_name)

        section_measurement_infos = []

        for section_name in measurement_sections:
            section_signals = self._experiment_dao.section_signals_with_children(
                section_name
            )

            def empty_device():
                return {"signals": set(), "monitor": None}

            infos_by_device_awg = {}
            for signal in section_signals:
                signal_info_for_section = self._experiment_dao.signal_info(signal)
                device_type = DeviceType.from_device_info_type(
                    signal_info_for_section.device.device_type
                )
                awg_nr = Compiler.calc_awg_number(
                    signal_info_for_section.channels[0], device_type
                )

                if signal_info_for_section.type == SignalInfoType.INTEGRATION:
                    device_id = signal_info_for_section.device.uid
                    device_awg_key = (device_id, awg_nr)
                    if device_awg_key not in infos_by_device_awg:
                        infos_by_device_awg[device_awg_key] = {
                            "section_name": section_name,
                            "devices": {},
                        }
                    section_measurement_info = infos_by_device_awg[device_awg_key]

                    device = section_measurement_info["devices"].setdefault(
                        (device_id, awg_nr), empty_device()
                    )
                    device["signals"].add(signal)

                    _logger.debug(
                        "Added measurement device %s",
                        signal_info_for_section.device.uid,
                    )

            section_measurement_infos.extend(infos_by_device_awg.values())

        _logger.debug("Found section_measurement_infos  %s", section_measurement_infos)
        measurements = {}

        for info in section_measurement_infos:
            for device_awg_nr in info["devices"]:
                device_id, awg_nr = device_awg_nr
                if (device_id, awg_nr) in measurements:
                    _logger.debug(
                        "Expanding existing measurement record for device %s awg %d (when looking at section %s )",
                        device_id,
                        awg_nr,
                        info["section_name"],
                    )
                    measurement = measurements[(device_id, awg_nr)]
                else:
                    measurement = {"length": None}

                    integration_time_info = integration_times.section_info(
                        info["section_name"]
                    )
                    if integration_time_info is not None:
                        _logger.debug(
                            "Found integration_time_info %s", integration_time_info
                        )

                        signal_info_for_section_and_device_awg = next(
                            i
                            for i in integration_time_info.values()
                            if i.awg == awg_nr and i.device_id == device_id
                        )
                        measurement[
                            "length"
                        ] = signal_info_for_section_and_device_awg.length_in_samples
                    else:
                        del measurement["length"]
                    _logger.debug(
                        "Added measurement %s\n  for %s", measurement, device_awg_nr
                    )

                measurements[(device_id, awg_nr)] = measurement

        retval = {}
        for device_awg_key, v in measurements.items():
            device_id, awg_nr = device_awg_key
            if device_id not in retval:
                # make sure measurements are sorted by awg_nr
                retval[device_id] = SortedDict()
            retval[device_id][awg_nr] = v
            v["channel"] = awg_nr
        for k in list(retval.keys()):
            retval[k] = list(retval[k].values())

        return retval

    @singledispatchmethod
    def _add_io_to_recipe(self, output, recipe_generator: RecipeGenerator):
        ...

    @singledispatchmethod
    def _add_acquire_info_to_recipe(self, output, recipe_generator: RecipeGenerator):
        ...

    @_add_io_to_recipe.register
    def _(
        self,
        output: CombinedRealtimeCompilerOutputPrettyPrinter,
        recipe_generator: RecipeGenerator,
    ):
        ...

    @_add_io_to_recipe.register
    def _(
        self,
        combined_output: CombinedRealtimeCompilerOutputCode,
        recipe_generator: RecipeGenerator,
    ):
        for output in self.calc_outputs(combined_output.signal_delays):
            _logger.debug("Adding output %s", output)
            recipe_generator.add_output(
                output["device_id"],
                output["channel"],
                output["offset"],
                output["diagonal"],
                output["off_diagonal"],
                precompensation=output.get("precompensation"),
                modulation=output["modulation"],
                lo_frequency=output["lo_frequency"],
                port_mode=output["port_mode"],
                output_range=output["range"],
                output_range_unit=output["range_unit"],
                port_delay=output["port_delay"],
                scheduler_port_delay=output["scheduler_port_delay"],
                marker_mode=output.get("marker_mode"),
                amplitude=output["amplitude"],
                output_routers=output["output_routers"],
            )

        for input in self.calc_inputs(combined_output.signal_delays):
            _logger.debug("Adding input %s", input)
            recipe_generator.add_input(
                input["device_id"],
                input["channel"],
                lo_frequency=input["lo_frequency"],
                input_range=input["range"],
                input_range_unit=input["range_unit"],
                port_delay=input["port_delay"],
                scheduler_port_delay=input["scheduler_port_delay"],
                port_mode=input["port_mode"],
            )

    @_add_acquire_info_to_recipe.register
    def _(
        self,
        output: CombinedRealtimeCompilerOutputPrettyPrinter,
        recipe_generator: RecipeGenerator,
    ):
        ...

    @_add_acquire_info_to_recipe.register
    def _(
        self,
        output: CombinedRealtimeCompilerOutputCode,
        recipe_generator: RecipeGenerator,
    ):
        recipe_generator.add_integrator_allocations(
            self._integration_unit_allocation,
            self._experiment_dao,
            output.integration_weights,
        )

        recipe_generator.add_acquire_lengths(integration_times=output.integration_times)

        recipe_generator.add_measurements(
            self.calc_measurement_map(integration_times=output.integration_times)
        )

        recipe_generator.add_simultaneous_acquires(output.simultaneous_acquires)

    def _drive_recipe_generator(
        self,
        combined_outputs: CombinedRealtimeCompilerOutput,
        recipe_generator: RecipeGenerator,
    ):
        recipe_generator.add_oscillator_params(self._experiment_dao)

        for output in combined_outputs.combined_output.values():
            self._add_io_to_recipe(output, recipe_generator)
        for device in self._experiment_dao.device_infos():
            recipe_generator.validate_and_postprocess_ios(device)

        for device_id, awgs in self._awgs.items():
            for awg in awgs.values():
                signal_type = awg.signal_type
                if signal_type == AWGSignalType.DOUBLE:
                    signal_type = AWGSignalType.SINGLE
                elif signal_type == AWGSignalType.MULTI:
                    signal_type = AWGSignalType.IQ
                recipe_generator.add_awg(
                    device_id=device_id,
                    awg_number=awg.awg_number,
                    signal_type=signal_type.value,
                    feedback_register_config=combined_outputs.get_feedback_register_configurations(
                        awg.key
                    ),
                )

        for step in combined_outputs.realtime_steps:
            recipe_generator.add_realtime_step(step)

        for output in combined_outputs.combined_output.values():
            self._add_acquire_info_to_recipe(output, recipe_generator)

    def _generate_recipe(self):
        recipe_generator = RecipeGenerator()
        recipe_generator.from_experiment(
            self._experiment_dao, self._leader_properties, self._clock_settings
        )

        self._drive_recipe_generator(self._combined_compiler_output, recipe_generator)

        recipe_generator.add_total_execution_time(
            self._combined_compiler_output.total_execution_time
        )
        recipe_generator.add_max_step_execution_time(
            self._combined_compiler_output.max_execution_time_per_step
        )

        self._recipe = recipe_generator.recipe()
        _logger.debug("Recipe generation completed")

    def compiler_output(self) -> CompiledExperiment:
        return CompiledExperiment(
            experiment_dict=ExperimentDAO.dump(self._experiment_dao),
            scheduled_experiment=ScheduledExperiment(
                recipe=self._recipe,
                artifacts=self._combined_compiler_output.get_artifacts(),
                execution=self._execution,
                schedule=self._combined_compiler_output.schedule,
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

    @trace("compiler.run()")
    def run(self, data) -> CompiledExperiment:
        _logger.debug("ES Compiler run")

        self.use_experiment(data)
        self._analyze_setup()
        self._process_experiment()

        self._generate_recipe()

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
