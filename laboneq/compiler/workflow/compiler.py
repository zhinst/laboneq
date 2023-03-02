# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sortedcollections import SortedDict

from laboneq._observability.tracing import trace
from laboneq.compiler.code_generator import CodeGenerator
from laboneq.compiler.common import compiler_settings
from laboneq.compiler.common.awg_info import AWGInfo
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.common.trigger_mode import TriggerMode
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
from laboneq.compiler.experiment_access.section_graph import SectionGraph
from laboneq.compiler.new_scheduler.scheduler import Scheduler as NewScheduler
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.compiler.workflow.precompensation_helpers import (
    compute_precompensation_delays_on_grid,
    compute_precompensations_and_delays,
    precompensation_is_nonzero,
    verify_precompensation_parameters,
)
from laboneq.compiler.workflow.recipe_generator import RecipeGenerator
from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.mixer_type import MixerType

_logger = logging.getLogger(__name__)


@dataclass
class LeaderProperties:
    global_leader: str = None
    is_desktop_setup: bool = False
    internal_followers: List[str] = field(default_factory=list)


_AWGMapping = Dict[str, Dict[int, AWGInfo]]


class Compiler:
    Scheduler = Scheduler

    def __init__(self, settings: Optional[Dict] = None):
        self._osc_numbering = None
        self._section_grids = {}
        self._experiment_dao: ExperimentDAO = None
        self._settings = compiler_settings.from_dict(settings)
        self._sampling_rate_tracker: SamplingRateTracker = None
        self.Scheduler = Scheduler
        if self._settings.USE_EXPERIMENTAL_SCHEDULER:
            self.Scheduler = NewScheduler
        self._scheduler: Compiler.Scheduler = None

        self._leader_properties = LeaderProperties()
        self._clock_settings = {}
        self._integration_unit_allocation = None
        self._section_graph_object = None
        self._awgs: _AWGMapping = {}
        self._precompensations: Dict[
            str, Dict[str, Union[Dict[str, Any], float]]
        ] = None

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

    def print_section_graph(self):
        self._section_graph_object.log_graph()

    def use_experiment(self, experiment):
        if "experiment" in experiment and "setup" in experiment:
            _logger.debug("Processing DSLv3 setup and experiment")
            self._experiment_dao = ExperimentDAO(
                None, experiment["setup"], experiment["experiment"]
            )
        else:
            self._experiment_dao = ExperimentDAO(experiment)

    def _analyze_setup(self):
        def get_first_instr_of(device_infos, type):
            return next((instr for instr in device_infos if instr.device_type == type))

        device_infos = self._experiment_dao.device_infos()
        device_type_list = [i.device_type for i in device_infos]
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
        used_devices = set(info.device_type for info in signal_infos)
        used_device_serials = set(info.device_serial for info in signal_infos)
        if (
            "hdawg" in used_devices
            and "uhfqa" in used_devices
            and bool(shf_types.intersection(used_devices))
        ):
            raise RuntimeError(
                "Setups with signals on each of HDAWG, UHFQA and SHF type "
                + "instruments are not supported"
            )

        self._leader_properties.is_desktop_setup = not has_pqsc and (
            used_devices == {"hdawg"}
            or used_devices == {"shfsg"}
            or used_devices == {"shfqa"}
            or used_devices == {"shfqa", "shfsg"}
            and len(used_device_serials) == 1  # SHFQC
            or used_devices == {"hdawg", "uhfqa"}
            or (used_devices == {"uhfqa"} and has_hdawg)  # No signal on leader
        )
        if (
            not has_pqsc
            and not self._leader_properties.is_desktop_setup
            and used_devices != {"uhfqa"}
            and bool(used_devices)  # Allow empty experiment (used in tests)
        ):
            raise RuntimeError(
                f"Unsupported device combination {used_devices} for small setup"
            )

        leader = self._experiment_dao.global_leader_device()
        if self._leader_properties.is_desktop_setup:
            if leader is None:
                if has_hdawg:
                    leader = get_first_instr_of(device_infos, "hdawg").id
                elif has_shfqa:
                    leader = get_first_instr_of(device_infos, "shfqa").id
                    if has_shfsg:  # SHFQC
                        self._leader_properties.internal_followers = [
                            get_first_instr_of(device_infos, "shfsg").id
                        ]
                elif has_shfsg:
                    leader = get_first_instr_of(device_infos, "shfsg").id

            _logger.debug("Using desktop setup configuration with leader %s", leader)

            if has_hdawg or has_shfsg and not has_shfqa:
                has_signal_on_awg_0_of_leader = False
                for signal_id in self._experiment_dao.signals():
                    signal_info = self._experiment_dao.signal_info(signal_id)
                    if signal_info.device_id == leader and (
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
                        device_id, channels, "out", signal_id, signal_type, False
                    )
                    _logger.debug(
                        "No pulses played on channels 1 or 2 of %s, adding dummy signal %s to ensure triggering of the setup",
                        leader,
                        signal_id,
                    )

            has_qa = type_counter["shfqa"] > 0 or type_counter["uhfqa"] > 0
            is_hdawg_solo = type_counter["hdawg"] == 1 and not has_shf and not has_qa
            if is_hdawg_solo:
                first_hdawg = get_first_instr_of(device_infos, "hdawg")
                if first_hdawg.reference_clock_source is None:
                    self._clock_settings[first_hdawg.id] = "internal"
            else:
                if not has_hdawg and has_shfsg:  # SHFSG or SHFQC solo
                    first_shfsg = get_first_instr_of(device_infos, "shfsg")
                    if first_shfsg.reference_clock_source is None:
                        self._clock_settings[first_shfsg.id] = "internal"
                if not has_hdawg and has_shfqa:  # SHFQA or SHFQC solo
                    first_shfqa = get_first_instr_of(device_infos, "shfqa")
                    if first_shfqa.reference_clock_source is None:
                        self._clock_settings[first_shfqa.id] = "internal"

        self._clock_settings["use_2GHz_for_HDAWG"] = has_shf
        self._leader_properties.global_leader = leader

    def _process_experiment(self, experiment):
        self.use_experiment(experiment)
        self._analyze_setup()
        self._calc_osc_numbering()
        self._calc_awgs()
        _logger.debug("Processing Sections:::::::")
        self._section_graph_object = SectionGraph.from_dao(self._experiment_dao)
        self.print_section_graph()
        self._sampling_rate_tracker = SamplingRateTracker(
            self._experiment_dao, self._clock_settings
        )

        self._scheduler = self.Scheduler(
            self._experiment_dao,
            self._section_graph_object,
            self._sampling_rate_tracker,
            self._clock_settings,
            self._settings,
        )
        self._scheduler.run()

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

    @trace("compiler.generate-code()")
    def _generate_code(self):
        self._calc_awgs()
        self._calc_shfqa_generator_allocation()
        self._precompensations = compute_precompensations_and_delays(
            self._experiment_dao
        )
        compute_precompensation_delays_on_grid(
            self._precompensations,
            self._experiment_dao,
            self._clock_settings["use_2GHz_for_HDAWG"],
        )

        code_generator = CodeGenerator(self._settings)
        self._code_generator = code_generator

        for signal_id in self._experiment_dao.signals():

            signal_info = self._experiment_dao.signal_info(signal_id)
            delay_signal = signal_info.delay_signal

            device_type = DeviceType(signal_info.device_type)
            device_id = signal_info.device_id

            sampling_rate = self._sampling_rate_tracker.sampling_rate_for_device(
                device_id
            )
            start_delay = self.get_lead_delay(
                device_type,
                self._leader_properties.is_desktop_setup,
                self._clock_settings["use_2GHz_for_HDAWG"],
            )
            start_delay += self._precompensations[signal_id]["computed_delay_signal"]

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

            signal_type = signal_info.signal_type

            _logger.debug(
                "Adding signal %s with signal type %s", signal_id, signal_type
            )

            oscillator_frequency = None

            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            if (
                oscillator_info is not None
                and not oscillator_info.hardware
                and signal_info.modulation
            ):
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
                oscillator_info.id
                if oscillator_info is not None and oscillator_info.hardware
                else None
            )

            mixer_type = MixerType.IQ
            if (
                device_type == DeviceType.UHFQA
                and oscillator_info
                and oscillator_info.hardware
            ):
                mixer_type = MixerType.UHFQA_ENVELOPE
            elif signal_type in ("single",):
                mixer_type = None

            signal_obj = SignalObj(
                id=signal_id,
                sampling_rate=sampling_rate,
                start_delay=start_delay,
                delay_signal=delay_signal,
                signal_type=signal_type,
                device_id=device_id,
                awg=awg,
                device_type=device_type,
                oscillator_frequency=oscillator_frequency,
                channels=channels,
                mixer_type=mixer_type,
                hw_oscillator=hw_oscillator,
            )
            code_generator.add_signal(signal_obj)

        _logger.debug("Preparing events for code generator")
        events = self._scheduler.event_timing(expand_loops=False)
        code_generator.gen_acquire_map(events, self._section_graph_object)
        integration_times, signal_delays = code_generator.gen_seq_c(
            events,
            {k: self._experiment_dao.pulse(k) for k in self._experiment_dao.pulses()},
        )
        self._command_table_match_offsets = code_generator.command_table_match_offsets()
        self._feedback_connections = code_generator.feedback_connections()
        code_generator.gen_waves()

        _logger.debug("Code generation completed")
        return integration_times, signal_delays

    def _calc_osc_numbering(self):
        self._osc_numbering = {}

        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            device_type = DeviceType(signal_info.device_type)

            if signal_info.signal_type == "integration":
                continue

            hw_osc_names = set()
            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            if oscillator_info is not None and oscillator_info.hardware:
                hw_osc_names.add(oscillator_info.id)

            base_channel = min(signal_info.channels)
            min_osc_number = base_channel * 2
            count = 0
            for osc_name in hw_osc_names:
                if device_type == DeviceType.SHFQA:
                    self._osc_numbering[osc_name] = min(signal_info.channels)
                else:
                    self._osc_numbering[osc_name] = min_osc_number + count
                    count += 1

    def _calc_integration_unit_allocation(self):
        self._integration_unit_allocation = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            _logger.debug("_integration_unit_allocation considering %s", signal_info)
            if signal_info.signal_type == "integration":
                _logger.debug(
                    "_integration_unit_allocation: found integration signal %s",
                    signal_info,
                )
                device_id = signal_info.device_id
                device_type = DeviceType(signal_info.device_type)
                awg_nr = Compiler.calc_awg_number(signal_info.channels[0], device_type)

                num_acquire_signals = len(
                    list(
                        filter(
                            lambda x: x["device_id"] == device_id
                            and x["awg_nr"] == awg_nr,
                            self._integration_unit_allocation.values(),
                        )
                    )
                )

                integrators_per_signal = (
                    device_type.num_integration_units_per_acquire_signal
                    if self._experiment_dao.acquisition_type
                    in [
                        AcquisitionType.RAW,
                        AcquisitionType.SPECTROSCOPY,
                        AcquisitionType.INTEGRATION,
                    ]
                    else 1
                )

                self._integration_unit_allocation[signal_id] = {
                    "device_id": device_id,
                    "awg_nr": awg_nr,
                    "channels": [
                        integrators_per_signal * num_acquire_signals + i
                        for i in range(integrators_per_signal)
                    ],
                }

    def _calc_shfqa_generator_allocation(self):
        self._shfqa_generator_allocation = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            device_type = DeviceType(signal_info.device_type)

            if signal_info.signal_type == "iq" and device_type == DeviceType.SHFQA:
                _logger.debug(
                    "_shfqa_generator_allocation: found SHFQA iq signal %s", signal_info
                )
                device_id = signal_info.device_id
                awg_nr = Compiler.calc_awg_number(signal_info.channels[0], device_type)
                num_generator_signals = len(
                    list(
                        filter(
                            lambda x: x["device_id"] == device_id
                            and x["awg_nr"] == awg_nr,
                            self._shfqa_generator_allocation.values(),
                        )
                    )
                )

                self._shfqa_generator_allocation[signal_id] = {
                    "device_id": device_id,
                    "awg_nr": awg_nr,
                    "channels": [num_generator_signals],
                }

    def osc_number(self, osc_name):
        if self._osc_numbering is None:
            raise Exception("Oscillator numbers not yet calculated")
        return self._osc_numbering[osc_name]

    @staticmethod
    def calc_awg_number(channel, device_type: DeviceType):
        if device_type == DeviceType.UHFQA:
            return 0
        return int(math.floor(channel / device_type.channels_per_awg))

    def _calc_awgs(self):
        awgs: _AWGMapping = {}
        signals_by_channel_and_awg: Dict[
            Tuple[str, int, int], Dict[str, Union[Set, AWGInfo]]
        ] = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            device_id = signal_info.device_id
            device_type = DeviceType(signal_info.device_type)
            for channel in sorted(signal_info.channels):
                awg_number = Compiler.calc_awg_number(channel, device_type)
                device_awgs = awgs.setdefault(device_id, SortedDict())
                awg = device_awgs.get(awg_number)
                if awg is None:
                    signal_type = signal_info.signal_type
                    # Treat "integration" signal type same as "iq" at AWG level
                    if signal_type == "integration":
                        signal_type = "iq"
                    awg = AWGInfo(
                        device_id=device_id,
                        signal_type=AWGSignalType(signal_type),
                        awg_number=awg_number,
                        seqc="seq_" + device_id + "_" + str(awg_number) + ".seqc",
                        device_type=device_type,
                        sampling_rate=None,
                    )
                    device_awgs[awg_number] = awg

                awg.signal_channels.append((signal_id, channel))

                if signal_info.signal_type == "iq":
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
                    self._experiment_dao.signal_info(signal_id).delay_signal or 0.0
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

        self._awgs = awgs

    def get_awg(self, signal_id) -> AWGInfo:
        awg_number = None
        signal_info = self._experiment_dao.signal_info(signal_id)

        device_id = signal_info.device_id
        device_type = DeviceType(signal_info.device_type)
        awg_number = Compiler.calc_awg_number(signal_info.channels[0], device_type)
        if signal_info.signal_type == "integration" and device_type != DeviceType.SHFQA:
            awg_number = 0
        return self._awgs[device_id][awg_number]

    def calc_outputs(self, signal_delays):
        all_channels = {}

        flipper = [1, 0]

        marker_signals = self._experiment_dao.marker_signals()
        trigger_signals = self._experiment_dao.trigger_signals()

        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            if signal_info.signal_type == "integration":
                continue
            oscillator_frequency = None
            oscillator_number = None

            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            oscillator_is_hardware = (
                oscillator_info is not None and oscillator_info.hardware
            )
            if oscillator_is_hardware:
                osc_name = oscillator_info.id
                oscillator_frequency = oscillator_info.frequency
                oscillator_number = self.osc_number(osc_name)

            voltage_offset = self._experiment_dao.voltage_offset(signal_id)
            mixer_calibration = self._experiment_dao.mixer_calibration(signal_id)
            lo_frequency = self._experiment_dao.lo_frequency(signal_id)
            port_mode = self._experiment_dao.port_mode(signal_id)
            signal_range, signal_range_unit = self._experiment_dao.signal_range(
                signal_id
            )
            device_type = DeviceType(signal_info.device_type)
            port_delay = self._experiment_dao.port_delay(signal_id)
            if signal_id in signal_delays:
                port_delay = (port_delay or 0) + signal_delays[signal_id]["on_device"]
                if abs(port_delay) < 1e-12:
                    port_delay = None
            precompensation = self._precompensations[signal_id]
            pc_port_delay = precompensation["computed_port_delay"]
            if pc_port_delay:
                port_delay = (port_delay or 0) + pc_port_delay

            base_channel = min(signal_info.channels)

            markers = marker_signals.get(signal_id, None)

            triggers = trigger_signals.get(signal_id)

            for channel in signal_info.channels:
                output = {
                    "device_id": signal_info.device_id,
                    "channel": channel,
                    "lo_frequency": lo_frequency,
                    "port_mode": port_mode,
                    "range": signal_range,
                    "range_unit": signal_range_unit,
                    "port_delay": port_delay,
                }
                signal_is_modulated = signal_info.modulation
                output_modulation_logic = {
                    (True, True): True,
                    (False, False): False,
                    (True, False): False,
                    (False, True): False,
                }
                if output_modulation_logic[
                    (oscillator_is_hardware, signal_is_modulated)
                ]:
                    output["modulation"] = True
                    if oscillator_frequency is None:
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

                if signal_info.signal_type == "single" and voltage_offset is not None:
                    output["offset"] = voltage_offset

                if signal_info.signal_type == "iq" and mixer_calibration is not None:
                    if mixer_calibration["voltage_offsets"] is not None:
                        output["offset"] = mixer_calibration["voltage_offsets"][
                            channel - base_channel
                        ]
                    if mixer_calibration["correction_matrix"] is not None:
                        output["diagonal"] = mixer_calibration["correction_matrix"][
                            channel - base_channel
                        ][channel - base_channel]
                        output["off_diagonal"] = mixer_calibration["correction_matrix"][
                            flipper[channel - base_channel]
                        ][channel - base_channel]

                if precompensation_is_nonzero(precompensation):
                    device_type = DeviceType(signal_info.device_type)
                    if not device_type.supports_precompensation:
                        raise RuntimeError(
                            f"Device {signal_info.device_id} does not"
                            + " support precompensation"
                        )
                    warnings = verify_precompensation_parameters(
                        precompensation,
                        self._sampling_rate_tracker.sampling_rate_for_device(
                            signal_info.device_id
                        ),
                        signal_id,
                    )
                    if warnings:
                        _logger.warn(warnings)
                    output["precompensation"] = precompensation
                if markers is not None:
                    if signal_info.signal_type == "iq":
                        marker_key = channel % 2 + 1
                        if f"marker{marker_key}" in markers:
                            output["marker_mode"] = "MARKER"

                if triggers is not None:
                    if signal_info.signal_type == "iq":
                        trigger_bit = 2 ** (channel % 2)
                        if triggers & trigger_bit:
                            if (
                                "marker_mode" in output
                                and output["marker_mode"] == "MARKER"
                            ):
                                raise RuntimeError(
                                    f"Trying to use marker and trigger on the same output channel {channel} with signal {signal_id}"
                                )
                            else:
                                output["marker_mode"] = "TRIGGER"

                output["oscillator_frequency"] = oscillator_frequency
                output["oscillator"] = oscillator_number
                channel_key = (signal_info.device_id, channel)
                # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
                all_channels[channel_key] = output
        retval = sorted(
            all_channels.values(),
            key=lambda output: output["device_id"] + str(output["channel"]),
        )
        return retval

    def calc_inputs(self):
        all_channels = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            if signal_info.signal_type != "integration":
                continue

            lo_frequency = self._experiment_dao.lo_frequency(signal_id)
            signal_range, signal_range_unit = self._experiment_dao.signal_range(
                signal_id
            )
            port_delay = self._experiment_dao.port_delay(signal_id)
            for channel in signal_info.channels:
                input = {
                    "device_id": signal_info.device_id,
                    "channel": channel,
                    "lo_frequency": lo_frequency,
                    "range": signal_range,
                    "range_unit": signal_range_unit,
                    "port_delay": port_delay,
                }
                channel_key = (signal_info.device_id, channel)
                # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
                all_channels[channel_key] = input
        retval = sorted(
            all_channels.values(),
            key=lambda input: input["device_id"] + str(input["channel"]),
        )
        return retval

    def calc_measurement_map(self, integration_times):
        measurement_sections = []
        for (
            graph_section_name
        ) in self._section_graph_object.topologically_sorted_sections():
            graph_section_info = self._section_graph_object.section_info(
                graph_section_name
            )
            section_name = graph_section_info.section_display_name
            section_info = self._experiment_dao.section_info(section_name)
            if section_info.acquisition_types is not None:
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
                device_type = DeviceType(signal_info_for_section.device_type)
                awg_nr = Compiler.calc_awg_number(
                    signal_info_for_section.channels[0], device_type
                )

                if signal_info_for_section.signal_type == "integration":
                    device_id = signal_info_for_section.device_id
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
                        signal_info_for_section.device_id,
                    )

            section_measurement_infos.extend(infos_by_device_awg.values())

        _logger.debug("Found section_measurement_infos  %s", section_measurement_infos)
        measurements = {}

        for info in section_measurement_infos:
            section_name = info["section_name"]

            for device_awg_nr, v in info["devices"].items():

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
                    measurement = {"length": None, "delay": 0}

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
                        measurement[
                            "delay"
                        ] = signal_info_for_section_and_device_awg.delay_in_samples
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

    def _generate_recipe(self, integration_times, signal_delays):
        recipe_generator = RecipeGenerator()
        recipe_generator.from_experiment(
            self._experiment_dao, self._leader_properties, self._clock_settings
        )

        for output in self.calc_outputs(signal_delays):
            _logger.debug("Adding output %s", output)
            recipe_generator.add_output(
                output["device_id"],
                output["channel"],
                output["offset"],
                output["diagonal"],
                output["off_diagonal"],
                precompensation=output.get("precompensation"),
                modulation=output["modulation"],
                oscillator=output["oscillator"],
                oscillator_frequency=output["oscillator_frequency"],
                lo_frequency=output["lo_frequency"],
                port_mode=output["port_mode"],
                output_range=output["range"],
                output_range_unit=output["range_unit"],
                port_delay=output["port_delay"],
                marker_mode=output.get("marker_mode"),
            )

        for input in self.calc_inputs():
            _logger.debug("Adding input %s", input)
            recipe_generator.add_input(
                input["device_id"],
                input["channel"],
                lo_frequency=input["lo_frequency"],
                input_range=input["range"],
                input_range_unit=input["range_unit"],
                port_delay=input["port_delay"],
            )

        for device_id, awgs in self._awgs.items():
            for awg in awgs.values():
                signal_type = awg.signal_type
                if signal_type == AWGSignalType.DOUBLE:
                    signal_type = AWGSignalType.SINGLE
                if signal_type == AWGSignalType.MULTI:
                    signal_type = AWGSignalType.IQ
                # Find the acquire signal from which we read the feedback from and which
                # is used via a match/state construct for the drive signals of this awg
                awg_signals = {c for c, _ in awg.signal_channels}
                qa_signal_ids = {
                    h.acquire
                    for h in self._feedback_connections.values()
                    if h.drive.intersection(awg_signals)
                }
                if len(qa_signal_ids) > 1:
                    raise Exception(
                        f"The drive signal(s) ({set(awg_signals)}) can only react to "
                        f"one acquire signal for feedback, got {qa_signal_ids}."
                    )
                recipe_generator.add_awg(
                    device_id,
                    awg.awg_number,
                    signal_type.value,
                    awg.seqc,
                    next(iter(qa_signal_ids), None),
                    self._command_table_match_offsets.get((device_id, awg.awg_number)),
                )

        if self._code_generator is None:
            raise Exception("Code generator not initialized")
        recipe_generator.add_oscillator_params(self._experiment_dao)
        recipe_generator.add_integrator_allocations(
            self._integration_unit_allocation,
            self._experiment_dao,
            self._code_generator.integration_weights(),
        )

        recipe_generator.add_acquire_lengths(integration_times)

        recipe_generator.add_measurements(
            self.calc_measurement_map(integration_times=integration_times)
        )

        recipe_generator.add_simultaneous_acquires(
            self._code_generator.simultaneous_acquires()
        )

        recipe_generator.add_total_execution_time(
            self._code_generator.total_execution_time()
        )

        self._recipe = recipe_generator.recipe()
        _logger.debug("Recipe generation completed")

    def compiler_output(self) -> CompiledExperiment:
        return CompiledExperiment(
            recipe=self._recipe,
            src=self._code_generator.src(),
            waves=self._code_generator.waves(),
            wave_indices=self._code_generator.wave_indices(),
            command_tables=self._code_generator.command_tables(),
            schedule=self._prepare_schedule(),
            experiment_dict=ExperimentDAO.dump(self._experiment_dao),
            pulse_map=self._code_generator.pulse_map(),
        )

    def _prepare_schedule(self):
        event_list = self._scheduler.event_timing(
            expand_loops=self._settings.EXPAND_LOOPS_FOR_SCHEDULE,
            max_events=self._settings.MAX_EVENTS_TO_PUBLISH,
        )

        event_list = [
            {k: v for k, v in event.items() if v is not None} for event in event_list
        ]

        section_graph = []
        section_info_out = {}
        subsection_map = {}

        try:
            root_section = self._section_graph_object.root_sections()[0]
        except IndexError:
            return {
                "event_list": [],
                "section_graph": {},
                "section_info": {},
                "subsection_map": {},
                "section_signals_with_children": {},
                "sampling_rates": [],
            }

        section_info_out = {
            k: {"depth": v} for k, v in self._section_graph_object.depth_map().items()
        }

        if len(list(section_info_out.keys())) == 0:
            section_info_out = {root_section: {"depth": 0}}

        preorder_map = self._section_graph_object.preorder_map()

        section_info_out = {
            k: {**v, **{"preorder": preorder_map[k]}}
            for k, v in section_info_out.items()
        }

        _logger.debug("Section_info: %s", section_info_out)
        subsection_map = self._section_graph_object.subsection_map()
        _logger.debug("subsection_map=%s", subsection_map)
        for k, v in section_info_out.items():
            if len(subsection_map[k]) == 0:
                v["is_leaf"] = True
            else:
                v["is_leaf"] = False
        section_graph = self._section_graph_object.as_primitive()
        section_signals_with_children = {}

        for section in self._section_graph_object.sections():
            section_info = self._section_graph_object.section_info(section)
            section_display_name = section_info.section_display_name
            section_signals_with_children[section] = list(
                self._experiment_dao.section_signals_with_children(section_display_name)
            )
            section_info_out[section]["section_display_name"] = section_display_name

        sampling_rate_tuples = []
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            device_id = signal_info.device_id
            device_type = signal_info.device_type
            sampling_rate_tuples.append(
                (
                    device_type,
                    int(
                        self._sampling_rate_tracker.sampling_rate_for_device(device_id)
                    ),
                )
            )

        sampling_rates = [
            [list(set([d[0] for d in sampling_rate_tuples if d[1] == r])), r]
            for r in set([t[1] for t in sampling_rate_tuples])
        ]

        _logger.debug("Pulse sheet generation completed")

        return {
            "event_list": event_list,
            "section_graph": section_graph,
            "section_info": section_info_out,
            "subsection_map": subsection_map,
            "section_signals_with_children": section_signals_with_children,
            "sampling_rates": sampling_rates,
        }

    def dump_src(self, info=False):
        for src in self.compiler_output().src:
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

        self._process_experiment(data)
        self._calc_integration_unit_allocation()

        integration_times, signal_delays = self._generate_code()
        self._generate_recipe(integration_times, signal_delays)

        retval = self.compiler_output()

        total_seqc_lines = 0
        for f in retval.src:
            total_seqc_lines += f["text"].count("\n")
        _logger.info("Total seqC lines generated: %d", total_seqc_lines)

        total_samples = 0
        for f in retval.waves:
            try:
                total_samples += len(f["samples"])
            except KeyError:
                pass
        _logger.info("Total sample points generated: %d", total_samples)

        _logger.info("Finished LabOne Q Compiler run.")
        return retval

    def get_lead_delay(self, device_type, desktop_setup, hdawg_uses_2GHz):
        if not isinstance(device_type, DeviceType):
            raise Exception(f"Device type {device_type} is not of type DeviceType")
        if device_type == DeviceType.HDAWG:
            if not desktop_setup:
                if hdawg_uses_2GHz:
                    return self._settings.HDAWG_LEAD_PQSC_2GHz
                else:
                    return self._settings.HDAWG_LEAD_PQSC
            else:
                if hdawg_uses_2GHz:
                    return self._settings.HDAWG_LEAD_DESKTOP_SETUP_2GHz
                else:
                    return self._settings.HDAWG_LEAD_DESKTOP_SETUP
        elif device_type == DeviceType.UHFQA:
            return self._settings.UHFQA_LEAD_PQSC
        elif device_type == DeviceType.SHFQA:
            return self._settings.SHFQA_LEAD_PQSC
        elif device_type == DeviceType.SHFSG:
            return self._settings.SHFSG_LEAD_PQSC
        else:
            raise Exception(f"Unsupported device type {device_type}")


def find_obj_by_id(object_list, id):
    for i in object_list:
        if i["id"] == id:
            return i

    return None
