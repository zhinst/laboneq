# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, TypeVar

import numpy as np

from laboneq._utils import UIDReference
from laboneq.compiler import DeviceType
from laboneq.core.exceptions import LabOneQException
from laboneq.core.path import LogicalSignalGroups_Path, insert_logical_signal_prefix
from laboneq.core.types.enums import acquisition_type as acq_type
from laboneq.core.types.units import Quantity
from laboneq.data.calibration import (
    AmplifierPump,
    MixerCalibration,
    ModulationType,
    Oscillator,
    PortMode,
    Precompensation,
    SignalCalibration,
)
from laboneq.data.compilation_job import (
    AmplifierPumpInfo,
    ChunkingInfo,
    DeviceInfo,
    DeviceInfoType,
    ExperimentInfo,
    MixerCalibrationInfo,
    OscillatorInfo,
    OutputRoute,
    ParameterInfo,
    PrecompensationInfo,
    SignalInfo,
    SignalInfoType,
    SignalRange,
)
from laboneq.data.experiment_description import (
    Acquire,
    AcquireLoopRt,
    Delay,
    ExecutionType,
    Experiment,
    ExperimentSignal,
    PlayPulse,
    PrngLoop,
    Section,
    SignalOperation,
    Sweep,
)
from laboneq.data.parameter import LinearSweepParameter, Parameter, SweepParameter
from laboneq.data.setup_description import (
    IODirection,
    LogicalSignal,
    PhysicalChannelType,
    Setup,
)
from laboneq.data.setup_description.setup_helper import SetupHelper
from laboneq.implementation.payload_builder.experiment_info_builder.device_info_builder import (
    DeviceInfoBuilder,
)
from laboneq.implementation.utils.devices import device_setup_fingerprint

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class ExperimentInfoBuilder:
    def __init__(
        self,
        experiment: Experiment,
        device_setup: Setup,
        signal_mappings: Dict[str, str],
    ):
        self._experiment = experiment
        self._device_setup = device_setup
        self._signal_mappings = signal_mappings
        self._ls_to_exp_sig_mapping = {
            ls: exp for exp, ls in self._signal_mappings.items()
        }
        self._params: dict[str, ParameterInfo] = {}
        self._nt_only_params: set[str] = set()
        self._chunking_info: ChunkingInfo | None = None
        self._oscillators: dict[str, OscillatorInfo] = {}
        self._signal_infos: dict[str, SignalInfo] = {}

        self._device_info = DeviceInfoBuilder(self._device_setup)
        self._setup_helper = SetupHelper(self._device_setup)

        self._ppc_connections = self._setup_helper.ppc_connections()

        self._sweep_params_min_maxes: dict[str, tuple[float, float]] = {}
        self._acquisition_type: acq_type.AcquisitionType | None = None
        # Rust migration helper fields:
        # Parameters defined in the experiment.
        self._dsl_parameters: dict[str, Parameter] = {}

    def load_experiment(self) -> ExperimentInfo:
        self._check_physical_channel_calibration_conflict()
        for signal in self._experiment.signals:
            self._load_signal(signal)

        section_uid_map: dict[str, Section] = {}
        for section in self._experiment.sections:
            self._walk_sections(section, section_uid_map)
        self._validate_realtime(self._experiment.sections)

        experiment_info = ExperimentInfo(
            uid=self._experiment.uid,
            device_setup_fingerprint=device_setup_fingerprint(self._device_setup),
            devices=list(self._device_info.device_mapping.values()),
            signals=sorted(self._signal_infos.values(), key=lambda s: s.uid),
            acquisition_type=self._acquisition_type,
            global_leader_device=self._device_info.global_leader,
            chunking=self._chunking_info,
            src=self._experiment,
            dsl_parameters=list(self._dsl_parameters.values()),
        )
        self._resolve_oscillator_modulation_type(experiment_info)
        return experiment_info

    def _check_physical_channel_calibration_conflict(self):
        PHYSICAL_CHANNEL_CALIBRATION_FIELDS = (
            "local_oscillator_frequency",
            "port_delay",
            "port_mode",
            "range",
            "voltage_offset",
            "amplitude",
            # # skip validation of these structured fields
            "mixer_calibration",
            "precompensation",
            "amplifier_pump",
        )

        exp_signals_by_pc = {}
        for signal in self._experiment.signals:
            try:
                mapped_ls_path: str = self._signal_mappings[signal.uid]
            except KeyError as e:
                raise LabOneQException(
                    f"Experiment signal '{signal.uid}' has no mapping to a logical signal."
                ) from e
            pc = self._setup_helper.instruments.physical_channel_by_logical_signal(
                mapped_ls_path
            )
            exp_signals_by_pc.setdefault((pc.group, pc.name), []).append(signal)

        # Merge the calibration of those ExperimentSignals that touch the same
        # PhysicalChannel.
        for (pc_group, pc_name), exp_signals in exp_signals_by_pc.items():
            for field_ in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
                unique_value = None
                conflicting = False
                for exp_signal in exp_signals:
                    exp_cal = self._experiment.calibration.items.get(exp_signal.uid)
                    if exp_cal is None:
                        continue
                    value = getattr(exp_cal, field_)
                    if value is not None:
                        if unique_value is None:
                            unique_value = value
                        elif unique_value != value:
                            conflicting = True
                            break
                if conflicting:
                    conflicting_signals = [
                        exp_signal.uid
                        for exp_signal in exp_signals
                        if (
                            other_signal_cal := self._experiment.calibration.items.get(
                                exp_signal.uid
                            )
                        )
                        is not None
                        and getattr(other_signal_cal, field_) is not None
                    ]
                    pc_uid = f"{pc_group}/{pc_name}"
                    raise LabOneQException(
                        f"The experiment signals {', '.join(conflicting_signals)} all "
                        f"touch physical channel '{pc_uid}', but provide conflicting "
                        f"settings for calibration field '{field_}'."
                    )
                if unique_value is not None:
                    # Make sure all the experiment signals agree.
                    for exp_signal in exp_signals:
                        exp_cal = self._experiment.calibration.items.setdefault(
                            exp_signal.uid, SignalCalibration()
                        )
                        setattr(exp_cal, field_, unique_value)

    def _get_signal_calibration(
        self, exp_signal: ExperimentSignal, logical_signal: LogicalSignal
    ) -> SignalCalibration:
        baseline_calib = self._setup_helper.calibration.by_logical_signal(
            logical_signal
        )

        exp_calib = self._experiment.calibration.items.get(exp_signal.uid)

        if baseline_calib is None:
            calibration = exp_calib
        elif exp_calib is None:
            calibration = baseline_calib
        else:
            _logger.debug(
                "Found overriding signal calibration for %s/%s %s",
                logical_signal.group,
                logical_signal.name,
                exp_calib,
            )
            calibration = AttributeOverrider(baseline_calib, exp_calib)
        return calibration

    def _load_oscillator(self, oscillator: Oscillator) -> OscillatorInfo:
        if oscillator.modulation_type == ModulationType.HARDWARE:
            is_hw = True
        elif oscillator.modulation_type == ModulationType.SOFTWARE:
            is_hw = False
        else:
            if oscillator.modulation_type not in (None, ModulationType.AUTO):
                raise LabOneQException(
                    f"Invalid modulation type '{oscillator.modulation_type}' for"
                    f" oscillator '{oscillator.uid}'"
                )
            is_hw = None  # Unspecified for now, will resolve later

        frequency = self.opt_param(oscillator.frequency, nt_only=False)
        oscillator_info = OscillatorInfo(oscillator.uid, frequency, is_hw)
        if oscillator.uid in self._oscillators:
            if self._oscillators[oscillator.uid] != oscillator_info:
                raise LabOneQException(
                    f"Found multiple, inconsistent oscillators with same UID '{oscillator.uid}'."
                )
            oscillator_info = self._oscillators[oscillator.uid]
        else:
            self._oscillators[oscillator.uid] = oscillator_info

        return oscillator_info

    def _load_mixer_cal(self, mixer_cal: MixerCalibration) -> MixerCalibrationInfo:
        return MixerCalibrationInfo(
            voltage_offsets=tuple(
                self.opt_param(offset) for offset in mixer_cal.voltage_offsets
            ),
            correction_matrix=tuple(
                [
                    [self.opt_param(ij) for ij in row]
                    for row in mixer_cal.correction_matrix
                ]
            ),
        )

    def _load_precompensation(self, precomp: Precompensation) -> PrecompensationInfo:
        return PrecompensationInfo(
            exponential=precomp.exponential,
            high_pass=precomp.high_pass,
            bounce=precomp.bounce,
            FIR=precomp.FIR,
        )

    def _load_amplifier_pump(
        self, amp_pump: AmplifierPump, channel, ppc_device: DeviceInfo
    ) -> AmplifierPumpInfo:
        return AmplifierPumpInfo(
            ppc_device=ppc_device,
            pump_frequency=self.opt_param(amp_pump.pump_frequency),
            pump_power=self.opt_param(amp_pump.pump_power),
            pump_on=amp_pump.pump_on,
            pump_filter_on=amp_pump.pump_filter_on,
            cancellation_on=amp_pump.cancellation_on,
            cancellation_phase=self.opt_param(amp_pump.cancellation_phase),
            cancellation_attenuation=self.opt_param(amp_pump.cancellation_attenuation),
            cancellation_source=amp_pump.cancellation_source,
            cancellation_source_frequency=amp_pump.cancellation_source_frequency,
            alc_on=amp_pump.alc_on,
            probe_on=amp_pump.probe_on,
            probe_frequency=self.opt_param(amp_pump.probe_frequency),
            probe_power=self.opt_param(amp_pump.probe_power),
            channel=channel,
        )

    def _load_signal(self, signal: ExperimentSignal):
        signal_info = SignalInfo(uid=signal.uid)
        mapped_ls_path: str = self._signal_mappings[signal.uid]
        mapped_ls = self._setup_helper.logical_signal_by_path(mapped_ls_path)

        signal_info.device = self._device_info.device_by_ls(mapped_ls)

        physical_channel = (
            self._setup_helper.instruments.physical_channel_by_logical_signal(mapped_ls)
        )

        if physical_channel.direction == IODirection.IN:
            signal_info.type = SignalInfoType.INTEGRATION
        elif physical_channel.type == PhysicalChannelType.RF_CHANNEL:
            signal_info.type = SignalInfoType.RF
        else:
            signal_info.type = SignalInfoType.IQ

        calibration = self._get_signal_calibration(signal, mapped_ls)
        if calibration is not None:
            signal_info.port_delay = self.opt_param(
                calibration.port_delay, nt_only=True
            )
            signal_info.delay_signal = calibration.delay_signal

            if (oscillator := calibration.oscillator) is not None:
                signal_info.oscillator = self._load_oscillator(oscillator)

            signal_info.voltage_offset = self.opt_param(
                calibration.voltage_offset, nt_only=True
            )

            if (mixer_cal := calibration.mixer_calibration) is not None:
                signal_info.mixer_calibration = self._load_mixer_cal(mixer_cal)
            if (precomp := calibration.precompensation) is not None:
                signal_info.precompensation = self._load_precompensation(precomp)

            signal_info.lo_frequency = self.opt_param(
                calibration.local_oscillator_frequency, nt_only=True
            )

            if isinstance(signal_range := calibration.range, Quantity):
                signal_info.signal_range = SignalRange(
                    signal_range.value, signal_range.unit
                )
            elif signal_range is not None:
                signal_info.signal_range = SignalRange(value=signal_range, unit=None)
            else:
                signal_info.signal_range = None
            signal_info.port_mode = calibration.port_mode
            signal_info.threshold = calibration.threshold
            signal_info.amplitude = self.opt_param(calibration.amplitude, nt_only=True)
            if (amp_pump := calibration.amplifier_pump) is not None:
                if physical_channel.direction != IODirection.IN:
                    _logger.warning(
                        "'amplifier_pump' calibration for logical signal %s will be ignored - "
                        "only applicable to acquire lines",
                        mapped_ls_path,
                    )
                elif (ppc_connection := self._ppc_connections.get(mapped_ls)) is None:
                    _logger.warning(
                        "'amplifier_pump' calibration for logical signal %s will be ignored - "
                        "no PPC is connected to it",
                        mapped_ls_path,
                    )
                else:
                    channel = ppc_connection.channel
                    device = self._device_info.device_mapping[ppc_connection.device.uid]
                    signal_info.amplifier_pump = self._load_amplifier_pump(
                        amp_pump, channel, device
                    )

            # Output router and adder (RTR SHFSG/QC). Requires: RTR option
            if calibration.output_routing:
                for port in physical_channel.ports:
                    if (
                        not re.match(r"SGCHANNELS/\d/OUTPUT", port.path)
                        and signal_info.device.device_type != DeviceType.SHFSG
                    ):
                        msg = f"Error on signal {mapped_ls_path}: Output routing can be only applied to output SGCHANNELS."
                        raise LabOneQException(msg)
            output_routers_per_channel = defaultdict(set)
            for output_router in calibration.output_routing:
                source_signal = output_router.source_signal
                if LogicalSignalGroups_Path not in source_signal:
                    source_signal = insert_logical_signal_prefix(source_signal)

                try:
                    source_signal = self._setup_helper.logical_signal_by_path(
                        source_signal
                    )
                except KeyError:
                    msg = f"Error on signal {mapped_ls_path}: Output routing source signal {output_router.source_signal} does not exist."
                    raise LabOneQException(msg) from None
                from_pc = (
                    self._setup_helper.instruments.physical_channel_by_logical_signal(
                        source_signal
                    )
                )
                from_device = self._device_info.device_by_ls(source_signal)

                if from_device != signal_info.device:
                    msg = f"Error on signal {mapped_ls_path}: Output routing can be only applied within the same device SGCHANNELS: {signal_info.device.uid} != {from_pc.group}"
                    raise LabOneQException(msg)
                assert len(physical_channel.ports) == 1 and len(from_pc.ports) == 1, (
                    "Output SG physical channels must have exactly one port."
                )
                to_port = physical_channel.ports[0]
                from_port = from_pc.ports[0]
                if to_port == from_port:
                    msg = f"Error on signal {mapped_ls_path}: Output routing source is the same as the target channel: {from_port.path}"
                    raise LabOneQException(msg)
                if from_port.channel in output_routers_per_channel[to_port.channel]:
                    msg = f"Error on signal {mapped_ls_path}: Duplicate output routing from channel {from_port.channel}."
                    raise LabOneQException(msg)
                output_routers_per_channel[to_port.channel].add(from_port.channel)
                if len(output_routers_per_channel[to_port.channel]) > 3:
                    msg = f"Error on signal {mapped_ls_path}: Maximum of three signals can be routed per output SGCHANNELS."
                    raise LabOneQException(msg)
                if isinstance(output_router.amplitude, Parameter):
                    if isinstance(output_router.amplitude, SweepParameter):
                        if (
                            output_router.amplitude.uid
                            not in self._sweep_params_min_maxes
                        ):
                            self._sweep_params_min_maxes[
                                output_router.amplitude.uid
                            ] = (
                                np.min(output_router.amplitude.values),
                                np.max(output_router.amplitude.values),
                            )
                        min_val, max_val = self._sweep_params_min_maxes[
                            output_router.amplitude.uid
                        ]
                    elif isinstance(output_router.amplitude, LinearSweepParameter):
                        min_val = output_router.amplitude.start
                        max_val = output_router.amplitude.stop
                    if min_val < 0.0 or max_val > 1.0:
                        msg = "Output route amplitude value must be between 0 and 1."
                        raise LabOneQException(
                            f"Invalid sweep parameter {output_router.amplitude.uid}: {msg}"
                        )
                signal_info.output_routing.append(
                    OutputRoute(
                        to_channel=to_port.channel,
                        to_signal=signal_info.uid,
                        from_channel=from_port.channel,
                        from_signal=self._ls_to_exp_sig_mapping.get(
                            self._setup_helper.logical_signal_path(source_signal)
                        ),
                        amplitude=self.opt_param(output_router.amplitude, nt_only=True),
                        phase=self.opt_param(output_router.phase, nt_only=True),
                    )
                )
            if calibration.automute:
                if physical_channel.direction == IODirection.IN:
                    raise LabOneQException(
                        "Automute can only be applied to output channels."
                    )
                if calibration.port_mode not in (PortMode.RF, None):
                    raise LabOneQException(
                        "Automute can only be applied when with `PortMode.RF`."
                    )
                if not DeviceType.from_device_info_type(
                    signal_info.device.device_type
                ).supports_output_mute:
                    msg = f"Automute is not available on device {signal_info.device.device_type.value.upper()}."
                    raise LabOneQException(msg)
            signal_info.automute = calibration.automute

        signal_info.channels = sorted((port.channel for port in physical_channel.ports))
        signal_info.channel_to_port = {
            str(port.channel): port.path for port in physical_channel.ports
        }

        self._signal_infos[signal.uid] = signal_info

    def _add_parameter(self, value: Parameter, nt_only=False) -> ParameterInfo:
        if isinstance(value, LinearSweepParameter):
            if value.count > 1:
                step = (value.stop - value.start) / (value.count - 1)
            else:
                step = 0
            param_info = ParameterInfo(
                uid=value.uid,
                start=value.start,
                step=step,
                axis_name=value.axis_name,
                values=np.linspace(value.start, value.stop, value.count),
            )
        else:
            assert isinstance(value, SweepParameter)
            param_info = ParameterInfo(
                uid=value.uid,
                values=value.values,
                axis_name=value.axis_name,
            )

            for parent_param in value.driven_by:
                self._add_parameter(parent_param)

        if value.uid not in self._params:
            self._params[value.uid] = param_info
        elif self._params[value.uid] != param_info:
            raise LabOneQException(
                f"Found multiple, inconsistent values for parameter {value.uid} with same UID."
            )
        if nt_only:
            self._nt_only_params.add(param_info.uid)
        self._dsl_parameters[value.uid] = value
        return param_info

    def opt_param(self, value: T | Parameter, nt_only=False) -> T | ParameterInfo:
        """Pass through numbers, but convert `Parameter` to `ParameterInfo`

        Args:
            value: the value that is possibly a parameter
            nt_only: whether the quantity that the value will be assigned to can only be
              possibly swept in near-time.

        Returns:
            the value or a `ParameterInfo`
        """
        if isinstance(value, Parameter):
            return self._add_parameter(value, nt_only)
        return value

    def opt_param_ref(
        self, value: float | int | complex | Parameter
    ) -> float | int | complex | UIDReference:
        val_or_param_info = self.opt_param(value, False)
        if isinstance(val_or_param_info, ParameterInfo):
            return UIDReference(val_or_param_info.uid)
        return val_or_param_info

    def _walk_sections(
        self,
        section: Section,
        section_uid_map: dict[str, Section],
    ):
        if isinstance(section, AcquireLoopRt):
            if self._acquisition_type is not None:
                raise LabOneQException(
                    "Experiment must not contain multiple real-time averaging loops"
                )
            self._acquisition_type = section.acquisition_type

        if section.uid in section_uid_map and section != section_uid_map[section.uid]:
            raise LabOneQException(
                f"Duplicate section uid '{section.uid}' found in experiment"
            )
        section_uid_map[section.uid] = section

        self._load_section(section)
        for child in section.children:
            if isinstance(child, Section):
                self._walk_sections(child, section_uid_map)
            else:
                self._visit_operation(child, section)

    def _visit_operation(
        self,
        operation: SignalOperation,
        section: Section,
    ):
        """Visit a signal operation and extract parameter information."""
        if isinstance(operation, Delay):
            self.opt_param(operation.time)

        elif isinstance(operation, Acquire):
            # Parametrized fields
            pulse_params = (
                operation.pulse_parameters
                if isinstance(operation.pulse_parameters, list)
                else [operation.pulse_parameters]
            )
            for params in pulse_params:
                for param_value in (params or {}).values():
                    self.opt_param(param_value)

        elif isinstance(operation, PlayPulse):
            # Parametrized fields
            for param_value in (operation.pulse_parameters or {}).values():
                self.opt_param(param_value)
            self.opt_param(operation.length)
            self.opt_param(operation.amplitude)
            self.opt_param(operation.phase)
            self.opt_param(operation.increment_oscillator_phase)
            self.opt_param(operation.set_oscillator_phase)

    def _load_section(
        self,
        section: Section,
    ):
        count = None

        if hasattr(section, "count"):
            count = int(section.count)  # cast to int; user may provide float via pow()

        if isinstance(section, Sweep):
            sweep_params_equal_len = all(
                len(section.parameters[0]) == len(section.parameters[i])
                for i in range(len(section.parameters))
            )
            if not sweep_params_equal_len:
                raise LabOneQException(
                    f"Error in experiment section '{section.uid}': Parallel executed sweep parameters must be of same length. {section.uid}"
                )
            for parameter in section.parameters:
                self._add_parameter(parameter)
                if isinstance(parameter, (SweepParameter, LinearSweepParameter)):
                    count = len(parameter)

        if isinstance(section, PrngLoop):
            count = section.prng_sample.count

        _auto_chunking = getattr(section, "auto_chunking", False)
        _chunk_count = getattr(section, "chunk_count", 1)
        chunked = _auto_chunking or _chunk_count > 1
        if chunked:
            self._chunking_info = ChunkingInfo(_auto_chunking, _chunk_count, count)

    def _validate_realtime(self, root_sections: list[Section]):
        """Verify that:
        - no near-time section is located inside a real-time section
        - there can be at most one AcquireLoopRt
        - if there is one, it must be the real-time boundary.

        With these conditions, execution_type=None is resolved to either NT or RT.
        """

        def traverse_set_execution_type_and_check_rt_loop(
            section: Section, in_realtime: bool
        ):
            if not isinstance(section, Section):
                return
            if isinstance(section, AcquireLoopRt):
                in_realtime = True
            execution_type = getattr(section, "execution_type", None)
            if execution_type == ExecutionType.NEAR_TIME and in_realtime:
                raise LabOneQException(
                    f"Near-time section '{section.uid}' is nested inside a RT section"
                )
            if execution_type == ExecutionType.REAL_TIME and not in_realtime:
                raise LabOneQException(
                    f"Section '{section.uid}' is marked as real-time, but it is"
                    f" located outside the RT averaging loop"
                )
            if in_realtime and isinstance(section, Sweep):
                for parameter in section.parameters:
                    if parameter.uid in self._nt_only_params:
                        raise LabOneQException(
                            f"Parameter {parameter.uid} can't be swept in real-time, it"
                            " is bound to a value that can only be set in near-time"
                        )
            for child in section.children:
                traverse_set_execution_type_and_check_rt_loop(child, in_realtime)

        for root_section in root_sections:
            traverse_set_execution_type_and_check_rt_loop(
                root_section, in_realtime=False
            )

    def _resolve_oscillator_modulation_type(self, experiment_info: ExperimentInfo):
        for signal in experiment_info.signals:
            if (osc := signal.oscillator) is None:
                continue
            if osc.is_hardware is not None:
                continue
            is_qa_device = DeviceType.from_device_info_type(
                signal.device.device_type
            ).is_qa_device
            if is_qa_device and "LRT" not in signal.device.dev_opts:
                osc.is_hardware = acq_type.is_spectroscopy(self._acquisition_type)
            elif (
                signal.type is SignalInfoType.RF
                and signal.device.device_type is DeviceInfoType.HDAWG
            ):
                # for HDAWG RF signals SW modulation tends to be more useful
                osc.is_hardware = False
            else:
                osc.is_hardware = True
            _logger.info(
                f"Resolved modulation type of oscillator '{osc.uid}' on signal"
                f" '{signal.uid}' to {'HARDWARE' if osc.is_hardware else 'SOFTWARE'}"
            )


class AttributeOverrider(object):
    def __init__(self, base, overrider):
        if overrider is None:
            raise RuntimeError("overrider must not be none")

        self._overrider = overrider
        self._base = base

    def __getattr__(self, attr):
        if hasattr(self._overrider, attr):
            overrider_value = getattr(self._overrider, attr)
            if overrider_value is not None or self._base is None:
                return overrider_value
        if self._base is not None and hasattr(self._base, attr):
            return getattr(self._base, attr)
        raise AttributeError(
            f"Field {attr} not found on overrider {self._overrider} (type {type(self._overrider)}) nor on base {self._base}"
        )
