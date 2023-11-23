# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import itertools
import logging
import re
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, Tuple
import numpy as np

from laboneq.core.path import LogicalSignalGroups_Path, insert_logical_signal_prefix
from laboneq._utils import UIDReference, ensure_list, id_generator
from laboneq.compiler import DeviceType
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType, AveragingMode
from laboneq.core.types.units import Quantity
from laboneq.data.calibration import (
    AmplifierPump,
    MixerCalibration,
    ModulationType,
    Oscillator,
    Precompensation,
    SignalCalibration,
)
from laboneq.data.compilation_job import (
    AcquireInfo,
    AmplifierPumpInfo,
    ExperimentInfo,
    Marker,
    MixerCalibrationInfo,
    OscillatorInfo,
    OutputRoute,
    ParameterInfo,
    PrecompensationInfo,
    PulseDef,
    SectionInfo,
    SectionSignalPulse,
    SignalInfo,
    SignalInfoType,
    SignalRange,
    PRNGInfo,
)
from laboneq.data.experiment_description import (
    Acquire,
    Delay,
    ExecutionType,
    Experiment,
    ExperimentSignal,
    PlayPulse,
    Reserve,
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

_logger = logging.getLogger(__name__)


class ExperimentInfoBuilder:
    def __init__(
        self,
        experiment: Experiment,
        device_setup: Setup,
        signal_mappings: Dict[str, str],
    ):
        self._experiment = copy.deepcopy(experiment)
        self._device_setup = copy.deepcopy(device_setup)
        self._signal_mappings = copy.deepcopy(signal_mappings)
        self._ls_to_exp_sig_mapping = {
            ls: exp for exp, ls in self._signal_mappings.items()
        }
        self._params: dict[str, ParameterInfo] = {}
        self._nt_only_params = []
        self._oscillators: dict[str, OscillatorInfo] = {}
        self._signal_infos: dict[str, SignalInfo] = {}
        self._pulse_defs: dict[str, PulseDef] = {}

        self._device_info = DeviceInfoBuilder(self._device_setup)
        self._setup_helper = SetupHelper(self._device_setup)

        self._section_operations_to_add = []
        self._ppc_connections = self._setup_helper.ppc_connections()

        self._parameter_parents: dict[str, list[str]] = {}
        self._sweep_params_min_maxes: dict[str, tuple[float, float]] = {}
        self._acquisition_type = None

    def load_experiment(self) -> ExperimentInfo:
        self._check_physical_channel_calibration_conflict()
        for signal in self._experiment.signals:
            self._load_signal(signal)

        section_uid_map = {}
        root_sections = [
            self._walk_sections(section, section_uid_map)
            for section in self._experiment.sections
        ]

        # Need to defer the insertion of section operations. In sequential averaging mode,
        # the tree-walking order might otherwise make us visit operations which depend on parameters
        # we haven't seen the sweep of yet.
        # todo (Pol): this is no longer required, load operations together with sections
        for (
            section,
            section_info,
            acquisition_type,
        ) in self._section_operations_to_add:
            self._load_section_operations(section, section_info, acquisition_type)

        self._sweep_all_derived_parameters(root_sections)
        self._validate_realtime(root_sections)

        experiment_info = ExperimentInfo(
            uid=self._experiment.uid,
            devices=list(self._device_info.device_mapping.values()),
            signals=sorted(self._signal_infos.values(), key=lambda s: s.uid),
            sections=root_sections,
            global_leader_device=self._device_info.global_leader,
            pulse_defs=sorted(self._pulse_defs.values(), key=lambda s: s.uid),
        )
        self._resolve_seq_averaging(experiment_info)
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
            # "mixer_calibration",
            # "precompensation",
            # "amplifier_pump"
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
                        exp_cal = self._experiment.calibration.items.get(exp_signal.uid)
                        if exp_cal is None:
                            continue
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
                    f"Found multiple, inconsistent oscillators with same UID  {oscillator.uid}."
                )
            oscillator_info = self._oscillators[oscillator.uid]
        else:
            self._oscillators[oscillator.uid] = oscillator_info

        return oscillator_info

    def _load_mixer_cal(self, mixer_cal: MixerCalibration) -> MixerCalibrationInfo:
        return MixerCalibrationInfo(
            mixer_cal.voltage_offsets, mixer_cal.correction_matrix
        )

    def _load_precompensation(self, precomp: Precompensation) -> PrecompensationInfo:
        return PrecompensationInfo(
            exponential=precomp.exponential,
            high_pass=precomp.high_pass,
            bounce=precomp.bounce,
            FIR=precomp.FIR,
        )

    def _load_amplifier_pump(
        self, amp_pump: AmplifierPump, channel
    ) -> AmplifierPumpInfo:
        return AmplifierPumpInfo(
            pump_freq=self.opt_param(amp_pump.pump_freq, nt_only=True),
            pump_power=self.opt_param(amp_pump.pump_power, nt_only=True),
            cancellation=amp_pump.cancellation,
            alc_engaged=amp_pump.alc_engaged,
            use_probe=amp_pump.use_probe,
            probe_frequency=self.opt_param(amp_pump.probe_frequency, nt_only=True),
            probe_power=self.opt_param(amp_pump.probe_power, nt_only=True),
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
        else:
            if physical_channel.type == PhysicalChannelType.RF_CHANNEL:
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

            signal_info.voltage_offset = calibration.voltage_offset

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
                    signal_info.amplifier_pump = self._load_amplifier_pump(
                        amp_pump, channel
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
                    from_pc = self._setup_helper.instruments.physical_channel_by_logical_signal(
                        source_signal
                    )
                except RuntimeError:
                    msg = f"Error on signal {mapped_ls_path}: Output routing source signal {output_router.source_signal} does not exist."
                    raise LabOneQException(msg) from None
                if from_pc.group != signal_info.device.uid:
                    msg = f"Error on signal {mapped_ls_path}: Output routing can be only applied within the same device SGCHANNELS: {signal_info.device.uid} != {from_pc.group}"
                    raise LabOneQException(msg)
                assert (
                    len(physical_channel.ports) == 1 and len(from_pc.ports) == 1
                ), "Output SG physical channels must have exactly one port."
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
                            source_signal, None
                        ),
                        amplitude=self.opt_param(output_router.amplitude, nt_only=True),
                        phase=self.opt_param(output_router.phase, nt_only=True),
                    )
                )

        signal_info.channels = sorted((port.channel for port in physical_channel.ports))

        self._signal_infos[signal.uid] = signal_info

    def _add_parameter(
        self, value: Parameter | None, nt_only=False
    ) -> float | ParameterInfo | None:
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

            seen = set()
            self._parameter_parents[value.uid] = []
            for parent_param in value.driven_by:
                uid = parent_param.uid
                if uid in seen:
                    continue
                self._parameter_parents[value.uid].append(uid)
                seen.add(uid)

        if value.uid not in self._params:
            self._params[value.uid] = param_info
        elif self._params[value.uid] != param_info:
            raise LabOneQException(
                f"Found multiple, inconsistent values for parameter {value.uid} with same UID."
            )
        if nt_only and param_info not in self._nt_only_params:
            self._nt_only_params.append(param_info.uid)

        return param_info

    def opt_param(
        self, value: float | int | complex | Parameter, nt_only=False
    ) -> float | int | complex | ParameterInfo:
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
        section_uid_map: Dict[str, Tuple[Any, int]],
    ) -> SectionInfo:
        assert section.uid is not None
        if (
            section.uid in section_uid_map
            and section != section_uid_map[section.uid][0]
        ):
            raise LabOneQException(
                f"Duplicate section uid '{section.uid}' found in experiment"
            )

        if hasattr(section, "acquisition_type"):
            if self._acquisition_type is not None:
                raise LabOneQException(
                    "Experiment must not contain multiple real-time averaging loops"
                )
            self._acquisition_type = section.acquisition_type

        section_info = self._load_section(
            section,
            self._acquisition_type,
            section_uid_map,
        )

        for child_section in section.children:
            if not isinstance(child_section, Section):
                continue
            section_info.children.append(
                self._walk_sections(
                    child_section,
                    section_uid_map,
                )
            )
        return section_info

    def _load_markers(self, operation):
        markers_raw = getattr(operation, "marker", None) or {}
        return [
            Marker(
                k,
                enable=v.get("enable"),
                start=v.get("start"),
                length=v.get("length"),
                pulse_id=v.get("waveform", {}).get("$ref", None),
            )
            for k, v in markers_raw.items()
        ]

    def _load_ssp(
        self,
        operation: SignalOperation,
        signal_info: SignalInfo,
        auto_pulse_id,
        acquisition_type,
        section: SectionInfo,
    ):
        if isinstance(operation, Delay):
            pulse_offset = self.opt_param(operation.time)
            precompensation_clear = operation.precompensation_clear
            section.pulses.append(
                SectionSignalPulse(
                    signal=signal_info,
                    pulse=None,
                    length=pulse_offset,
                    precompensation_clear=precompensation_clear,
                )
            )
            if signal_info not in section.signals:
                section.signals.append(signal_info)
            return
        if isinstance(operation, Reserve):
            if signal_info not in section.signals:
                section.signals.append(signal_info)
            return

        assert isinstance(operation, (PlayPulse, Acquire))
        pulses = []
        markers = self._load_markers(operation)

        length = getattr(operation, "length", None)
        operation_length = self.opt_param(length)

        if hasattr(operation, "pulse"):
            pulses = ensure_list(operation.pulse)
            if len(pulses) > 1:
                raise RuntimeError(
                    f"Only one pulse can be provided for pulse play command in section"
                    f" {section.uid}."
                )
        if pulses == [None] and markers:
            # generate a zero amplitude pulse to play the markers
            # TODO: generate a proper constant pulse here
            pulses = [pulse] = [SimpleNamespace()]
            pulse.uid = next(auto_pulse_id)
            pulse.function = "const"
            pulse.amplitude = 0.0
            pulse.length = max([m.start + m.length for m in markers])
            pulse.can_compress = False

        if hasattr(operation, "kernel"):
            pulses = ensure_list(operation.kernel or [])
            kernel_count = len(pulses)
            if signal_info.kernel_count is None:
                signal_info.kernel_count = kernel_count
            elif signal_info.kernel_count != kernel_count:
                raise LabOneQException(
                    f"Inconsistent count of integration kernels on signal {signal_info.uid}"
                )
        if len(pulses) == 0 and length is not None:
            # TODO: generate a proper constant pulse here
            pulses = [pulse] = [SimpleNamespace()]
            pulse.uid = next(auto_pulse_id)
            pulse.length = length

        assert pulses is not None and isinstance(pulses, list)

        pulse_group = None if len(pulses) == 1 else id_generator("pulse_group")
        if markers:
            for m in markers:
                if m.pulse_id is None:
                    assert len(pulses) == 1 and pulses[0] is not None
                    m.pulse_id = pulses[0].uid

        if hasattr(operation, "handle") and len(pulses) == 0:
            raise RuntimeError(
                f"Either 'kernel' or 'length' must be provided for the acquire"
                f" operation with handle '{operation.handle}'."
            )

        amplitude = self.opt_param(getattr(operation, "amplitude", None))
        phase = self.opt_param(getattr(operation, "phase", None))
        increment_oscillator_phase = self.opt_param(
            getattr(operation, "increment_oscillator_phase", None)
        )
        set_oscillator_phase = self.opt_param(
            getattr(operation, "set_oscillator_phase", None)
        )

        acquire_params = None
        if hasattr(operation, "handle"):
            acquire_params = AcquireInfo(
                handle=operation.handle,
                acquisition_type=acquisition_type.value,
            )

        operation_pulse_parameters = operation.pulse_parameters
        if operation_pulse_parameters is not None:
            operation_pulse_parameters_list = ensure_list(operation_pulse_parameters)
            operation_pulse_parameters_list = [
                {
                    param: self.opt_param_ref(val)
                    for param, val in operation_pulse_parameters.items()
                }
                if operation_pulse_parameters is not None
                else {}
                for operation_pulse_parameters in operation_pulse_parameters_list
            ]
        else:
            operation_pulse_parameters_list = [None] * len(pulses)

        for pulse, op_pars in zip(pulses, operation_pulse_parameters_list):
            if pulse is not None:
                pulse_def = self._add_pulse(pulse)

                pulse_pulse_parameters = getattr(pulse, "pulse_parameters", {})
                if pulse_pulse_parameters is not None:
                    pulse_pulse_parameters = {
                        param: self.opt_param_ref(val)
                        for param, val in pulse_pulse_parameters.items()
                    }

                section.pulses.append(
                    SectionSignalPulse(
                        signal=signal_info,
                        pulse=pulse_def,
                        length=operation_length,
                        amplitude=amplitude,
                        phase=phase,
                        increment_oscillator_phase=increment_oscillator_phase,
                        set_oscillator_phase=set_oscillator_phase,
                        precompensation_clear=False,
                        play_pulse_parameters=op_pars,
                        pulse_pulse_parameters=pulse_pulse_parameters,
                        acquire_params=acquire_params,
                        markers=markers,
                        pulse_group=pulse_group,
                    )
                )
                if signal_info not in section.signals:
                    section.signals.append(signal_info)

            elif (
                getattr(operation, "increment_oscillator_phase", None) is not None
                or getattr(operation, "set_oscillator_phase", None) is not None
                or getattr(operation, "phase", None) is not None
            ):
                # virtual Z gate
                if operation.phase is not None:
                    raise LabOneQException(
                        "Phase argument has no effect for virtual Z gates."
                    )

                increment_oscillator_phase = self.opt_param(
                    operation.increment_oscillator_phase
                )
                set_oscillator_phase = self.opt_param(operation.set_oscillator_phase)
                for par in [
                    "precompensation_clear",
                    "amplitude",
                    "phase",
                    "pulse_parameters",
                    "handle",
                    "length",
                ]:
                    if getattr(operation, par, None) is not None:
                        raise LabOneQException(
                            f"parameter {par} not supported for virtual Z gates"
                        )

                section.pulses.append(
                    SectionSignalPulse(
                        signal=signal_info,
                        precompensation_clear=False,
                        set_oscillator_phase=set_oscillator_phase,
                        increment_oscillator_phase=increment_oscillator_phase,
                    )
                )
                if signal_info not in section.signals:
                    section.signals.append(signal_info)

    def _load_section_operations(
        self,
        section: Section,
        section_info: SectionInfo,
        acquisition_type,
    ):
        _auto_pulse_id = (f"{section.uid}__auto_pulse_{i}" for i in itertools.count())

        for operation in section.children:
            if not isinstance(operation, SignalOperation):
                continue
            signal_info = self._signal_infos[operation.signal]
            self._load_ssp(
                operation,
                signal_info,
                _auto_pulse_id,
                acquisition_type,
                section_info,
            )

    def _load_section(
        self,
        section: Section,
        exp_acquisition_type,
        section_uid_map: Dict[str, Tuple[Any, int]],
    ) -> SectionInfo:
        if section.uid not in section_uid_map:
            section_uid_map[section.uid] = (section, 0)
            instance_id = section.uid
        else:
            visit_count = section_uid_map[section.uid][1] + 1
            instance_id = f"{section.uid}_{visit_count}"
            section_uid_map[section.uid] = (section, visit_count)

        count = None

        if hasattr(section, "count"):
            count = int(section.count)  # cast to int; user may provide float via pow()

        section_parameters = []

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
                section_parameters.append(self._add_parameter(parameter))
                if isinstance(parameter, (SweepParameter, LinearSweepParameter)):
                    count = len(parameter)
                if count < 1:
                    raise ValueError(
                        f"Repeat count must be at least 1, but section {section.uid} has count={count}"
                    )
                if (
                    section.execution_type is not None
                    and section.execution_type == ExecutionType.REAL_TIME
                    and parameter.uid in self._nt_only_params
                ):
                    raise LabOneQException(
                        f"Parameter {parameter.uid} can't be swept in real-time, it is bound to a value "
                        f"that can only be set in near-time"
                    )
        execution_type = section.execution_type
        align = section.alignment
        on_system_grid = section.on_system_grid
        length = section.length
        averaging_mode = getattr(section, "averaging_mode", None)
        repetition_mode = getattr(section, "repetition_mode", None)
        repetition_time = getattr(section, "repetition_time", None)
        reset_oscillator_phase = (
            getattr(section, "reset_oscillator_phase", None) or False
        )
        handle = getattr(section, "handle", None)
        state = getattr(section, "state", None)
        local = getattr(section, "local", None)
        user_register = getattr(section, "user_register", None)
        assert section.trigger is not None
        triggers = [
            {"signal_id": k, "state": v["state"]} for k, v in section.trigger.items()
        ]
        chunk_count = getattr(section, "chunk_count", 1)

        prng_seed_info = None
        if hasattr(section, "prng"):
            prng_seed_info = PRNGInfo(section.prng.range, section.prng.seed)
            on_system_grid = True

        draw_from_prng = False
        if hasattr(section, "prng_sample"):
            prng_sample = section.prng_sample
            count = prng_sample.count
            draw_from_prng = True

        this_acquisition_type = None
        if any(isinstance(operation, Acquire) for operation in section.children):
            # an acquire event - add acquisition_types
            this_acquisition_type = exp_acquisition_type

        play_after = getattr(section, "play_after", None)
        if play_after:
            section_uid = lambda x: x.uid if hasattr(x, "uid") else x
            play_after = section_uid(play_after)
            if isinstance(play_after, list):
                play_after = [section_uid(s) for s in play_after]
        section_info = SectionInfo(
            uid=instance_id,
            length=length,
            alignment=align,
            count=count,
            chunk_count=chunk_count,
            handle=handle,
            user_register=user_register,
            state=state,
            local=local,
            execution_type=execution_type,
            averaging_mode=averaging_mode,
            acquisition_type=this_acquisition_type,
            repetition_mode=repetition_mode,
            repetition_time=repetition_time,
            reset_oscillator_phase=reset_oscillator_phase,
            on_system_grid=on_system_grid,
            triggers=triggers,
            play_after=play_after,
            parameters=section_parameters,
            prng=prng_seed_info,
            draw_prng=draw_from_prng,
        )

        self._section_operations_to_add.append(
            (section, section_info, exp_acquisition_type)
        )

        return section_info

    def _add_pulse(self, pulse) -> PulseDef:
        if pulse.uid not in self._pulse_defs:
            function = getattr(pulse, "function", None)
            length = getattr(pulse, "length", None)
            samples = getattr(pulse, "samples", None)

            amplitude = self.opt_param(getattr(pulse, "amplitude", None))
            if amplitude is None:
                amplitude = 1.0
            can_compress = getattr(pulse, "can_compress", False)

            self._pulse_defs[pulse.uid] = PulseDef(
                uid=pulse.uid,
                function=function,
                length=length,
                amplitude=amplitude,
                can_compress=can_compress,
                samples=samples,
            )
        return self._pulse_defs[pulse.uid]

    @staticmethod
    def _find_sweeps_by_parameter(
        root_sections: list[SectionInfo],
    ) -> dict[str, list[SectionInfo]]:
        sweeps_by_parameter = {}
        sections_to_visit = root_sections[:]
        while len(sections_to_visit):
            section = sections_to_visit.pop()
            for param in section.parameters:
                sweeps_by_parameter.setdefault(param.uid, []).append(section)
            sections_to_visit.extend(section.children)
        return sweeps_by_parameter

    def _sweep_derived_parameter(
        self,
        parameter: ParameterInfo,
        sweeps_by_parameter: dict[str, list[SectionInfo]],
    ):
        if parameter.uid in sweeps_by_parameter:
            return

        # This parameter is not swept directly, but derived from a another parameter;
        # we must add it to the corresponding loop.
        parameter = self._params[parameter.uid]
        parents = self._parameter_parents[parameter.uid]
        for parent_id in parents:
            parent = self._params.get(parent_id)
            if parent is None:
                raise LabOneQException(
                    f"Parameter '{parameter.uid}' is driven by a parameter '{parent_id}' which is unknown."
                )

            self._sweep_derived_parameter(parent, sweeps_by_parameter)

            # The parent should now have been added correctly. If it has not, than
            # that means that the parent (or its parents in turn) are not used anywhere
            # in the experiment. We can just ignore them.
            if parent_id not in sweeps_by_parameter:
                continue

            for sweep in sweeps_by_parameter[parent_id]:
                if parameter not in sweep.parameters:
                    sweep.parameters.append(parameter)
                    sweeps_by_parameter.setdefault(parameter.uid, []).append(sweep)

    def _sweep_all_derived_parameters(
        self,
        root_sections: list[SectionInfo],
    ):
        sweeps_by_parameter = self._find_sweeps_by_parameter(root_sections)

        for child in self._parameter_parents.keys():
            self._sweep_derived_parameter(self._params[child], sweeps_by_parameter)

    def _validate_realtime(self, root_sections: list[SectionInfo]):
        """Verify that:
        - no near-time section is located inside a real-time section
        - there can be at most one AcquireLoopRt
        - if there is one, it must be the real-time boundary.

        With these conditions, execution_type=None is resolved to either NT or RT.
        """

        acquire_loop = None

        def traverse_set_execution_type_and_check_rt_loop(
            section: SectionInfo, in_realtime: bool
        ):
            if section.execution_type == ExecutionType.NEAR_TIME and in_realtime:
                raise LabOneQException(
                    f"Near-time section '{section.uid}' is nested inside a RT section"
                )
            elif section.execution_type == ExecutionType.REAL_TIME:
                in_realtime = True
            else:
                section.execution_type = (
                    ExecutionType.REAL_TIME if in_realtime else ExecutionType.NEAR_TIME
                )

            if section.averaging_mode is not None:
                nonlocal acquire_loop
                # Note: this should have been checked earlier already, so we make it an
                # assertion rather than a LabOneQException.
                assert acquire_loop is None, "multiple AcquireLoopRt not permitted"
                acquire_loop = section
            for child in section.children:
                traverse_set_execution_type_and_check_rt_loop(child, in_realtime)

        for root_section in root_sections:
            traverse_set_execution_type_and_check_rt_loop(
                root_section, in_realtime=False
            )

        def traverse_check_all_rt_inside_rt_loop(section: SectionInfo):
            if section.averaging_mode is not None:
                return
            assert (
                section.execution_type is not None
            ), "should have been set in first traverse"
            if section.execution_type == ExecutionType.REAL_TIME:
                raise LabOneQException(
                    f"Section '{section.uid}' is marked as real-time, but it is"
                    f" located outside the RT averaging loop"
                )
            if section.execution_type is None:
                section.execution_type = ExecutionType.NEAR_TIME
            for child in section.children:
                traverse_check_all_rt_inside_rt_loop(child)

        for root_section in root_sections:
            traverse_check_all_rt_inside_rt_loop(root_section)

    @staticmethod
    def _find_acquire_loop_with_parent(
        parent: SectionInfo | ExperimentInfo,
    ) -> tuple[SectionInfo | ExperimentInfo, SectionInfo] | None:
        """DFS for the acquire loop"""
        if isinstance(parent, SectionInfo):
            children = parent.children
        else:
            children = parent.sections
        for child in children:
            if child.averaging_mode is not None:
                return parent, child
            if acquire_loop := ExperimentInfoBuilder._find_acquire_loop_with_parent(
                child
            ):
                return acquire_loop
        return None

    @staticmethod
    def _find_innermost_sweep_for_seq_averaging(
        parent: SectionInfo,
    ) -> SectionInfo | None:
        innermost_sweep = None
        for child in parent.children:
            this_innermost_sweep = (
                ExperimentInfoBuilder._find_innermost_sweep_for_seq_averaging(child)
            )
            if innermost_sweep is not None and this_innermost_sweep is not None:
                raise LabOneQException(
                    f"Section '{parent.uid}' has multiple sweeping subsections."
                    f" This is illegal in sequential averaging mode."
                )
            innermost_sweep = this_innermost_sweep
        if innermost_sweep is not None:
            return innermost_sweep
        if (
            parent.count is not None
            and parent.averaging_mode is None
            and not parent.draw_prng  # PRNG loop is considered to be inside shot
        ):
            # this section is a sweep (aka loop but not averaging)
            return parent
        return None

    def _resolve_seq_averaging(self, experiment_info: ExperimentInfo):
        acquire_loop_with_parent = self._find_acquire_loop_with_parent(experiment_info)
        if acquire_loop_with_parent is None:
            return  # no acquire loop
        parent, acquire_loop = acquire_loop_with_parent

        if acquire_loop.averaging_mode != AveragingMode.SEQUENTIAL:
            return

        innermost_sweep = self._find_innermost_sweep_for_seq_averaging(acquire_loop)
        if innermost_sweep is None:
            _logger.debug("Sequential averaging but no real-time sweep")
            return

        current_section = acquire_loop
        while current_section is not innermost_sweep:
            if len(current_section.children) != 1:
                raise LabOneQException(
                    f"Section '{current_section.uid}' has multiple children."
                    " With sequential averaging, the section graph from acquire loop to"
                    " inner-most sweep must be a linear chain, with only a single"
                    " subsection at each level. "
                )
            [current_section] = current_section.children

        # We now know where the acquire loop _should_ go. Let's graft it there.
        # First, remove it from its original location...
        if not isinstance(parent, ExperimentInfo):
            parent.children = acquire_loop.children
        else:
            parent.sections = acquire_loop.children
        # ... and then re-insert it into the bottom of the tree.
        acquire_loop.children = innermost_sweep.children
        innermost_sweep.children = [acquire_loop]

        # Similarly, move the pulses (also children) of the loops. Here, the added
        # caveat is that any pulses directly in the acquire loop have nowhere to go, so
        # we forbid them.
        if acquire_loop.pulses or acquire_loop.signals:
            raise LabOneQException(
                "Pulses directly in the acquire loop are not allowed in sequential "
                "averaging mode. Place them inside the sweep instead."
            )
        acquire_loop.pulses, innermost_sweep.pulses = innermost_sweep.pulses, []
        acquire_loop.signals, innermost_sweep.signals = innermost_sweep.signals, []

        # The acquire loop inherits the sweep's alignment; this is required for
        # fixed-repetition-time shots to be right-aligned.
        #
        # [----------------- sweep iteration 1 ------------------][--- sweep iteration 2 ...
        # [-- shot 1 --][-- shot 2 --][-- shot 3 --][-- shot 4 --][-- shot 1 --][...
        #     [==body==]    [==body==]    [==body==]    [==body==]    [...
        #               |<---------->|
        #               repetition time
        acquire_loop.alignment = innermost_sweep.alignment

        # todo(PW): What about repetition time?
        #  Should repetition time be associated with the outermost sweep?
        #  Currently the scheduler appears to handle this just fine; it picks up the
        #  correct repetition time no matter where it is located in the tree.

    def _resolve_oscillator_modulation_type(self, experiment_info: ExperimentInfo):
        for signal in experiment_info.signals:
            if (osc := signal.oscillator) is None:
                continue
            if osc.is_hardware is not None:
                continue
            is_qa_device = DeviceType.from_device_info_type(
                signal.device.device_type
            ).is_qa_device
            is_spectroscopy_mode = self._acquisition_type in (
                AcquisitionType.SPECTROSCOPY,
                AcquisitionType.SPECTROSCOPY_IQ,
                AcquisitionType.SPECTROSCOPY_PSD,
            )
            if is_qa_device:
                osc.is_hardware = is_spectroscopy_mode
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
