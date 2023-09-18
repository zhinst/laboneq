# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import itertools
import logging
from types import SimpleNamespace
from typing import Any, Dict, Tuple

from laboneq._utils import ensure_list, id_generator
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AveragingMode
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
    ParameterInfo,
    PrecompensationInfo,
    PulseDef,
    SectionInfo,
    SectionSignalPulse,
    SignalInfo,
    SignalInfoType,
    SignalRange,
    SweepParamRef,
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

    def load_experiment(self) -> ExperimentInfo:
        self._check_physical_channel_calibration_conflict()
        for signal in self._experiment.signals:
            self._load_signal(signal)

        section_uid_map = {}
        acquisition_type_map = {}
        root_sections = [
            self._walk_sections(section, None, section_uid_map, acquisition_type_map)
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
        is_hw = oscillator.modulation_type == ModulationType.HARDWARE
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
            self._parameter_parents[value.uid] = [
                parent_param.uid for parent_param in value.driven_by
            ]

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
    ) -> float | int | complex | SweepParamRef:
        val_or_param_info = self.opt_param(value, False)
        if isinstance(val_or_param_info, ParameterInfo):
            return SweepParamRef(val_or_param_info.uid)
        return val_or_param_info

    def _walk_sections(
        self,
        section: Section,
        acquisition_type,
        section_uid_map: Dict[str, Tuple[Any, int]],
        acquisition_type_map,
    ) -> SectionInfo:
        assert section.uid is not None
        if (
            section.uid in section_uid_map
            and section != section_uid_map[section.uid][0]
        ):
            raise LabOneQException(
                f"Duplicate section uid '{section.uid}' found in experiment"
            )
        current_acquisition_type = acquisition_type

        if hasattr(section, "acquisition_type"):
            current_acquisition_type = section.acquisition_type

        acquisition_type_map[section.uid] = current_acquisition_type

        section_info = self._load_section(
            section,
            current_acquisition_type,
            section_uid_map,
        )

        for index, child_section in enumerate(section.children):
            if not isinstance(child_section, Section):
                continue
            section_info.children.append(
                self._walk_sections(
                    child_section,
                    current_acquisition_type,
                    section_uid_map,
                    acquisition_type_map,
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
            pulses = ensure_list(getattr(operation, "pulse"))
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
            pulses = ensure_list(getattr(operation, "kernel") or [])
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
                f" operation with handle '{getattr(operation, 'handle')}'."
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

        operation_pulse_parameters = getattr(operation, "pulse_parameters")
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

        if hasattr(section, "parameters"):
            for parameter in section.parameters:
                section_parameters.append(self._add_parameter(parameter))
                if hasattr(parameter, "count"):
                    count = parameter.count
                elif hasattr(parameter, "values"):
                    count = len(parameter.values)
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

            # the parent should now have been added correctly
            assert parent_id in sweeps_by_parameter

            for sweep in sweeps_by_parameter[parent_id]:
                assert parameter not in sweep.parameters
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
                if acquire_loop is not None:
                    raise LabOneQException("Found multiple AcquireLoopRt")
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
        if parent.count is not None and parent.averaging_mode is None:
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
