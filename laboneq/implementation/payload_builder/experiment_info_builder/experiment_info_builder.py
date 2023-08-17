# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import itertools
import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AveragingMode
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
    MixerCalibrationInfo,
    OscillatorInfo,
    ParameterInfo,
    PrecompensationInfo,
    PulseDef,
    SectionInfo,
    SectionSignalPulse,
    SignalInfo,
    SignalInfoType,
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
        self._nt_only_params = set()
        self._oscillators: dict[str, OscillatorInfo] = {}
        self._signal_infos: dict[str, SignalInfo] = {}
        self._pulse_defs: dict[str, PulseDef] = {}

        self._device_info = DeviceInfoBuilder(self._device_setup)
        self._setup_helper = SetupHelper(self._device_setup)

        self._section_operations_to_add = []

    def load_experiment(self) -> ExperimentInfo:
        self._check_physical_channel_calibration_conflict()
        for signal in self._experiment.signals:
            self._load_signal(signal)

            # todo: dio & zsync & pqsc

        seq_avg_section, sweep_sections = find_sequential_averaging(self._experiment)
        if seq_avg_section is not None and len(sweep_sections) > 0:
            if len(sweep_sections) > 1:
                raise LabOneQException(
                    f"Sequential averaging section {seq_avg_section.uid} has multiple "
                    f"sweeping subsections: {[s.uid for s in sweep_sections]}. There "
                    f"must be at most one."
                )

            def exchanger_map(section):
                if section is sweep_sections[0]:
                    return seq_avg_section
                if section is seq_avg_section:
                    return sweep_sections[0]
                return section

        else:
            exchanger_map = lambda section: section

        section_uid_map = {}
        acquisition_type_map = {}
        root_sections = [
            self._walk_sections(
                section, None, section_uid_map, acquisition_type_map, exchanger_map
            )
            for section in self._experiment.sections
        ]

        # Need to defer the insertion of section operations. In sequential averaging mode,
        # the tree-walking order might otherwise make us visit operations which depend on parameters
        # we haven't seen the sweep of yet.
        for (
            section,
            section_info,
            acquisition_type,
        ) in self._section_operations_to_add:
            self._load_section_operations(
                section, section_info, acquisition_type, exchanger_map
            )

        return ExperimentInfo(
            uid=self._experiment.uid,
            devices=list(self._device_info.device_mapping.values()),
            signals=sorted(self._signal_infos.values(), key=lambda s: s.uid),
            sections=root_sections,
            global_leader_device=self._device_info.global_leader,
            pulse_defs=sorted(self._pulse_defs.values(), key=lambda s: s.uid),
        )

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
                    f"Experiment signal '{signal.uid}' is not mapped to a logical signal."
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
                        if exp_signal.is_calibrated()
                        and getattr(exp_signal.calibration, field_) is not None
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

    def _load_amplifier_pump(self, amp_pump: AmplifierPump) -> AmplifierPumpInfo:
        return AmplifierPumpInfo(
            pump_freq=self.opt_param(amp_pump.pump_freq),
            pump_power=self.opt_param(amp_pump.pump_power),
            cancellation=amp_pump.cancellation,
            alc_engaged=amp_pump.alc_engaged,
            use_probe=amp_pump.use_probe,
            probe_frequency=self.opt_param(amp_pump.probe_frequency),
            probe_power=self.opt_param(amp_pump.probe_power),
            channel=None,  # todo
        )

    def _load_signal(self, signal: ExperimentSignal):
        signal_info = SignalInfo(uid=signal.uid)
        mapped_ls_path: str = self._signal_mappings[signal.uid]
        mapped_ls = self._setup_helper.logical_signal_by_path(mapped_ls_path)

        signal_info.device = self._device_info.device_by_ls(mapped_ls)

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

            signal_info.signal_range = calibration.range  # todo: units
            signal_info.port_mode = calibration.port_mode
            signal_info.threshold = calibration.threshold
            signal_info.amplitude = calibration.amplitude
            if (amp_pump := calibration.amplifier_pump) is not None:
                signal_info.amplifier_pump = self._load_amplifier_pump(amp_pump)

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
        signal_info.channels = sorted((port.channel for port in physical_channel.ports))

        self._signal_infos[signal.uid] = signal_info

    def _add_parameter(
        self, value: float | Parameter | None, nt_only=False
    ) -> float | ParameterInfo | None:
        if isinstance(value, LinearSweepParameter):
            step = (value.stop - value.start) / value.count
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
        if value.uid not in self._params:
            self._params[value.uid] = param_info
        elif self._params[value.uid] != param_info:
            raise LabOneQException(
                f"Found multiple, inconsistent values for parameter {value.uid} with same UID."
            )
        if nt_only:
            self._nt_only_params.add(param_info)
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
        if value is None or isinstance(value, (float, int, complex)):
            return value
        return self._add_parameter(value, nt_only)

    def _walk_sections(
        self,
        section: Section,
        acquisition_type,
        section_uid_map: Dict[str, Tuple[Any, int]],
        acquisition_type_map,
        exchanger_map,
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
            exchanger_map(section),
            current_acquisition_type,
            exchanger_map,
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
                    exchanger_map,
                )
            )
        return section_info

    def _load_markers(self, operation):
        # todo
        return []

    def _load_ssp(
        self,
        operation: SignalOperation,
        signal_info: SignalInfo,
        _auto_pulse_id,
        acquisition_type,
    ) -> SectionSignalPulse:
        if isinstance(operation, Delay):
            pulse_offset = self.opt_param(operation.time)
            precompensation_clear = operation.precompensation_clear
            return SectionSignalPulse(
                signal=signal_info,
                pulse_def=None,
                length=pulse_offset,
                precompensation_clear=precompensation_clear,
            )

        assert isinstance(operation, (PlayPulse, Reserve, Acquire))
        pulse = None
        markers = self._load_markers(operation)

        length = getattr(operation, "length", None)
        operation_length = self.opt_param(length)

        if hasattr(operation, "pulse"):
            pulse = getattr(operation, "pulse")
        if hasattr(operation, "kernel"):
            pulse = getattr(operation, "kernel")
        if pulse is None and length is not None:
            pulse = SimpleNamespace()
            setattr(pulse, "uid", next(_auto_pulse_id))
            setattr(pulse, "length", length)
        if pulse is None and markers:
            # generate a zero amplitude pulse to play the markers
            pulse = SimpleNamespace()
            pulse.uid = next(_auto_pulse_id)
            pulse.function = "const"
            pulse.amplitude = 0.0
            pulse.length = max([m.start + m.length for m in markers])
            pulse.can_compress = False
            pulse.pulse_parameters = None
        if markers:
            for m in markers:
                if m.pulse_id is None:
                    m.pulse_id = pulse.uid

        if hasattr(operation, "handle") and pulse is None:
            raise RuntimeError(
                f"Either 'kernel' or 'length' must be provided for the acquire operation with handle '{getattr(operation, 'handle')}'."
            )

        if pulse is not None:
            pulse_def = self._add_pulse(pulse)

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

            operation_pulse_parameters = getattr(operation, "pulse_parameters", None)

            if operation_pulse_parameters is not None:
                operation_pulse_parameters = {
                    param: self.opt_param(val)
                    for param, val in operation_pulse_parameters.items()
                }

            # todo: pulse parameters

            return SectionSignalPulse(
                signal=signal_info,
                pulse=pulse_def,
                length=operation_length,
                amplitude=amplitude,
                phase=phase,
                increment_oscillator_phase=increment_oscillator_phase,
                set_oscillator_phase=set_oscillator_phase,
                precompensation_clear=False,
                play_pulse_parameters=operation_pulse_parameters,
                acquire_params=acquire_params,
                markers=markers,
            )
        if (
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

            return SectionSignalPulse(
                signal=signal_info,
                precompensation_clear=False,
                set_oscillator_phase=set_oscillator_phase,
                increment_oscillator_phase=increment_oscillator_phase,
            )

    def _load_section_operations(
        self,
        section: Section,
        section_info: SectionInfo,
        acquisition_type,
        exchanger_map: Callable[[Any], Any],
    ):
        _auto_pulse_id = (f"{section.uid}__auto_pulse_{i}" for i in itertools.count())
        section_signal_pulses = []

        for operation in exchanger_map(section).children:
            if not isinstance(operation, SignalOperation):
                continue
            signal_info = self._signal_infos[operation.signal]
            section_signal_pulses.append(
                self._load_ssp(operation, signal_info, _auto_pulse_id, acquisition_type)
            )
            section_info.pulses = section_signal_pulses

    def _load_section(
        self,
        section: Section,
        exp_acquisition_type,
        exchanger_map: Callable[[Any], Any],
        section_uid_map: Dict[str, Tuple[Any, int]],
    ) -> SectionInfo:

        if section.uid not in section_uid_map:
            section_uid_map[section.uid] = (section, 0)
            instance_id = section.uid
        else:
            visit_count = section_uid_map[section.uid][1] + 1
            instance_id = f"{section.uid}_{visit_count}"
            section_uid_map[section.uid] = (section, visit_count)

        count = 1

        if hasattr(section, "count"):
            count = section.count

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
                    and section.execution_type != ExecutionType.REAL_TIME
                    and parameter.uid in self._nt_only_params
                ):
                    raise LabOneQException(
                        f"Parameter {parameter.uid} can't be swept in real-time, it is bound to a value "
                        f"that can only be set in near-time"
                    )

        execution_type = section.execution_type
        align = exchanger_map(section).alignment
        on_system_grid = exchanger_map(section).on_system_grid
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
        assert section.trigger is not None
        triggers = [
            {"signal_id": k, "state": v["state"]} for k, v in section.trigger.items()
        ]
        chunk_count = getattr(section, "chunk_count", 1)

        this_acquisition_type = None
        if any(
            isinstance(operation, Acquire)
            for operation in exchanger_map(section).children
        ):
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


def find_sequential_averaging(section: Section | Experiment) -> Tuple[Any, Tuple]:
    avg_section, sweep_sections = None, ()

    for child_section in (
        section.sections if isinstance(section, Experiment) else section.children
    ):
        if not isinstance(child_section, Section):
            continue  # skip operations

        if getattr(child_section, "averaging_mode", None) == AveragingMode.SEQUENTIAL:
            avg_section = child_section

        parameters = getattr(child_section, "parameters", None)
        if parameters is not None and len(parameters) > 0:
            sweep_sections = (child_section,)

        child_avg_section, child_sweep_sections = find_sequential_averaging(
            child_section
        )
        if avg_section is not None and child_avg_section is not None:
            raise LabOneQException(
                "Illegal nesting of sequential averaging loops detected."
            )
        sweep_sections = (*sweep_sections, *child_sweep_sections)

    return avg_section, sweep_sections


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
