# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, TypeVar

import numpy as np

from laboneq.compiler import DeviceType
from laboneq.core.exceptions import LabOneQException
from laboneq.core.path import LogicalSignalGroups_Path, insert_logical_signal_prefix
from laboneq.core.types.enums import IODirection, PhysicalChannelType
from laboneq.core.types.units import Quantity
from laboneq.data.compilation_job import (
    AmplifierPumpInfo,
    DeviceInfo,
    ExperimentInfo,
    MixerCalibrationInfo,
    OutputRoute,
    PrecompensationInfo,
    SignalInfo,
    SignalInfoType,
    SignalRange,
)
from laboneq.data.setup_description.setup_helper import SetupHelper
from laboneq.dsl.calibration import (
    Oscillator,
    SignalCalibration,
)
from laboneq.dsl.enums import ExecutionType
from laboneq.dsl.experiment import Experiment
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.dsl.experiment.section import (
    AcquireLoopRt,
    Section,
)
from laboneq.dsl.parameter import (
    LinearSweepParameter,
    Parameter,
    SweepParameter,
)
from laboneq.implementation.payload_builder.experiment_info_builder.device_info_builder import (
    DeviceInfoBuilder,
)
from laboneq.implementation.utils.devices import device_setup_fingerprint

if TYPE_CHECKING:
    from laboneq.data.compilation_job import (
        DeviceInfo,
    )
    from laboneq.data.setup_description import (
        LogicalSignal,
        Setup,
    )
    from laboneq.dsl.calibration import (
        AmplifierPump,
        MixerCalibration,
        Precompensation,
    )
    from laboneq.dsl.experiment import Experiment
    from laboneq.dsl.experiment.experiment_signal import ExperimentSignal

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
        self._signal_infos: dict[str, SignalInfo] = {}

        self._device_info = DeviceInfoBuilder(self._device_setup)
        self._setup_helper = SetupHelper(self._device_setup)

        self._ppc_connections = self._setup_helper.ppc_connections()

        self._sweep_params_min_maxes: dict[str, tuple[float, float]] = {}
        self._calibration_items: dict[str, Any] = {}

    def load_experiment(self) -> ExperimentInfo:
        self._calibration_items = {
            sig.uid: sig.calibration
            for sig in self._experiment.signals.values()
            if sig.calibration is not None
        }
        self._check_physical_channel_calibration_conflict()
        for signal in self._experiment.signals.values():
            self._load_signal(signal)

        self._validate_realtime(self._experiment.sections)

        experiment_info = ExperimentInfo(
            device_setup_fingerprint=device_setup_fingerprint(self._device_setup),
            devices=list(self._device_info.device_mapping.values()),
            signals=sorted(self._signal_infos.values(), key=lambda s: s.uid),
            src=self._experiment,
        )
        return experiment_info

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        value = getattr(value, "value", value)
        if isinstance(value, str):
            return value.lower()
        return value

    def _get_or_create_signal_calibration(self, signal_uid: str) -> Any:
        calibration = self._calibration_items.get(signal_uid)
        if calibration is None:
            calibration = SignalCalibration()
            self._calibration_items[signal_uid] = calibration
            self._experiment.signals[signal_uid].calibration = calibration
        return calibration

    @staticmethod
    def _get_lo_frequency(calibration: Any) -> Any:
        local_oscillator = getattr(calibration, "local_oscillator", None)
        if local_oscillator is not None:
            return local_oscillator.frequency
        if hasattr(calibration, "local_oscillator_frequency"):
            return calibration.local_oscillator_frequency
        return None

    @staticmethod
    def _set_lo_frequency(calibration: Any, frequency: Any):
        if hasattr(calibration, "local_oscillator_frequency"):
            calibration.local_oscillator_frequency = frequency
            return
        local_oscillator = getattr(calibration, "local_oscillator", None)
        if local_oscillator is None:
            calibration.local_oscillator = Oscillator(frequency=frequency)
        else:
            local_oscillator.frequency = frequency

    @staticmethod
    def _get_output_routes(calibration: Any) -> list[Any]:
        # Prefer DSL output routing fields; when calibration is an AttributeOverrider,
        # data-shaped base fields may exist as empty lists and would otherwise mask
        # DSL routes from the overriding experiment calibration.
        routes = getattr(calibration, "added_outputs", None)
        if routes is not None:
            return routes
        routes = getattr(calibration, "output_routing", None)
        return routes or []

    @staticmethod
    def _route_source(route: Any) -> str | None:
        return getattr(route, "source_signal", None) or getattr(route, "source", None)

    @staticmethod
    def _route_amplitude(route: Any) -> Any:
        amplitude = getattr(route, "amplitude", None)
        if amplitude is None:
            amplitude = getattr(route, "amplitude_scaling", None)
        return amplitude

    @staticmethod
    def _route_phase(route: Any) -> Any:
        phase = getattr(route, "phase", None)
        if phase is None:
            phase = getattr(route, "phase_shift", None)
        return phase

    @staticmethod
    def _is_parameter(value: Any) -> bool:
        return isinstance(value, Parameter)

    def _check_physical_channel_calibration_conflict(self):
        field_accessors: dict[str, Any] = {
            "local_oscillator_frequency": self._get_lo_frequency,
            "port_delay": lambda cal: getattr(cal, "port_delay", None),
            "port_mode": lambda cal: getattr(cal, "port_mode", None),
            "range": lambda cal: getattr(cal, "range", None),
            "voltage_offset": lambda cal: getattr(cal, "voltage_offset", None),
            "amplitude": lambda cal: getattr(cal, "amplitude", None),
            # skip validation of structured fields
            "mixer_calibration": lambda cal: getattr(cal, "mixer_calibration", None),
            "precompensation": lambda cal: getattr(cal, "precompensation", None),
            "amplifier_pump": lambda cal: getattr(cal, "amplifier_pump", None),
        }

        exp_signals_by_pc = {}
        for signal in self._experiment.signals.values():
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
            for field_, accessor in field_accessors.items():
                unique_value = None
                conflicting = False
                for exp_signal in exp_signals:
                    exp_cal = self._calibration_items.get(exp_signal.uid)
                    if exp_cal is None:
                        continue
                    value = accessor(exp_cal)
                    if value is not None:
                        if unique_value is None:
                            unique_value = value
                        elif self._enum_value(unique_value) != self._enum_value(value):
                            conflicting = True
                            break
                if conflicting:
                    conflicting_signals = [
                        exp_signal.uid
                        for exp_signal in exp_signals
                        if (
                            other_signal_cal := self._calibration_items.get(
                                exp_signal.uid
                            )
                        )
                        is not None
                        and accessor(other_signal_cal) is not None
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
                        exp_cal = self._get_or_create_signal_calibration(exp_signal.uid)
                        if field_ == "local_oscillator_frequency":
                            self._set_lo_frequency(exp_cal, unique_value)
                        else:
                            setattr(exp_cal, field_, unique_value)

    def _get_signal_calibration(
        self, exp_signal: ExperimentSignal, logical_signal: LogicalSignal
    ) -> SignalCalibration:
        baseline_calib = self._setup_helper.calibration.by_logical_signal(
            logical_signal
        )

        exp_calib = self._calibration_items.get(exp_signal.uid)

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

    def _load_mixer_cal(self, mixer_cal: MixerCalibration) -> MixerCalibrationInfo:
        return MixerCalibrationInfo(
            voltage_offsets=tuple(mixer_cal.voltage_offsets or []),
            correction_matrix=tuple(mixer_cal.correction_matrix or []),
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
            pump_frequency=amp_pump.pump_frequency,
            pump_power=amp_pump.pump_power,
            pump_on=amp_pump.pump_on,
            pump_filter_on=amp_pump.pump_filter_on,
            cancellation_on=amp_pump.cancellation_on,
            cancellation_phase=amp_pump.cancellation_phase,
            cancellation_attenuation=amp_pump.cancellation_attenuation,
            cancellation_source=amp_pump.cancellation_source,
            cancellation_source_frequency=amp_pump.cancellation_source_frequency,
            alc_on=amp_pump.alc_on,
            probe_on=amp_pump.probe_on,
            probe_frequency=amp_pump.probe_frequency,
            probe_power=amp_pump.probe_power,
            channel=channel,
        )

    def _load_signal(self, signal: ExperimentSignal):
        mapped_ls_path: str = self._signal_mappings[signal.uid]
        mapped_ls = self._setup_helper.logical_signal_by_path(mapped_ls_path)

        device = self._device_info.device_by_ls(mapped_ls)
        device_uid = device.uid

        physical_channel = (
            self._setup_helper.instruments.physical_channel_by_logical_signal(mapped_ls)
        )
        if physical_channel.direction == IODirection.IN:
            signal_type = SignalInfoType.INTEGRATION
        elif physical_channel.type == PhysicalChannelType.RF_CHANNEL:
            signal_type = SignalInfoType.RF
        else:
            signal_type = SignalInfoType.IQ

        signal_info = SignalInfo(
            uid=signal.uid,
            device_uid=device_uid,
            ports=[port.path for port in physical_channel.ports],
            type=signal_type,
        )

        calibration = self._get_signal_calibration(signal, mapped_ls)
        if calibration is not None:
            signal_info.port_delay = calibration.port_delay
            signal_info.delay_signal = calibration.delay_signal

            if (oscillator := calibration.oscillator) is not None:
                signal_info.oscillator = oscillator

            signal_info.voltage_offset = calibration.voltage_offset

            if (mixer_cal := calibration.mixer_calibration) is not None:
                signal_info.mixer_calibration = self._load_mixer_cal(mixer_cal)
            if (precomp := calibration.precompensation) is not None:
                signal_info.precompensation = self._load_precompensation(precomp)

            signal_info.lo_frequency = self._get_lo_frequency(calibration)

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
            signal_info.amplitude = calibration.amplitude
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
            output_routes = self._get_output_routes(calibration)
            if output_routes:
                for port in physical_channel.ports:
                    if (
                        not re.match(r"SGCHANNELS/\d/OUTPUT", port.path)
                        and device.device_type != DeviceType.SHFSG
                    ):
                        msg = f"Error on signal {mapped_ls_path}: Output routing can be only applied to output SGCHANNELS."
                        raise LabOneQException(msg)
            output_routers_per_channel = defaultdict(set)
            for output_router in output_routes:
                source_signal = self._route_source(output_router)
                if source_signal is None:
                    msg = (
                        f"Error on signal {mapped_ls_path}: Output routing source signal"
                        " is not set."
                    )
                    raise LabOneQException(msg)
                if LogicalSignalGroups_Path not in source_signal:
                    source_signal = insert_logical_signal_prefix(source_signal)

                try:
                    source_signal = self._setup_helper.logical_signal_by_path(
                        source_signal
                    )
                except KeyError:
                    msg = f"Error on signal {mapped_ls_path}: Output routing source signal {self._route_source(output_router)} does not exist."
                    raise LabOneQException(msg) from None
                from_pc = (
                    self._setup_helper.instruments.physical_channel_by_logical_signal(
                        source_signal
                    )
                )
                from_device = self._device_info.device_by_ls(source_signal)

                if from_device != device:
                    msg = f"Error on signal {mapped_ls_path}: Output routing can be only applied within the same device SGCHANNELS: {device_uid} != {from_device.uid}"
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
                route_amplitude = self._route_amplitude(output_router)
                if self._is_parameter(route_amplitude):
                    if isinstance(route_amplitude, SweepParameter):
                        if route_amplitude.uid not in self._sweep_params_min_maxes:
                            self._sweep_params_min_maxes[route_amplitude.uid] = (
                                np.min(route_amplitude.values),
                                np.max(route_amplitude.values),
                            )
                        min_val, max_val = self._sweep_params_min_maxes[
                            route_amplitude.uid
                        ]
                    elif isinstance(
                        route_amplitude,
                        LinearSweepParameter,
                    ):
                        min_val = route_amplitude.start
                        max_val = route_amplitude.stop
                    if min_val < 0.0 or max_val > 1.0:
                        msg = "Output route amplitude value must be between 0 and 1."
                        raise LabOneQException(
                            f"Invalid sweep parameter {route_amplitude.uid}: {msg}"
                        )
                signal_info.output_routing.append(
                    OutputRoute(
                        to_channel=to_port.channel,
                        to_signal=signal_info.uid,
                        from_channel=from_port.channel,
                        from_signal=self._ls_to_exp_sig_mapping.get(
                            self._setup_helper.logical_signal_path(source_signal)
                        ),
                        from_port=from_port.path,
                        amplitude=route_amplitude,
                        phase=self._route_phase(output_router),
                    )
                )
            signal_info.automute = calibration.automute

        self._signal_infos[signal.uid] = signal_info

    def _validate_realtime(self, root_sections: list[Section]):
        """Verify that:
        - no near-time section is located inside a real-time section
        - if there is one, it must be the real-time boundary.
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
            for child in section.children:
                traverse_set_execution_type_and_check_rt_loop(child, in_realtime)

        for root_section in root_sections:
            traverse_set_execution_type_and_check_rt_loop(
                root_section, in_realtime=False
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
