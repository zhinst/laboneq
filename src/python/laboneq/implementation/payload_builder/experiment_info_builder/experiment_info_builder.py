# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import IODirection
from laboneq.data.calibration import SignalCalibration
from laboneq.data.compilation_job import (
    DeviceInfo,
    ExperimentInfo,
    ExperimentSignalInfo,
    InternalConnectionInfo,
    PhysicalChannelInfo,
)
from laboneq.data.setup_description import DeviceType
from laboneq.dsl.enums import ExecutionType
from laboneq.dsl.experiment import Experiment
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.dsl.experiment.section import (
    AcquireLoopRt,
    Section,
)
from laboneq.implementation.legacy_adapters.calibration_converter import (
    convert_signal_calibration,
)
from laboneq.implementation.legacy_adapters.device_setup_converter import (
    convert_device_setup_to_setup,
)
from laboneq.implementation.legacy_adapters.utils import parse_logical_signal

if TYPE_CHECKING:
    from laboneq.data.setup_description import (
        LogicalSignal,
        PhysicalChannel,
        Setup,
    )
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment, ExperimentSignal

_logger = logging.getLogger(__name__)

T = TypeVar("T")


def _convert_signal_map(experiment: Experiment) -> dict[str, LogicalSignal]:
    mapping = {}
    for signal in experiment.signals.values():
        if signal.mapped_logical_signal_path is not None:
            mapping[signal.uid] = parse_logical_signal(
                signal.mapped_logical_signal_path
            )
        else:
            raise LabOneQException(
                f"Experiment signal '{signal.uid}' has no mapping to a logical signal."
            )
    return mapping


class ExperimentInfoBuilder:
    def __init__(
        self,
        device_setup: DeviceSetup,
        experiment: Experiment,
    ):
        self._experiment = experiment
        self._signal_mappings = _convert_signal_map(experiment)

        self._device_setup = convert_device_setup_to_setup(device_setup)
        self._experiment_calibration: dict[str, SignalCalibration] = {
            sig.uid: convert_signal_calibration(sig.calibration)
            for sig in self._experiment.signals.values()
            if sig.calibration is not None
        }
        self._logical_to_physical: dict[LogicalSignal, PhysicalChannel] = {
            conn.logical_signal: conn.physical_channel
            for device in self._device_setup.instruments
            for conn in device.connections
        }

    def load_experiment(self) -> ExperimentInfo:
        physical_channels = self._create_physical_channels(self._device_setup)
        self._check_physical_channel_calibration_conflict()

        experiment_signals = [
            self._load_signal(signal) for signal in self._experiment.signals.values()
        ]
        self._sanitize_amplifier_pump_calibration(experiment_signals)
        self._validate_realtime(self._experiment.sections)

        experiment_info = ExperimentInfo(
            src=self._experiment,
            signals=experiment_signals,
            signal_map={exp: ls.path() for exp, ls in self._signal_mappings.items()},
            devices=self._build_devices(self._device_setup),
            physical_channels=physical_channels,
            setup_description=self._device_setup.setup_description,
            internal_connections=self._build_internal_connections(),
        )
        return experiment_info

    def _create_physical_channels(
        self, device_setup: Setup
    ) -> list[PhysicalChannelInfo]:
        signals = []
        for instrument in device_setup.instruments:
            if instrument.device_type == DeviceType.UNMANAGED:
                continue
            for conn in instrument.connections:
                physical_channel = conn.physical_channel
                logical_signal = conn.logical_signal
                signals.append(
                    PhysicalChannelInfo(
                        uid=logical_signal.path(),
                        device_uid=instrument.uid,
                        ports=[p.path for p in physical_channel.ports],
                        channel_direction=physical_channel.direction,
                        channel_type=physical_channel.type,
                    )
                )
        return signals

    def _build_devices(self, device_setup: Setup) -> list[DeviceInfo]:
        devices = []
        for instrument in device_setup.instruments:
            if instrument.device_type == DeviceType.UNMANAGED:
                continue
            device = DeviceInfo(
                uid=instrument.uid,
                device_type=instrument.device_type,
                options=instrument.device_options or "",
                reference_clock_source=instrument.reference_clock.source,
            )
            devices.append(device)
        return devices

    def _build_internal_connections(self) -> list[InternalConnectionInfo]:
        internal_connections = []
        for conn in self._device_setup.setup_internal_connections:
            if conn.from_instrument.device_type != DeviceType.SHFPPC:
                continue
            from_instrument = conn.from_instrument.uid
            from_port = conn.from_port.path
            to_instrument = conn.to_instrument.uid
            to_port = conn.to_port.path
            internal_connections.append(
                InternalConnectionInfo(
                    from_instrument=from_instrument,
                    from_port=from_port,
                    to_instrument=to_instrument,
                    to_port=to_port,
                )
            )
        return internal_connections

    def _get_or_create_signal_calibration(self, signal_uid: str) -> SignalCalibration:
        calibration = self._experiment_calibration.get(signal_uid)
        if calibration is None:
            calibration = SignalCalibration()
            self._experiment_calibration[signal_uid] = calibration
        return calibration

    def _check_physical_channel_calibration_conflict(self):
        field_accessors: dict[str, Any] = {
            "local_oscillator_frequency": lambda cal: getattr(
                cal, "local_oscillator_frequency", None
            ),
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

        exp_signals_by_pc: dict[tuple[str, str], list[ExperimentSignal]] = {}
        for signal in self._experiment.signals.values():
            mapped_ls = self._signal_mappings[signal.uid]
            physical_channel = self._logical_to_physical[mapped_ls]
            exp_signals_by_pc.setdefault(physical_channel, []).append(signal)

        # Merge the calibration of those ExperimentSignals that touch the same
        # PhysicalChannel.
        for physical_channel, exp_signals in exp_signals_by_pc.items():
            for field_, accessor in field_accessors.items():
                unique_value = None
                conflicting = False
                for exp_signal in exp_signals:
                    exp_cal = self._experiment_calibration.get(exp_signal.uid)
                    if exp_cal is None:
                        continue
                    value = accessor(exp_cal)
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
                            other_signal_cal := self._experiment_calibration.get(
                                exp_signal.uid
                            )
                        )
                        is not None
                        and accessor(other_signal_cal) is not None
                    ]
                    pc_uid = f"{physical_channel.group}/{physical_channel.name}"
                    raise LabOneQException(
                        f"The experiment signals {', '.join(conflicting_signals)} all "
                        f"touch physical channel '{pc_uid}', but provide conflicting "
                        f"settings for calibration field '{field_}'."
                    )
                if unique_value is not None:
                    # Make sure all the experiment signals agree.
                    for exp_signal in exp_signals:
                        exp_cal = self._get_or_create_signal_calibration(exp_signal.uid)
                        setattr(exp_cal, field_, unique_value)

    def _get_signal_calibration(
        self, exp_signal: ExperimentSignal, logical_signal: LogicalSignal
    ) -> SignalCalibration:
        baseline_calib = self._device_setup.calibration.get(logical_signal.path())
        exp_calib = self._experiment_calibration.get(exp_signal.uid)

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
        return calibration or SignalCalibration()

    def _load_signal(self, signal: ExperimentSignal) -> ExperimentSignalInfo:
        mapped_ls = self._signal_mappings[signal.uid]
        calibration = self._get_signal_calibration(signal, mapped_ls)
        return ExperimentSignalInfo(uid=signal.uid, calibration=calibration)

    def _sanitize_amplifier_pump_calibration(
        self, experiment_signals: list[ExperimentSignalInfo]
    ):
        # NOTE: Compiler will raise an error if amplifier pump is set for a non-SHFQA acquire line.
        # Python DSL allows this, but we log a warning here to avoid silent misconfigurations.
        # TODO: Remove and let compiler fail?
        logical_signals_with_ppc_connection: set[LogicalSignal] = set()
        for conn in self._device_setup.setup_internal_connections:
            if conn.from_instrument.device_type != DeviceType.SHFPPC:
                continue
            for ls_to_pc in conn.to_instrument.connections:
                if conn.to_port in ls_to_pc.physical_channel.ports:
                    logical_signals_with_ppc_connection.add(ls_to_pc.logical_signal)

        for signal in experiment_signals:
            logical_signal = self._signal_mappings[signal.uid]
            physical_channel = self._logical_to_physical[logical_signal]
            if signal.calibration.amplifier_pump is not None:
                if physical_channel.direction != IODirection.IN:
                    _logger.warning(
                        "'amplifier_pump' calibration for logical signal %s will be ignored - "
                        "only applicable to SHFQA acquire lines",
                        logical_signal.path(),
                    )
                    signal.calibration.amplifier_pump = None
                elif logical_signal not in logical_signals_with_ppc_connection:
                    _logger.warning(
                        "'amplifier_pump' calibration for logical signal %s will be ignored - "
                        "no PPC is connected to it",
                        logical_signal.path(),
                    )
                    signal.calibration.amplifier_pump = None

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
