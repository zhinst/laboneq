# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import itertools
import logging
import typing
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

from laboneq._utils import ensure_list, id_generator
from laboneq.compiler.experiment_access.loader_base import LoaderBase
from laboneq.compiler.experiment_access.param_ref import ParamRef
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import (
    AcquisitionType,
    AveragingMode,
    ExecutionType,
    IODirection,
    IOSignalType,
    SectionAlignment,
)
from laboneq.data.compilation_job import (
    AcquireInfo,
    AmplifierPumpInfo,
    FollowerInfo,
    Marker,
    MixerCalibrationInfo,
    PrecompensationInfo,
    PulseDef,
    SectionInfo,
    SectionSignalPulse,
    SignalRange,
)

if typing.TYPE_CHECKING:
    from laboneq.dsl.device import DeviceSetup
    from laboneq.dsl.device.io_units import LogicalSignal
    from laboneq.dsl.device.ports import Port
    from laboneq.dsl.experiment import Experiment, ExperimentSignal
    from laboneq.dsl.parameter import Parameter

_logger = logging.getLogger(__name__)


def find_value_or_parameter_attr(
    entity: Any, attr: str, value_types: tuple[type, ...]
) -> tuple[Any, str]:
    param = None
    value = getattr(entity, attr, None)
    if value is not None and not isinstance(value, value_types):
        param = getattr(value, "uid", None)
        value = None

    return value, param


class DSLLoader(LoaderBase):
    def __init__(self):
        super().__init__()
        self._nt_only_params = []
        self._section_operations_to_add = []

    def load(self, experiment: Experiment, device_setup: DeviceSetup):
        self.global_leader_device_id = None

        for server in device_setup.servers.values():
            if hasattr(server, "leader_uid"):
                self.global_leader_device_id = server.leader_uid

        dest_path_devices = {}

        # signal -> (device_uid, channel)
        ppc_connections: dict[str, tuple[str, int]] = {}

        reference_clock = None
        for device in device_setup.instruments:
            if hasattr(device, "reference_clock"):
                reference_clock = device.reference_clock

        for device in sorted(device_setup.instruments, key=lambda x: x.uid):
            driver = type(device).__name__.lower()
            reference_clock_source = getattr(device, "reference_clock_source", None)
            is_qc = getattr(device, "is_qc", None)

            self.add_device(
                device.uid,
                driver,
                reference_clock=reference_clock,
                reference_clock_source=None
                if reference_clock_source is None
                else reference_clock_source.value,
                is_qc=is_qc,
            )

            for connection in device.connections:
                if connection.signal_type == IOSignalType.PPC:
                    port = next(
                        p for p in device.ports if p.uid == connection.local_port
                    )
                    ppc_connections[connection.remote_path] = (
                        device.uid,
                        int(port.physical_port_ids[0]),
                    )
                    continue

                multiplex_key = (
                    device.uid,
                    connection.local_port,
                    connection.direction.value,
                )

                if connection.remote_path in dest_path_devices:
                    dpd = dest_path_devices[connection.remote_path]
                    dpd["local_paths"].append(connection.local_path)
                    dpd["local_ports"].append(connection.local_port)
                    dpd["remote_ports"].append(connection.remote_port)
                    dpd["types"].append(
                        connection.signal_type.value
                        if connection.signal_type is not None
                        else None
                    )
                    dpd["multiplex_keys"].append(multiplex_key)
                else:
                    dest_path_devices[connection.remote_path] = {
                        "device": device.uid,
                        "root_path": "",
                        "local_paths": [connection.local_path],
                        "local_ports": [connection.local_port],
                        "remote_ports": [connection.remote_port],
                        "types": [
                            connection.signal_type.value
                            if connection.signal_type is not None
                            else None
                        ],
                        "multiplex_keys": [multiplex_key],
                    }

        ls_map = {}
        modulated_paths = {}
        ls_voltage_offsets = {}
        ls_mixer_calibrations = {}
        ls_precompensations = {}
        ls_lo_frequencies = {}
        ls_ranges = {}
        ls_range_units = {}
        ls_port_delays = {}
        ls_delays_signal = {}
        ls_port_modes = {}
        ls_thresholds = {}
        ls_amplitudes = {}
        ls_amplifier_pumps = {}
        self._nt_only_params = []

        all_logical_signals = [
            ls
            for lsg in device_setup.logical_signal_groups.values()
            for ls in lsg.logical_signals.values()
        ]
        for ls in all_logical_signals:
            ls_map[ls.path] = ls

        mapped_logical_signals: Dict["LogicalSignal", "ExperimentSignal"] = {}
        experiment_signals_by_physical_channel = {}
        for signal in experiment.signals.values():
            # Need to create copy here as we'll possibly patch those ExperimentSignals
            # that touch the same PhysicalChannel
            try:
                mapped_ls = ls_map[signal.mapped_logical_signal_path]
            except KeyError:
                raise LabOneQException(
                    f"Experiment signal '{signal.uid}' has no mapping to a logical signal."
                )
            sig_copy = copy.deepcopy(signal)
            mapped_logical_signals[mapped_ls] = sig_copy
            experiment_signals_by_physical_channel.setdefault(
                mapped_ls.physical_channel, []
            ).append(sig_copy)

        from laboneq.dsl.device.io_units.physical_channel import (
            PHYSICAL_CHANNEL_CALIBRATION_FIELDS,
        )

        # Merge the calibration of those ExperimentSignals that touch the same
        # PhysicalChannel.
        for pc, exp_signals in experiment_signals_by_physical_channel.items():
            for field_ in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
                if field_ in ["mixer_calibration", "precompensation"]:
                    continue
                unique_value = None
                conflicting = False
                for exp_signal in exp_signals:
                    if not exp_signal.is_calibrated():
                        continue
                    value = getattr(exp_signal, field_)
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
                    raise LabOneQException(
                        f"The experiment signals {', '.join(conflicting_signals)} all "
                        f"touch physical channel '{pc.uid}', but provide conflicting "
                        f"settings for calibration field '{field_}'."
                    )
                if unique_value is not None:
                    # Make sure all the experiment signals agree.
                    for exp_signal in exp_signals:
                        if exp_signal.is_calibrated():
                            setattr(exp_signal.calibration, field_, unique_value)

        for ls in all_logical_signals:
            calibration = ls.calibration
            experiment_signal_for_ls = mapped_logical_signals.get(ls)
            if experiment_signal_for_ls is not None:
                experiment_signal_calibration = experiment_signal_for_ls.calibration
                if experiment_signal_calibration is not None:
                    _logger.debug(
                        "Found overriding signal calibration for %s %s",
                        ls.path,
                        experiment_signal_calibration,
                    )
                    calibration = AttributeOverrider(
                        calibration, experiment_signal_calibration
                    )

            if calibration is not None:

                def opt_param(
                    val: float | Parameter | None,
                ) -> float | int | str | None:
                    if val is None or isinstance(val, (float, int)):
                        return val
                    self._nt_only_params.append(val.uid)
                    return val.uid

                if hasattr(calibration, "port_delay"):
                    ls_port_delays[ls.path] = opt_param(calibration.port_delay)

                if hasattr(calibration, "delay_signal"):
                    ls_delays_signal[ls.path] = calibration.delay_signal

                if hasattr(calibration, "oscillator"):
                    if calibration.oscillator is not None:
                        oscillator = calibration.oscillator
                        is_hardware = oscillator.modulation_type.value == "HARDWARE"

                        oscillator_uid = oscillator.uid

                        frequency = oscillator.frequency
                        if hasattr(frequency, "uid"):
                            frequency = self._get_or_create_parameter(frequency.uid)

                        modulated_paths[ls.path] = {
                            "oscillator_id": oscillator_uid,
                            "is_hardware": is_hardware,
                        }
                        known_oscillator = self._oscillators.get(oscillator_uid)
                        if known_oscillator is None:
                            self.add_oscillator(oscillator_uid, frequency, is_hardware)

                            if is_hardware:
                                device_id = dest_path_devices[ls.path]["device"]
                                self.add_device_oscillator(device_id, oscillator_uid)
                        else:
                            if (
                                known_oscillator.frequency,
                                known_oscillator.is_hardware,
                            ) != (frequency, is_hardware):
                                raise Exception(
                                    f"Duplicate oscillator uid {oscillator_uid} found in {ls.path}"
                                )
                try:
                    ls_voltage_offsets[ls.path] = calibration.voltage_offset
                except AttributeError:
                    pass

                mixer_cal = getattr(calibration, "mixer_calibration", None)
                if mixer_cal is not None:
                    ls_mixer_calibrations[ls.path] = MixerCalibrationInfo(
                        voltage_offsets=calibration.mixer_calibration.voltage_offsets,
                        correction_matrix=calibration.mixer_calibration.correction_matrix,
                    )

                precomp = getattr(calibration, "precompensation", None)
                if precomp is not None:
                    ls_precompensations[ls.path] = PrecompensationInfo(
                        precomp.exponential,
                        precomp.high_pass,
                        precomp.bounce,
                        precomp.FIR,
                    )

                local_oscillator = calibration.local_oscillator
                if local_oscillator is not None:
                    ls_lo_frequencies[ls.path] = opt_param(local_oscillator.frequency)

                signal_range = getattr(calibration, "range")
                if signal_range is not None:
                    if hasattr(signal_range, "unit"):
                        ls_ranges[ls.path] = signal_range.value
                        ls_range_units[ls.path] = str(signal_range.unit)
                    else:
                        ls_ranges[ls.path] = signal_range
                        ls_range_units[ls.path] = None

                if (
                    hasattr(calibration, "port_mode")
                    and calibration.port_mode is not None
                ):
                    ls_port_modes[ls.path] = calibration.port_mode.value

                ls_thresholds[ls.path] = getattr(calibration, "threshold", None)

                ls_amplitudes[ls.path] = opt_param(calibration.amplitude)

                amp_pump = calibration.amplifier_pump
                if amp_pump is not None:
                    if ls.direction != IODirection.IN:
                        _logger.warning(
                            "'amplifier_pump' calibration for logical signal %s will be ignored - "
                            "only applicable to acquire lines",
                            ls.path,
                        )
                    elif ls.path not in ppc_connections:
                        _logger.warning(
                            "'amplifier_pump' calibration for logical signal %s will be ignored - "
                            "no PPC is connected to it",
                            ls.path,
                        )
                    else:
                        ls_amplifier_pumps[ls.path] = AmplifierPumpInfo(
                            channel=ppc_connections[ls.path][1],
                            device=self._devices[ppc_connections[ls.path][0]],
                            pump_freq=opt_param(amp_pump.pump_freq),
                            pump_power=opt_param(amp_pump.pump_power),
                            cancellation=amp_pump.cancellation,
                            alc_engaged=amp_pump.alc_engaged,
                            use_probe=amp_pump.use_probe,
                            probe_frequency=opt_param(amp_pump.probe_frequency),
                            probe_power=opt_param(amp_pump.probe_power),
                        )

        for signal in sorted(experiment.signals.values(), key=lambda x: x.uid):
            dev_sig_types = []
            if signal.mapped_logical_signal_path is not None:
                dev_sig_types = dest_path_devices[signal.mapped_logical_signal_path][
                    "types"
                ]
            signal_type = (
                "single"
                if (len(dev_sig_types) == 1 and dev_sig_types[0] != "IQ")
                else "iq"
            )
            ls = ls_map.get(signal.mapped_logical_signal_path)
            if ls is None:
                raise RuntimeError(f"No logical signal found for {signal}")
            if ls is not None and ls.direction == IODirection.IN:
                signal_type = "integration"
                _logger.debug("exp signal %s ls %s IS AN INPUT", signal, ls)
            else:
                _logger.debug("exp signal %s ls %s IS AN OUTPUT", signal, ls)

            self.add_signal(signal.uid, signal_type)

            if signal.mapped_logical_signal_path in modulated_paths:
                oscillator_id = modulated_paths[signal.mapped_logical_signal_path][
                    "oscillator_id"
                ]
                self.add_signal_oscillator(signal.uid, oscillator_id)

        for signal, lsuid in experiment.get_signal_map().items():
            local_paths = dest_path_devices[lsuid].get("local_paths")

            remote_ports = dest_path_devices[lsuid].get("remote_ports")

            channels = []
            if local_paths:
                device = dest_path_devices[lsuid]["device"]

                local_ports = dest_path_devices[lsuid].get("local_ports")

                for i, local_port in enumerate(local_ports):
                    instrument = device_setup.instrument_by_uid(device)
                    current_port = instrument.output_by_uid(
                        local_port
                    ) or instrument.input_by_uid(local_port)
                    if current_port is None:
                        raise RuntimeError(
                            f"local port {local_port} not found in {instrument}"
                        )
                    if current_port.direction == IODirection.IN:
                        if len(current_port.physical_port_ids) < 2:
                            for physical_port_id in current_port.physical_port_ids:
                                channels.append(int(physical_port_id))
                        else:
                            channels.append(int(remote_ports[i]))
                        dest_path_devices[lsuid]["type"] = "in"

                    else:
                        dest_path_devices[lsuid]["type"] = "out"
                        for physical_port_id in current_port.physical_port_ids:
                            channels.append(int(physical_port_id))

            else:
                local_ports = "N/A"
            if len(channels) > 1:
                if len(set(channels)) < len(channels):
                    raise RuntimeError(
                        f"Channels for a signal must be distinct, but got {channels}"
                        f" for signal {signal}, connection ports: {local_ports}"
                    )

            self.add_signal_connection(
                signal,
                {
                    "signal_id": signal,
                    "device_id": dest_path_devices[lsuid]["device"],
                    "connection_type": dest_path_devices[lsuid]["type"],
                    "channels": channels,
                    "voltage_offset": ls_voltage_offsets.get(lsuid),
                    "mixer_calibration": ls_mixer_calibrations.get(lsuid),
                    "precompensation": ls_precompensations.get(lsuid),
                    "lo_frequency": ls_lo_frequencies.get(lsuid),
                    "range": SignalRange(
                        ls_ranges.get(lsuid), ls_range_units.get(lsuid)
                    )
                    if ls_ranges.get(lsuid) is not None
                    else None,
                    "port_delay": ls_port_delays.get(lsuid),
                    "delay_signal": ls_delays_signal.get(lsuid),
                    "port_mode": ls_port_modes.get(lsuid),
                    "threshold": ls_thresholds.get(lsuid),
                    "amplitude": ls_amplitudes.get(lsuid),
                    "amplifier_pump": ls_amplifier_pumps.get(lsuid),
                },
            )

        available_inputs = {
            (instrument.uid, input_obj.signal_type)
            for instrument in device_setup.instruments
            for input_obj in instrument.ports
            if input_obj.direction == IODirection.IN
        }

        @dataclass
        class _SyncingConnection:
            leader_device_uid: str
            follower_device_uid: str
            signal_type: IOSignalType
            output: Port | None

        syncing_connections: list[_SyncingConnection] = [
            _SyncingConnection(
                leader_device_uid=instrument.uid,
                follower_device_uid=connection.remote_path,
                signal_type=connection.signal_type,
                output=instrument.output_by_uid(connection.local_port),
            )
            for instrument in device_setup.instruments
            for connection in instrument.connections
            if (connection.remote_path, connection.signal_type) in available_inputs
        ]

        for sc in syncing_connections:
            if sc.signal_type == IOSignalType.DIO:
                leader = self._devices[sc.leader_device_uid]
                follower = self._devices[sc.follower_device_uid]
                leader.followers.append(FollowerInfo(follower, 0))
            elif sc.signal_type == IOSignalType.ZSYNC:
                port = int(sc.output.physical_port_ids[0])
                self._devices[sc.leader_device_uid].followers.append(
                    FollowerInfo(self._devices[sc.follower_device_uid], port)
                )

        seq_avg_section, sweep_sections = find_sequential_averaging(experiment)
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

        self._root_sections = [exchanger_map(s).uid for s in experiment.sections]

        section_uid_map = {}
        acquisition_type_map = {}
        for section in experiment.sections:
            self._process_section(
                section, None, section_uid_map, acquisition_type_map, exchanger_map
            )

        # Need to defer the insertion of section operations. In sequential averaging mode,
        # the tree-walking order might otherwise make us visit operations which depend on parameters
        # we haven't seen the sweep of yet.
        for section, acquisition_type, instance_id in self._section_operations_to_add:
            self._insert_section_operations(
                section, acquisition_type, exchanger_map, instance_id
            )

        if seq_avg_section is not None and len(sweep_sections):
            avg_info = self._sections[seq_avg_section.uid]
            sweep_info = self._sections[sweep_sections[0].uid]
            avg_info.children, sweep_info.children = (
                sweep_info.children,
                avg_info.children,
            )

            for parent in self._sections.values():
                for i, c in enumerate(parent.children):
                    if c is avg_info:
                        parent.children[i] = sweep_info
                    elif c is sweep_info:
                        parent.children[i] = avg_info

        self.acquisition_type = AcquisitionType(
            next(
                (at for at in acquisition_type_map.values() if at is not None),
                AcquisitionType.INTEGRATION,
            )
        )

    def _process_section(
        self,
        section,
        acquisition_type,
        section_uid_map: Dict[str, Tuple[Any, int]],
        acquisition_type_map,
        exchanger_map,
        parent_instance_id=None,
    ):
        if section.uid is None:
            raise RuntimeError(f"Section uid must not be None: {section}")
        if (
            section.uid in section_uid_map
            and section is not section_uid_map[section.uid][0]
        ):
            raise LabOneQException(
                f"Duplicate section uid '{section.uid}' found in experiment"
            )
        current_acquisition_type = acquisition_type

        if hasattr(section, "acquisition_type"):
            current_acquisition_type = section.acquisition_type

        acquisition_type_map[section.uid] = current_acquisition_type

        if section.uid not in section_uid_map:
            section_uid_map[section.uid] = (section, 0)
            instance_id = section.uid
        else:
            visit_count = section_uid_map[section.uid][1] + 1
            instance_id = f"{section.uid}_{visit_count}"
            section_uid_map[section.uid] = (section, visit_count)
        self._insert_section(
            section, current_acquisition_type, exchanger_map, instance_id
        )

        if parent_instance_id is not None:
            self.add_section_child(parent_instance_id, instance_id)

        for index, child_section in enumerate(section.sections):
            self._process_section(
                child_section,
                current_acquisition_type,
                section_uid_map,
                acquisition_type_map,
                exchanger_map,
                parent_instance_id=instance_id,
            )

    def _extract_markers(self, operation):
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

    def _sweep_derived_param(self, param: Parameter):
        base_swept_params = {
            p.uid: s for s, pp in self._section_parameters.items() for p in pp
        }
        if param.uid in base_swept_params:
            return

        # This parameter is not swept directly, but derived from a swept parameter;
        # we must add it to the corresponding loop.
        parent = param.driven_by[0]
        self._sweep_derived_param(parent)
        # the parent should now be added correctly, so try the initial test again
        base_swept_params = {
            p.uid: s for s, pp in self._section_parameters.items() for p in pp
        }
        assert parent.uid in base_swept_params
        values_list = None
        if param.values is not None:
            values_list = list(param.values)
        axis_name = param.axis_name
        self.add_section_parameter(
            base_swept_params[parent.uid],
            param.uid,
            values_list=values_list,
            axis_name=axis_name,
        )

    def _insert_section(
        self,
        section,
        exp_acquisition_type,
        exchanger_map: Callable[[Any], Any],
        instance_id: str,
    ):
        count = None

        if hasattr(section, "count"):
            count = int(section.count)

        if hasattr(section, "parameters"):
            for parameter in section.parameters:
                values_list = None
                if parameter.values is not None:
                    values_list = list(parameter.values)
                axis_name = getattr(parameter, "axis_name", None)
                self.add_section_parameter(
                    instance_id,
                    parameter.uid,
                    values_list=values_list,
                    axis_name=axis_name,
                )
                if hasattr(parameter, "count"):
                    count = parameter.count
                elif hasattr(parameter, "values"):
                    count = len(parameter.values)
                if count < 1:
                    raise Exception(
                        f"Repeat count must be at least 1, but section {section.uid}"
                        f" has count={count}"
                    )
                if (
                    section.execution_type is not None
                    and section.execution_type.value == "hardware"
                    and parameter.uid in self._nt_only_params
                ):
                    raise Exception(
                        f"Parameter {parameter.uid} can't be swept in real-time, it is"
                        f" bound to a value that can only be set in near-time"
                    )

        execution_type = (
            ExecutionType(section.execution_type) if section.execution_type else None
        )
        align = SectionAlignment(exchanger_map(section).alignment)
        on_system_grid = exchanger_map(section).on_system_grid
        length = section.length

        averaging_mode = getattr(section, "averaging_mode", None)
        repetition_mode = getattr(section, "repetition_mode", None)
        repetition_time = getattr(section, "repetition_time", None)
        reset_oscillator_phase = getattr(section, "reset_oscillator_phase", False)
        handle = getattr(section, "handle", None)
        user_register = getattr(section, "user_register", None)
        state = getattr(section, "state", None)
        local = getattr(section, "local", None)

        trigger = [
            {"signal_id": k, "state": v["state"]} for k, v in section.trigger.items()
        ]

        chunk_count = getattr(section, "chunk_count", 1)

        acquisition_type = None
        for operation in exchanger_map(section).operations:
            if hasattr(operation, "handle"):
                # an acquire event - add acquisition_types
                acquisition_type = exp_acquisition_type

        play_after = getattr(section, "play_after", None)
        if play_after:
            section_uid = lambda x: x.uid if hasattr(x, "uid") else x
            play_after = section_uid(play_after)
            if isinstance(play_after, list):
                play_after = [section_uid(s) for s in play_after]

        self.add_section(
            instance_id,
            SectionInfo(
                uid=instance_id,
                execution_type=execution_type,
                count=count,
                chunk_count=chunk_count,
                acquisition_type=acquisition_type,
                alignment=align,
                on_system_grid=on_system_grid,
                length=length,
                averaging_mode=averaging_mode,
                repetition_mode=repetition_mode,
                repetition_time=repetition_time,
                play_after=play_after,
                reset_oscillator_phase=reset_oscillator_phase,
                triggers=trigger,
                handle=handle,
                user_register=user_register,
                state=state,
                local=local,
            ),
        )

        for t in trigger:
            if "signal_id" in t:
                signal_id = t["signal_id"]
                v = self._signal_trigger.get(signal_id, 0)
                self._signal_trigger[signal_id] = v | t["state"]

        for operation in exchanger_map(section).operations:
            if not hasattr(operation, "signal"):
                continue
            self.add_section_signal(instance_id, operation.signal)

        self._section_operations_to_add.append(
            (section, exp_acquisition_type, instance_id)
        )

    def _insert_section_operations(
        self,
        section,
        acquisition_type,
        exchanger_map: Callable[[Any], Any],
        instance_id: str,
    ):
        _auto_pulse_id = (f"{section.uid}__auto_pulse_{i}" for i in itertools.count())
        for operation in exchanger_map(section).operations:
            if hasattr(operation, "signal"):
                pulse_offset = None

                if hasattr(operation, "time"):  # Delay operation
                    pulse_offset = operation.time
                    precompensation_clear = (
                        getattr(operation, "precompensation_clear", None) or False
                    )
                    if not isinstance(operation.time, float) and not isinstance(
                        operation.time, int
                    ):
                        pulse_offset = self._get_or_create_parameter(operation.time.uid)
                        self._sweep_derived_param(operation.time)

                    ssp = SectionSignalPulse(
                        signal=self._signals[operation.signal],
                        offset=pulse_offset,
                        precompensation_clear=precompensation_clear,
                    )
                    self.add_section_signal_pulse(instance_id, operation.signal, ssp)
                else:  # All operations, except Delay
                    pulses = None
                    markers = self._extract_markers(operation)

                    length = getattr(operation, "length", None)
                    operation_length = length
                    if operation_length is not None and not isinstance(
                        operation_length, (float, complex, int)
                    ):
                        self._sweep_derived_param(operation_length)
                        operation_length = self._get_or_create_parameter(
                            operation_length.uid
                        )

                    if hasattr(operation, "pulse"):
                        pulses = getattr(operation, "pulse")
                        if isinstance(pulses, list) and len(pulses) > 1:
                            raise RuntimeError(
                                f"Only one pulse can be provided for pulse play command in section {instance_id}."
                            )
                    if hasattr(operation, "kernel"):
                        # We allow multiple kernels for multistate
                        pulses = getattr(operation, "kernel")
                        if not hasattr(operation, "handle"):
                            raise RuntimeError(
                                f"Kernels {pulses} in section {instance_id} do not have a handle."
                            )
                    if pulses is None and length is not None:
                        pulses = SimpleNamespace()
                        setattr(pulses, "uid", next(_auto_pulse_id))
                        setattr(pulses, "length", length)
                    if pulses is None and markers:
                        # generate an zero amplitude pulse to play the markers
                        pulses = SimpleNamespace()
                        pulses.uid = next(_auto_pulse_id)
                        pulses.function = "const"
                        pulses.amplitude = 0.0
                        pulses.length = max([m.start + m.length for m in markers])
                        pulses.can_compress = False
                        pulses.pulse_parameters = None
                    pulses = ensure_list(pulses)
                    pulse_group = (
                        None if len(pulses) == 1 else id_generator("pulse_group")
                    )
                    if markers:
                        for m in markers:
                            if m.pulse_id is None:
                                assert len(pulses) == 1 and pulses[0] is not None
                                m.pulse_id = pulses[0].uid

                    if hasattr(operation, "handle") and pulses is None:
                        raise RuntimeError(
                            f"Either 'kernel' or 'length' must be provided for the"
                            f" acquire operation with handle '{getattr(operation, 'handle')}'."
                        )

                    signals = ensure_list(operation.signal)
                    if len(signals) != len(pulses):
                        raise RuntimeError(
                            f"Number of pulses ({len(pulses)}) must be equal to number of signals ({len(signals)}) for section {instance_id}."
                        )

                    # Play/acquire/measure command parameters
                    (
                        pulse_amplitude,
                        pulse_amplitude_param,
                    ) = find_value_or_parameter_attr(
                        operation, "amplitude", (int, float, complex)
                    )
                    if pulse_amplitude_param is not None:
                        self._sweep_derived_param(operation.amplitude)
                        pulse_amplitude = self._get_or_create_parameter(
                            pulse_amplitude_param
                        )

                    pulse_phase, pulse_phase_param = find_value_or_parameter_attr(
                        operation, "phase", (int, float)
                    )
                    if pulse_phase_param is not None:
                        self._sweep_derived_param(operation.phase)
                        pulse_phase = self._get_or_create_parameter(pulse_phase_param)

                    (
                        pulse_increment_oscillator_phase,
                        pulse_increment_oscillator_phase_param,
                    ) = find_value_or_parameter_attr(
                        operation, "increment_oscillator_phase", (int, float)
                    )
                    if pulse_increment_oscillator_phase_param is not None:
                        self._sweep_derived_param(operation.increment_oscillator_phase)
                        pulse_increment_oscillator_phase = (
                            self._get_or_create_parameter(
                                pulse_increment_oscillator_phase_param
                            )
                        )
                    (
                        pulse_set_oscillator_phase,
                        pulse_set_oscillator_phase_param,
                    ) = find_value_or_parameter_attr(
                        operation, "set_oscillator_phase", (int, float)
                    )
                    if pulse_set_oscillator_phase_param is not None:
                        self._sweep_derived_param(operation.set_oscillator_phase)
                        pulse_set_oscillator_phase = self._get_or_create_parameter(
                            pulse_set_oscillator_phase_param
                        )

                    acquire_params = None
                    if hasattr(operation, "handle"):
                        acquire_params = AcquireInfo(
                            handle=operation.handle,
                            acquisition_type=acquisition_type.value,
                        )

                    operation_pulse_parameters = copy.deepcopy(
                        getattr(operation, "pulse_parameters", None)
                    )

                    if operation_pulse_parameters is not None:
                        operation_pulse_parameters_list = ensure_list(
                            operation_pulse_parameters
                        )
                        for p in operation_pulse_parameters_list:
                            for param, val in p.items():
                                if hasattr(val, "uid"):
                                    # Take the presence of "uid" as a proxy for isinstance(val, SweepParameter)
                                    p[param] = ParamRef(val.uid)
                                    self._sweep_derived_param(val)
                    else:
                        operation_pulse_parameters_list = [None] * len(pulses)

                    if markers:
                        for m in markers:
                            self.add_signal_marker(operation.signal, m.marker_selector)

                    for pulse, signal, op_pars in zip(
                        pulses, signals, operation_pulse_parameters_list
                    ):
                        if pulse is not None:
                            function = None
                            length = None

                            pulse_parameters = getattr(pulse, "pulse_parameters", None)

                            if pulse.uid not in self._pulses:
                                samples = None
                                if hasattr(pulse, "function"):
                                    function = pulse.function
                                if hasattr(pulse, "length"):
                                    length = pulse.length
                                if hasattr(pulse, "samples"):
                                    samples = pulse.samples

                                (
                                    amplitude,
                                    amplitude_param,
                                ) = find_value_or_parameter_attr(
                                    pulse, "amplitude", (float, int, complex)
                                )
                                if amplitude_param is not None:
                                    raise LabOneQException(
                                        f"Amplitude of pulse '{pulse.uid}' cannot be a parameter."
                                        f" To sweep the amplitude, pass the parameter in the"
                                        f" corresponding `play()` command."
                                    )

                                can_compress = False
                                if hasattr(pulse, "can_compress"):
                                    can_compress = pulse.can_compress

                                self.add_pulse(
                                    pulse.uid,
                                    PulseDef(
                                        uid=pulse.uid,
                                        function=function,
                                        length=length,
                                        amplitude=amplitude,
                                        can_compress=can_compress,
                                        samples=samples,
                                    ),
                                )
                            # Replace sweep params with a ParamRef
                            if pulse_parameters is not None:
                                for param, val in pulse_parameters.items():
                                    if hasattr(val, "uid"):
                                        # Take the presence of "uid" as a proxy for isinstance(val, SweepParameter)
                                        pulse_parameters[param] = ParamRef(val.uid)
                                        self._sweep_derived_param(val)

                            ssp = SectionSignalPulse(
                                signal=self._signals[signal],
                                pulse=self._pulses[pulse.uid],
                                offset=pulse_offset,
                                amplitude=pulse_amplitude,
                                length=operation_length,
                                acquire_params=acquire_params,
                                phase=pulse_phase,
                                increment_oscillator_phase=pulse_increment_oscillator_phase,
                                set_oscillator_phase=pulse_set_oscillator_phase,
                                play_pulse_parameters=op_pars,
                                pulse_pulse_parameters=pulse_parameters,
                                precompensation_clear=False,  # only for delay
                                markers=markers,
                                pulse_group=pulse_group,
                            )
                            self.add_section_signal_pulse(instance_id, signal, ssp)
                        elif (
                            getattr(operation, "increment_oscillator_phase", None)
                            is not None
                            or getattr(operation, "set_oscillator_phase", None)
                            is not None
                            or getattr(operation, "phase", None) is not None
                        ):
                            # virtual Z gate
                            if getattr(operation, "phase", None) is not None:
                                raise LabOneQException(
                                    "Phase argument has no effect for virtual Z gates."
                                )
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

                            ssp = SectionSignalPulse(
                                signal=self._signals[operation.signal],
                                increment_oscillator_phase=pulse_increment_oscillator_phase,
                                set_oscillator_phase=pulse_set_oscillator_phase,
                                precompensation_clear=False,  # only for delay
                            )
                            self.add_section_signal_pulse(
                                instance_id, operation.signal, ssp
                            )


def find_sequential_averaging(section) -> Tuple[Any, Tuple]:
    avg_section, sweep_sections = None, ()

    for child_section in section.sections:
        if (
            hasattr(child_section, "averaging_mode")
            and child_section.averaging_mode == AveragingMode.SEQUENTIAL
        ):
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
            f"Field {attr} not found on overrider {self._overrider}"
            f" (type {type(self._overrider)}) nor on base {self._base}"
        )
