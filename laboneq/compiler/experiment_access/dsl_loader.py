# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import typing
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple, Union

from laboneq.compiler.experiment_access.acquire_info import AcquireInfo
from laboneq.compiler.experiment_access.loader_base import LoaderBase
from laboneq.compiler.experiment_access.marker import Marker
from laboneq.compiler.experiment_access.param_ref import ParamRef
from laboneq.compiler.experiment_access.pulse_def import PulseDef
from laboneq.compiler.experiment_access.section_info import SectionInfo
from laboneq.compiler.experiment_access.section_signal_pulse import SectionSignalPulse
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import (
    AcquisitionType,
    AveragingMode,
    IODirection,
    IOSignalType,
)

if typing.TYPE_CHECKING:
    from laboneq.dsl.device import DeviceSetup
    from laboneq.dsl.device.io_units import LogicalSignal
    from laboneq.dsl.experiment import Experiment, ExperimentSignal

_logger = logging.getLogger(__name__)


def find_value_or_parameter_attr(entity: Any, attr: str, value_types: Tuple[type, ...]):
    param = None
    value = getattr(entity, attr, None)
    if value is not None and not isinstance(value, value_types):
        param = getattr(value, "uid", None)
        value = None
    return value, param


class DSLLoader(LoaderBase):
    def load(self, experiment: Experiment, device_setup: DeviceSetup):
        global_leader_device_id = None

        for server in device_setup.servers.values():
            if hasattr(server, "leader_uid"):
                global_leader_device_id = server.leader_uid
            self.add_server(server.uid, server.host, server.port, server.api_level)

        dest_path_devices = {}
        ppc_connections = {}

        reference_clock = None
        for device in device_setup.instruments:
            if hasattr(device, "reference_clock"):
                reference_clock = device.reference_clock

        for device in sorted(device_setup.instruments, key=lambda x: x.uid):

            server = device.server_uid

            driver = type(device).__name__.lower()
            serial = device.address
            interface = device.interface
            is_global_leader = 0
            if global_leader_device_id == device.uid:
                is_global_leader = 1
            reference_clock_source = getattr(device, "reference_clock_source", None)
            is_qc = getattr(device, "is_qc", None)

            self.add_device(
                device.uid,
                driver,
                serial,
                server,
                interface,
                is_global_leader,
                reference_clock,
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
        ls_amplifier_pumps = {}
        self._nt_only_params = []

        all_logical_signals = [
            ls
            for lsg in device_setup.logical_signal_groups.values()
            for ls in lsg.logical_signals.values()
        ]
        for ls in all_logical_signals:
            ls_map[ls.path] = ls

        mapped_logical_signals: Dict["LogicalSignal", "ExperimentSignal"] = {
            # Need to create copy here as we'll possibly patch those ExperimentSignals
            # that touch the same PhysicalChannel
            ls_map[signal.mapped_logical_signal_path]: copy.deepcopy(signal)
            for signal in experiment.signals.values()
        }

        experiment_signals_by_physical_channel = {}
        for ls, exp_signal in mapped_logical_signals.items():
            experiment_signals_by_physical_channel.setdefault(
                ls.physical_channel, []
            ).append(exp_signal)

        from laboneq.dsl.device.io_units.physical_channel import (
            PHYSICAL_CHANNEL_CALIBRATION_FIELDS,
        )

        # Merge the calibration of those ExperimentSignals that touch the same
        # PhysicalChannel.
        for pc, exp_signals in experiment_signals_by_physical_channel.items():
            for field_ in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
                if field_ in ["mixer_calibration", "precompensation"]:
                    continue
                values = set()
                for exp_signal in exp_signals:
                    if not exp_signal.is_calibrated():
                        continue
                    value = getattr(exp_signal, field_)
                    if value is not None:
                        values.add(value)
                if len(values) > 1:
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
                if len(values) > 0:
                    # Make sure all the experiment signals agree.
                    value = values.pop()
                    for exp_signal in exp_signals:
                        if exp_signal.is_calibrated():
                            setattr(exp_signal.calibration, field_, value)

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
                if hasattr(calibration, "port_delay"):
                    ls_port_delays[ls.path] = calibration.port_delay

                if hasattr(calibration, "delay_signal"):
                    ls_delays_signal[ls.path] = calibration.delay_signal

                if hasattr(calibration, "oscillator"):
                    if calibration.oscillator is not None:
                        oscillator = calibration.oscillator
                        is_hardware = oscillator.modulation_type.value == "HARDWARE"

                        oscillator_uid = oscillator.uid

                        frequency_param = None

                        frequency = oscillator.frequency
                        try:
                            frequency = float(frequency)
                        except (ValueError, TypeError):
                            if frequency is not None and hasattr(frequency, "uid"):
                                frequency_param = frequency.uid
                                frequency = None
                            else:
                                raise
                        modulated_paths[ls.path] = {
                            "oscillator_id": oscillator_uid,
                            "is_hardware": is_hardware,
                        }
                        known_oscillator = self._oscillators.get(oscillator_uid)
                        if known_oscillator is None:
                            self.add_oscillator(
                                oscillator_uid, frequency, frequency_param, is_hardware
                            )

                            if is_hardware:
                                device_id = dest_path_devices[ls.path]["device"]
                                self.add_device_oscillator(device_id, oscillator_uid)
                        else:
                            if (
                                known_oscillator["frequency"],
                                known_oscillator["frequency_param"],
                                known_oscillator["hardware"],
                            ) != (frequency, frequency_param, is_hardware):
                                raise Exception(
                                    f"Duplicate oscillator uid {oscillator_uid} found in {ls.path}"
                                )
                try:
                    ls_voltage_offsets[ls.path] = calibration.voltage_offset
                except AttributeError:
                    pass
                try:
                    ls_mixer_calibrations[ls.path] = {
                        "voltage_offsets": calibration.mixer_calibration.voltage_offsets,
                        "correction_matrix": calibration.mixer_calibration.correction_matrix,
                    }
                except (AttributeError, KeyError):
                    pass
                try:
                    precomp = calibration.precompensation
                    if precomp is None:
                        raise AttributeError
                except AttributeError:
                    pass
                else:
                    precomp_dict = {}

                    if precomp.exponential:
                        precomp_exp = [
                            {"timeconstant": e.timeconstant, "amplitude": e.amplitude}
                            for e in precomp.exponential
                        ]
                        precomp_dict["exponential"] = precomp_exp
                    if precomp.high_pass is not None:
                        # Since we currently only support clearing the integrator
                        # inside a delay, the different modes are not relevant.
                        # Instead, we would like to merge subsequent pulses into the
                        # same waveform, so we restrict the choice to "rise", regardless
                        # of what the user may have specified.
                        clearing = "rise"

                        precomp_dict["high_pass"] = {
                            "timeconstant": precomp.high_pass.timeconstant,
                            "clearing": clearing,
                        }
                    if precomp.bounce is not None:
                        precomp_dict["bounce"] = {
                            "delay": precomp.bounce.delay,
                            "amplitude": precomp.bounce.amplitude,
                        }
                    if precomp.FIR is not None:
                        precomp_dict["FIR"] = {
                            "coefficients": copy.deepcopy(precomp.FIR.coefficients),
                        }
                    if precomp_dict:
                        ls_precompensations[ls.path] = precomp_dict

                ls_local_oscillator = getattr(calibration, "local_oscillator")
                if ls_local_oscillator is not None:
                    ls_lo_frequencies[ls.path] = getattr(
                        ls_local_oscillator, "frequency"
                    )
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

                def amplifier_pump_to_dict(amplifier_pump) -> Dict[str, Any]:
                    def opt_param(val) -> Union[str, float]:
                        if val is None or isinstance(val, float):
                            return val
                        self._nt_only_params.append(val.uid)
                        return val.uid

                    return {
                        "pump_freq": opt_param(amplifier_pump.pump_freq),
                        "pump_power": opt_param(amplifier_pump.pump_power),
                        "cancellation": amplifier_pump.cancellation,
                        "alc_engaged": amplifier_pump.alc_engaged,
                        "use_probe": amplifier_pump.use_probe,
                        "probe_frequency": opt_param(amplifier_pump.probe_frequency),
                        "probe_power": opt_param(amplifier_pump.probe_power),
                    }

                if calibration.amplifier_pump is not None:
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
                        ls_amplifier_pumps[ls.path] = (
                            *ppc_connections[ls.path],
                            amplifier_pump_to_dict(calibration.amplifier_pump),
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

            self.add_signal(
                signal.uid,
                signal_type,
                modulation=signal.mapped_logical_signal_path in modulated_paths,
            )

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
                    current_port = device_setup.instrument_by_uid(device).output_by_uid(
                        local_port
                    )
                    if current_port is None:
                        current_port = device_setup.instrument_by_uid(
                            device
                        ).input_by_uid(local_port)
                    if current_port is None:
                        raise RuntimeError(
                            f"local port {local_port} not found in {device_setup.instrument_by_uid(device)}"
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
                        f"Channels for a signal must be distinct, but got {channels} for signal {signal}, connection ports: {local_ports}"
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
                    "range": ls_ranges.get(lsuid),
                    "range_unit": ls_range_units.get(lsuid),
                    "port_delay": ls_port_delays.get(lsuid),
                    "delay_signal": ls_delays_signal.get(lsuid),
                    "port_mode": ls_port_modes.get(lsuid),
                    "threshold": ls_thresholds.get(lsuid),
                    "amplifier_pump": ls_amplifier_pumps.get(lsuid),
                },
            )

        open_inputs = {}
        for instrument in device_setup.instruments:
            for input_obj in instrument.ports:
                if input_obj.direction == IODirection.IN:
                    open_inputs[
                        (instrument.uid, input_obj.signal_type)
                    ] = input_obj.connector_labels

        syncing_connections = []
        for instrument in device_setup.instruments:
            for connection in instrument.connections:
                open_input_found = open_inputs.get(
                    (connection.remote_path, connection.signal_type)
                )
                output = instrument.output_by_uid(connection.local_port)

                if open_input_found is not None:
                    syncing_connections.append(
                        (
                            instrument.uid,
                            connection.remote_path,
                            connection.signal_type,
                            open_input_found,
                            output,
                        )
                    )

        for syncing_connection in syncing_connections:
            signal_type = syncing_connection[2]
            assert isinstance(syncing_connection[2], type(IOSignalType.DIO))
            if signal_type == IOSignalType.DIO:
                dio_leader = syncing_connection[0]
                dio_follower = syncing_connection[1]
                self._dios.append((dio_leader, dio_follower))

            elif signal_type == IOSignalType.ZSYNC:
                zsync_leader = syncing_connection[0]
                zsync_follower = syncing_connection[1]
                port = syncing_connection[4].physical_port_ids[0]
                self._pqsc_ports.append((zsync_leader, zsync_follower, int(port)))

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

        if seq_avg_section is not None and len(sweep_sections):
            avg_children = self._section_tree.get(sweep_sections[0].uid, [])
            sweep_children = self._section_tree.get(seq_avg_section.uid, [])

            self._section_tree[seq_avg_section.uid] = avg_children
            self._section_tree[sweep_sections[0].uid] = sweep_children

            for _parent, children in self._section_tree.items():
                for i, c in enumerate(children):
                    if c == sweep_sections[0].uid:
                        children[i] = seq_avg_section.uid
                    elif c == seq_avg_section.uid:
                        children[i] = sweep_sections[0].uid

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

    def _insert_section(
        self,
        section,
        acquisition_type,
        exchanger_map: Callable[[Any], Any],
        instance_id: str,
    ):
        has_repeat = False
        count = 1

        averaging_type = None
        if hasattr(section, "count"):
            has_repeat = True
            count = section.count
            if hasattr(section, "averaging_mode"):
                if section.averaging_mode.value in ["cyclic", "sequential"]:
                    averaging_type = "hardware"

        if hasattr(section, "parameters"):
            for parameter in section.parameters:
                values_list = None
                if parameter.values is not None:
                    values_list = list(parameter.values)
                axis_name = getattr(parameter, "axis_name", None)
                has_repeat = True
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
                        f"Repeat count must be at least 1, but section {section.uid} has count={count}"
                    )
                if (
                    section.execution_type is not None
                    and section.execution_type.value == "hardware"
                    and parameter.uid in self._nt_only_params
                ):
                    raise Exception(
                        f"Parameter {parameter.uid} can't be swept in real-time, it is bound to a value "
                        f"that can only be set in near-time"
                    )

        execution_type = None
        if section.execution_type is not None:
            execution_type = section.execution_type.value

        align = "left"
        if exchanger_map(section).alignment is not None:
            align = exchanger_map(section).alignment.value

        on_system_grid = None
        if exchanger_map(section).on_system_grid is not None:
            on_system_grid = exchanger_map(section).on_system_grid

        length = None
        if section.length is not None:
            length = section.length

        averaging_mode = None
        if hasattr(section, "averaging_mode"):
            averaging_mode = section.averaging_mode.value

        repetition_mode = None
        if hasattr(section, "repetition_mode"):
            repetition_mode = section.repetition_mode.value

        repetition_time = None
        if hasattr(section, "repetition_time"):
            repetition_time = section.repetition_time

        reset_oscillator_phase = False
        if hasattr(section, "reset_oscillator_phase"):
            reset_oscillator_phase = section.reset_oscillator_phase

        handle = None
        if hasattr(section, "handle"):
            handle = section.handle

        state = None
        if hasattr(section, "state"):
            state = section.state

        local = None
        if hasattr(section, "local"):
            local = section.local

        trigger = [
            {"signal_id": k, "state": v["state"]} for k, v in section.trigger.items()
        ]

        acquisition_types = None
        for operation in exchanger_map(section).operations:
            if hasattr(operation, "handle"):
                # an acquire event - add acquisition_types
                acquisition_types = [acquisition_type.value]

        self.add_section(
            instance_id,
            SectionInfo(
                section_id=instance_id,
                section_display_name=section.uid,
                has_repeat=has_repeat,
                execution_type=execution_type,
                count=count,
                acquisition_types=acquisition_types,
                averaging_type=averaging_type,
                align=align,
                on_system_grid=on_system_grid,
                length=length,
                averaging_mode=averaging_mode,
                repetition_mode=repetition_mode,
                repetition_time=repetition_time,
                play_after=getattr(section, "play_after", None),
                reset_oscillator_phase=reset_oscillator_phase,
                trigger_output=trigger,
                handle=handle,
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

        for operation in exchanger_map(section).operations:
            if hasattr(operation, "signal"):
                pulse_offset = None
                pulse_offset_param = None

                if hasattr(operation, "time"):  # Delay operation

                    pulse_offset = operation.time
                    precompensation_clear = (
                        getattr(operation, "precompensation_clear", None) or False
                    )
                    if not isinstance(operation.time, float) and not isinstance(
                        operation.time, int
                    ):
                        pulse_offset = None
                        pulse_offset_param = operation.time.uid

                    ssp = SectionSignalPulse(
                        signal_id=operation.signal,
                        offset=pulse_offset,
                        offset_param=pulse_offset_param,
                        precompensation_clear=precompensation_clear,
                    )
                    self.add_section_signal_pulse(instance_id, operation.signal, ssp)
                else:  # All operations, except Delay
                    pulse = None
                    operation_length_param = None

                    if hasattr(operation, "pulse"):
                        pulse = getattr(operation, "pulse")
                    if hasattr(operation, "kernel"):
                        pulse = getattr(operation, "kernel")
                    length = getattr(operation, "length", None)
                    operation_length = length
                    if (
                        operation_length is not None
                        and not isinstance(operation_length, float)
                        and not isinstance(operation_length, complex)
                        and not isinstance(operation_length, int)
                    ):
                        operation_length_param = operation_length.uid
                        operation_length = None
                    if pulse is None and length is not None:
                        pulse = SimpleNamespace()
                        setattr(pulse, "uid", uuid.uuid4().hex)
                        setattr(pulse, "length", length)
                    if hasattr(operation, "handle") and pulse is None:
                        raise RuntimeError(
                            f"Either 'kernel' or 'length' must be provided for the acquire operation with handle '{getattr(operation, 'handle')}'."
                        )
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

                            amplitude, amplitude_param = find_value_or_parameter_attr(
                                pulse, "amplitude", (float, int, complex)
                            )

                            can_compress = False
                            if hasattr(pulse, "can_compress"):
                                can_compress = pulse.can_compress

                            self.add_pulse(
                                pulse.uid,
                                PulseDef(
                                    id=pulse.uid,
                                    function=function,
                                    length=length,
                                    amplitude=amplitude,
                                    amplitude_param=amplitude_param,
                                    play_mode=None,
                                    can_compress=can_compress,
                                    samples=samples,
                                ),
                            )
                        (
                            pulse_amplitude,
                            pulse_amplitude_param,
                        ) = find_value_or_parameter_attr(
                            operation, "amplitude", (int, float, complex)
                        )
                        pulse_phase, pulse_phase_param = find_value_or_parameter_attr(
                            operation, "phase", (int, float)
                        )
                        (
                            pulse_increment_oscillator_phase,
                            pulse_increment_oscillator_phase_param,
                        ) = find_value_or_parameter_attr(
                            operation, "increment_oscillator_phase", (int, float)
                        )
                        (
                            pulse_set_oscillator_phase,
                            pulse_set_oscillator_phase_param,
                        ) = find_value_or_parameter_attr(
                            operation, "set_oscillator_phase", (int, float)
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

                        # Replace sweep params with a ParamRef
                        if pulse_parameters is not None:
                            for param, val in pulse_parameters.items():
                                if hasattr(val, "uid"):
                                    # Take the presence of "uid" as a proxy for isinstance(val, SweepParameter)
                                    pulse_parameters[param] = ParamRef(val.uid)
                        if operation_pulse_parameters is not None:
                            for param, val in operation_pulse_parameters.items():
                                if hasattr(val, "uid"):
                                    # Take the presence of "uid" as a proxy for isinstance(val, SweepParameter)
                                    operation_pulse_parameters[param] = ParamRef(
                                        val.uid
                                    )

                        markers = None
                        if hasattr(operation, "marker"):
                            markers_raw = operation.marker
                            if markers_raw is not None:
                                markers = []
                                for k, v in markers_raw.items():
                                    marker_pulse_id = None
                                    pulse_ref = v.get("waveform")
                                    if pulse_ref is not None:
                                        marker_pulse_id = pulse_ref["$ref"]

                                    markers.append(
                                        Marker(
                                            k,
                                            enable=v.get("enable"),
                                            start=v.get("start"),
                                            length=v.get("length"),
                                            pulse_id=marker_pulse_id,
                                        )
                                    )
                                    self.add_signal_marker(operation.signal, k)

                        ssp = SectionSignalPulse(
                            signal_id=operation.signal,
                            pulse_id=pulse.uid,
                            offset=pulse_offset,
                            offset_param=pulse_offset_param,
                            amplitude=pulse_amplitude,
                            amplitude_param=pulse_amplitude_param,
                            length=operation_length,
                            length_param=operation_length_param,
                            acquire_params=acquire_params,
                            phase=pulse_phase,
                            phase_param=pulse_phase_param,
                            increment_oscillator_phase=pulse_increment_oscillator_phase,
                            increment_oscillator_phase_param=pulse_increment_oscillator_phase_param,
                            set_oscillator_phase=pulse_set_oscillator_phase,
                            set_oscillator_phase_param=pulse_set_oscillator_phase_param,
                            play_pulse_parameters=operation_pulse_parameters,
                            pulse_pulse_parameters=pulse_parameters,
                            precompensation_clear=False,  # not supported
                            markers=markers,
                        )
                        self.add_section_signal_pulse(
                            instance_id, operation.signal, ssp
                        )
                    elif (
                        getattr(operation, "increment_oscillator_phase", None)
                        is not None
                        or getattr(operation, "set_oscillator_phase", None) is not None
                        or getattr(operation, "phase", None) is not None
                    ):
                        if getattr(operation, "phase", None) is not None:
                            raise LabOneQException(
                                "Phase argument has no effect for virtual Z gates."
                            )
                        # virtual Z gate
                        (
                            pulse_increment_oscillator_phase,
                            pulse_increment_oscillator_phase_param,
                        ) = find_value_or_parameter_attr(
                            operation, "increment_oscillator_phase", (int, float)
                        )
                        (
                            pulse_set_oscillator_phase,
                            pulse_set_oscillator_phase_param,
                        ) = find_value_or_parameter_attr(
                            operation, "set_oscillator_phase", (int, float)
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
                            signal_id=operation.signal,
                            offset=pulse_offset,
                            offset_param=pulse_offset_param,
                            increment_oscillator_phase=pulse_increment_oscillator_phase,
                            increment_oscillator_phase_param=pulse_increment_oscillator_phase_param,
                            set_oscillator_phase=pulse_set_oscillator_phase,
                            set_oscillator_phase_param=pulse_set_oscillator_phase_param,
                            precompensation_clear=False,  # not supported
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
            f"Field {attr} not found on overrider {self._overrider} (type {type(self._overrider)}) nor on base {self._base}"
        )
