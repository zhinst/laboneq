# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import uuid
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace

from typing import List, Dict, Tuple, Optional, Set, Any

from jsonschema import ValidationError
from jsonschema.validators import validator_for
import copy

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AveragingMode, IODirection, AcquisitionType

_logger = logging.getLogger(__name__)

import typing

if typing.TYPE_CHECKING:
    from laboneq.dsl.experiment import ExperimentSignal
    from laboneq.dsl.device.io_units import LogicalSignal


def find_value_or_parameter_dict(pulse_ref, name, types):
    value = None
    param = None
    try:
        value = pulse_ref[name]
        if value is not None and not isinstance(value, types):
            param = value["$ref"]
            value = None
    except KeyError:
        pass
    return value, param


def find_value_or_parameter_attr(operation, name, types):
    value = None
    param = None
    try:
        value = getattr(operation, name)
        if value is not None and not isinstance(value, types):
            param = value.uid
            value = None
    except AttributeError:
        pass
    return value, param


class ExperimentDAO:

    _validator = None

    def __init__(self, experiment, core_device_setup=None, core_experiment=None):
        self._data = {}
        self._acquisition_type: AcquisitionType = None
        if core_device_setup is not None and core_experiment is not None:
            self._load_from_core(core_device_setup, core_experiment)
        else:
            try:
                validator = ExperimentDAO.schema_validator()
                validator.validate(experiment)
            except ValidationError as exception:
                _logger.warning("Failed to validate input:")
                for line in str(exception).splitlines():
                    _logger.warning("validation error: %s", line)

            self._load_experiment(experiment)
        self.validate_experiment()

    @property
    def acquisition_type(self) -> AcquisitionType:
        return self._acquisition_type

    def add_signal(
        self, device_id, channels, connection_type, signal_id, signal_type, modulation
    ):
        seq_nr = len(self.signals()) + 1

        self._data["signal"][signal_id] = {
            "signal_id": signal_id,
            "signal_type": signal_type,
            "modulation": modulation,
            "seq_nr": seq_nr,
        }

        self._data["signal_connection"].append(
            {
                "signal_id": signal_id,
                "device_id": device_id,
                "connection_type": connection_type,
                "channels": channels,
                "mixer_calibration": None,
                "lo_frequency": None,
                "range": None,
                "port_delay": None,
                "delay_signal": None,
                "port_mode": None,
                "threshold": None,
            }
        )

    def _load_experiment(self, experiment):

        self._data["server"] = {}
        for server in experiment["servers"]:
            self._data["server"][server["id"]] = {
                k: server.get(k) for k in ["id", "host", "port", "api_level"]
            }

        self._data["device"] = OrderedDict()
        self._data["device_oscillator"] = []

        for device in sorted(experiment["devices"], key=lambda x: x["id"]):
            if "server" in device:
                server = device["server"]["$ref"]
            else:
                server = None
            if "driver" in device:
                driver = device["driver"]
            else:
                driver = device["device_type"]

            if "serial" in device:
                serial = device["serial"]
            else:
                serial = None

            if "interface" in device:
                interface = device["interface"]
            else:
                interface = None

            if (
                "reference_clock_source" in device
                and device["reference_clock_source"] is not None
            ):
                reference_clock_source = device["reference_clock_source"]
            else:
                reference_clock_source = None

            self._data["device"][device["id"]] = {
                "id": device["id"],
                "device_type": driver,
                "serial": serial,
                "server": server,
                "interface": interface,
                "reference_clock_source": reference_clock_source,
            }

            if "oscillators_list" in device:
                for oscillator_ref in device["oscillators_list"]:
                    self._data["device_oscillator"].append(
                        {
                            "device_id": device["id"],
                            "oscillator_id": oscillator_ref["$ref"],
                        }
                    )

        self._data["oscillator"] = OrderedDict()

        if "oscillators" in experiment:
            for oscillator in experiment["oscillators"]:
                frequency = None
                frequency_param = None
                if "frequency" in oscillator:
                    frequency = oscillator["frequency"]
                    if not isinstance(frequency, float) and not isinstance(
                        frequency, int
                    ):
                        if frequency is not None and "$ref" in frequency:
                            frequency_param = frequency["$ref"]
                        frequency = None

                self._data["oscillator"][oscillator["id"]] = {
                    "id": oscillator["id"],
                    "frequency": frequency,
                    "frequency_param": frequency_param,
                    "hardware": True if oscillator["hardware"] else False,
                }

        self._data["dio"] = []
        self._data["pqsc_port"] = []
        if "connectivity" in experiment:
            if "dios" in experiment["connectivity"]:
                for dio in experiment["connectivity"]["dios"]:
                    self._data["dio"].append(
                        (dio["leader"]["$ref"], dio["follower"]["$ref"])
                    )
            if "leader" in experiment["connectivity"]:

                leader_device_id = experiment["connectivity"]["leader"]["$ref"]
                self._data["device"][leader_device_id]["is_global_leader"] = True

            if "reference_clock" in experiment["connectivity"]:
                reference_clock = experiment["connectivity"]["reference_clock"]
                for device in self._data["device"].values():
                    if device["device_type"] in {"hdawg", "uhfqa", "pqsc"}:
                        device["reference_clock"] = reference_clock

            if "pqscs" in experiment["connectivity"]:
                pqscs = experiment["connectivity"]["pqscs"]
                for pqsc in pqscs:
                    pqsc_device_id = pqsc["device"]["$ref"]
                    if "ports" in pqsc:
                        for port in pqsc["ports"]:
                            self._data["pqsc_port"].append(
                                (pqsc_device_id, port["device"]["$ref"], port["port"])
                            )
        self._data["signal"] = {}
        self._data["signal_oscillator"] = []

        for seq_nr, signal in enumerate(experiment["signals"]):

            self._data["signal"][signal["id"]] = {
                "signal_id": signal["id"],
                "signal_type": signal["signal_type"],
                "modulation": True if signal.get("modulation") else False,
                "seq_nr": seq_nr,
            }
            if "oscillators_list" in signal:
                for oscillator_ref in signal["oscillators_list"]:
                    oscillator_id = oscillator_ref["$ref"]
                    self._data["signal_oscillator"].append(
                        {"signal_id": signal["id"], "oscillator_id": oscillator_id}
                    )

        self._data["signal_connection"] = []
        for connection in experiment["signal_connections"]:
            if "mixer_calibration" in connection:
                mixer_calibration = copy.deepcopy(connection["mixer_calibration"])
            else:
                mixer_calibration = None
            range = connection.get("range")
            lo_frequency = connection.get("lo_frequency")
            port_delay = connection.get("port_delay")
            delay_signal = connection.get("delay_signal")

            self._data["signal_connection"].append(
                {
                    "signal_id": connection["signal"]["$ref"],
                    "device_id": connection["device"]["$ref"],
                    "connection_type": connection["connection"]["type"],
                    "channels": connection["connection"]["channels"],
                    "mixer_calibration": mixer_calibration,
                    "lo_frequency": lo_frequency,
                    "range": range,
                    "port_delay": port_delay,
                    "delay_signal": delay_signal,
                    "port_mode": None,
                    "threshold": None,
                }
            )

        self._data["pulse"] = {}
        for pulse in experiment["pulses"]:
            samples = pulse.get("samples", None)

            amplitude, amplitude_param = find_value_or_parameter_dict(
                pulse, "amplitude", (int, float, complex)
            )

            self._data["pulse"][pulse["id"]] = {
                "id": pulse["id"],
                "function": pulse.get("function"),
                "length": pulse.get("length"),
                "amplitude": amplitude,
                "amplitude_param": amplitude_param,
                "play_mode": pulse.get("play_mode"),
                "samples": samples,
            }

        self._data["section_parameter"] = []
        self._data["section_tree"] = []
        self._data["section_signal"] = []
        self._data["section_signal_pulse"] = []

        self._data["section"] = {}

        section_signal_id = 0

        for section in sorted(experiment["sections"], key=lambda x: x["id"]):
            has_repeat = 0
            execution_type = None
            align = None
            offset = None
            offset_param = None
            length = None
            count = 1
            experiment_seq = None
            for index, experiment_section in enumerate(
                experiment["experiment"]["sections_list"]
            ):
                if experiment_section["$ref"] == section["id"]:
                    experiment_seq = index
                    break

            averaging_type = None

            if "repeat" in section:
                has_repeat = 1
                execution_type = section["repeat"]["execution_type"]
                if "averaging_type" in section["repeat"]:
                    averaging_type = section["repeat"]["averaging_type"]

                count = section["repeat"]["count"]
                if "parameters" in section["repeat"]:
                    for parameter in section["repeat"]["parameters"]:
                        values = None
                        if parameter.get("values") is not None:
                            values = copy.deepcopy(list(parameter["values"]))

                        self._data["section_parameter"].append(
                            {
                                "section_id": section["id"],
                                "id": parameter["id"],
                                "start": parameter.get("start"),
                                "step": parameter.get("step"),
                                "values": values,
                                "axis_name": None,
                            }
                        )

            sections_list = None
            if "repeat" in section and "sections_list" in section["repeat"]:
                sections_list = section["repeat"]["sections_list"]
            if "sections_list" in section:
                sections_list = section["sections_list"]

            if sections_list is not None:
                for index, child_section_ref in enumerate(sections_list):
                    self._data["section_tree"].append(
                        {
                            "parent_section_id": section["id"],
                            "child_section_id": child_section_ref["$ref"],
                            "index": index,
                        }
                    )

            trigger = section.get("trigger")
            if self._acquisition_type is None and trigger is not None:
                self._acquisition_type = AcquisitionType(trigger[0])

            if "align" in section:
                align = section["align"]

            if "offset" in section:
                offset = section["offset"]
                if not isinstance(offset, float) and not isinstance(offset, int):
                    if offset is not None and "$ref" in offset:
                        offset_param = offset["$ref"]
                        offset = None

            reset_oscillator_phase = False
            if "reset_oscillator_phase" in section:
                reset_oscillator_phase = section["reset_oscillator_phase"]

            if "length" in section:
                length = section["length"]

            averaging_mode = None
            if "averaging_mode" in section:
                averaging_mode = section["averaging_mode"]

            repetition_time = None
            if "repetition_time" in section:
                repetition_time = section["repetition_time"]

            repetition_mode = None
            if "repetition_mode" in section:
                repetition_mode = section["repetition_mode"]

            play_after = section.get("play_after")

            self._data["section"][section["id"]] = dict(
                section_id=section["id"],
                experiment_seq=experiment_seq,
                has_repeat=has_repeat,
                execution_type=execution_type,
                count=count,
                trigger=trigger,
                averaging_type=averaging_type,
                align=align,
                offset=offset,
                offset_param=offset_param,
                length=length,
                averaging_mode=averaging_mode,
                repetition_mode=repetition_mode,
                repetition_time=repetition_time,
                play_after=play_after,
                reset_oscillator_phase=reset_oscillator_phase,
            )

            if "signals_list" in section:
                for signals_list_entry in section["signals_list"]:
                    signal_id = signals_list_entry["signal"]["$ref"]
                    self._data["section_signal"].append(
                        {
                            "id": section_signal_id,
                            "section_id": section["id"],
                            "signal_id": signal_id,
                        }
                    )

                    if "pulses_list" in signals_list_entry:
                        seq_nr = 0
                        for pulse_ref in signals_list_entry["pulses_list"]:
                            (
                                pulse_offset,
                                pulse_offset_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "offset", (int, float)
                            )
                            (
                                pulse_amplitude,
                                pulse_amplitude_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "amplitude", (int, float, complex)
                            )
                            (
                                pulse_increment,
                                pulse_increment_oscillator_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "increment_oscillator_phase", (int, float)
                            )
                            (
                                pulse_set_oscillator_phase,
                                pulse_set_oscillator_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "set_oscillator_phase", (int, float)
                            )
                            (
                                pulse_phase,
                                pulse_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "phase", (int, float)
                            )
                            (
                                resulting_pulse_instance_length,
                                resulting_pulse_instance_length_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "length", (int, float)
                            )

                            pulse_id = None
                            if pulse_ref.get("pulse") is not None:
                                pulse_id = pulse_ref["pulse"]["$ref"]

                            new_ssp = dict(
                                section_signal_id=section_signal_id,
                                pulse_id=pulse_id,
                                section_id=section["id"],
                                signal_id=signal_id,
                                offset=pulse_offset,
                                offset_param=pulse_offset_param,
                                amplitude=pulse_amplitude,
                                amplitude_param=pulse_amplitude_param,
                                length=resulting_pulse_instance_length,
                                length_param=resulting_pulse_instance_length_param,
                                seq_nr=seq_nr,
                                acquire_params=None,
                                phase=pulse_phase,
                                phase_param=pulse_phase_param,
                                increment_oscillator_phase=pulse_increment,
                                increment_oscillator_phase_param=pulse_increment_oscillator_phase_param,
                                set_oscillator_phase=pulse_set_oscillator_phase,
                                set_oscillator_phase_param=pulse_set_oscillator_phase_param,
                            )
                            self._data["section_signal_pulse"].append(new_ssp)
                            seq_nr += 1
                    section_signal_id += 1

    def server_infos(self):
        return copy.deepcopy(list(self._data["server"].values()))

    def signals(self):
        return [s["signal_id"] for s in self._data["signal"].values()]

    def devices(self):
        return [d["id"] for d in self._data["device"].values()]

    def global_leader_device(self):
        try:
            return next(
                d for d in self._data["device"].values() if d.get("is_global_leader")
            )["id"]
        except StopIteration:
            return None

    @classmethod
    def _device_info_keys(cls):
        return [
            "id",
            "device_type",
            "serial",
            "server",
            "interface",
            "reference_clock_source",
        ]

    def device_info(self, device_id):
        device_info = self._data["device"].get(device_id)
        if device_info is not None:
            return {k: device_info[k] for k in self._device_info_keys()}
        return None

    def device_infos(self):
        return [
            {k: device_info[k] for k in self._device_info_keys()}
            for device_info in self._data["device"].values()
        ]

    def device_reference_clock(self, device_id):
        return self._data["device"][device_id].get("reference_clock")

    def device_from_signal(self, signal_id):
        signal_connection = next(
            c for c in self._data["signal_connection"] if c["signal_id"] == signal_id
        )
        return signal_connection["device_id"]

    def device_signals(self, device_id):
        return [
            sc["signal_id"]
            for sc in self._data["signal_connection"]
            if sc["device_id"] == device_id
            and sc["device_id"] in self._data["device"].keys()
        ]

    def devices_in_section_no_descend(self, section_id):
        section_signals = {
            ss["signal_id"]
            for ss in self._data["section_signal"]
            if ss["section_id"] == section_id
        }
        devices = {
            sc["device_id"]
            for sc in self._data["signal_connection"]
            if sc["signal_id"] in section_signals
        }
        return devices

    def _device_types_in_section_no_descend(self, section_id):
        devices = self.devices_in_section_no_descend(section_id)
        return {
            d["device_type"]
            for d in self._data["device"].values()
            if d["id"] in devices
        }

    def device_types_in_section(self, section_id):
        retval = set()
        section_with_children = self.all_section_children(section_id)
        section_with_children.add(section_id)
        for child in section_with_children:
            retval = retval.union(self._device_types_in_section_no_descend(child))
        return retval

    @classmethod
    def _signal_info_keys(cls):
        return [
            "signal_id",
            "signal_type",
            "device_id",
            "device_serial",
            "device_type",
            "connection_type",
            "channels",
            "delay_signal",
            "modulation",
        ]

    def signal_info(self, signal_id):

        signal_info = self._data["signal"].get(signal_id)
        if signal_info is not None:
            signal_info_copy = copy.deepcopy(signal_info)
            signal_connection = next(
                sc
                for sc in self._data["signal_connection"]
                if sc["signal_id"] == signal_id
            )

            for k in ["device_id", "connection_type", "channels", "delay_signal"]:
                signal_info_copy[k] = signal_connection[k]

            device_info = self._data["device"][signal_connection["device_id"]]

            signal_info_copy["device_type"] = device_info["device_type"]
            signal_info_copy["device_serial"] = device_info["serial"]
            return {k: signal_info_copy[k] for k in self._signal_info_keys()}
        else:
            raise Exception(f"Signal_id {signal_id} not found")

    def sections(self) -> List[str]:
        return list(self._data["section"].keys())

    def section_info(self, section_id):
        retval = {
            k: self._data["section"][section_id][k]
            for k in [
                "section_id",
                "has_repeat",
                "execution_type",
                "trigger",
                "averaging_type",
                "count",
                "align",
                "offset",
                "offset_param",
                "length",
                "averaging_mode",
                "repetition_mode",
                "repetition_time",
                "play_after",
                "reset_oscillator_phase",
            ]
        }
        if retval["count"] is not None:
            retval["count"] = int(retval["count"])
        return retval

    def root_sections(self):
        return [
            s["section_id"]
            for s in self._data["section"].values()
            if s["experiment_seq"] is not None
        ]

    def direct_section_children(self, section_id):
        return [
            t["child_section_id"]
            for t in self._data["section_tree"]
            if t["parent_section_id"] == section_id
        ]

    def all_section_children(self, section_id):
        retval = []

        direct_children = self.direct_section_children(section_id)
        retval = retval + direct_children
        for child in direct_children:
            retval = retval + list(self.all_section_children(child))

        return set(retval)

    def section_parent(self, section_id):
        try:
            return next(
                s
                for s in self._data["section_tree"]
                if s["child_section_id"] == section_id
            )["parent_section_id"]
        except StopIteration:
            return None

    def pqscs(self):
        return [p[0] for p in self._data["pqsc_port"]]

    def pqsc_ports(self, pqsc_device_id):
        return [
            {"device": p[1], "port": p[2]}
            for p in self._data["pqsc_port"]
            if p[0] == pqsc_device_id
        ]

    def dio_followers(self):
        return [d[1] for d in self._data["dio"]]

    def dio_leader(self, device_id):
        try:
            return next(d[0] for d in self._data["dio"] if d[1] == device_id)
        except StopIteration:
            return None

    def dio_connections(self):
        return [(dio[0], dio[1]) for dio in self._data["dio"]]

    def is_dio_leader(self, device_id):
        return bool({d[1] for d in self._data["dio"] if d[0] == device_id})

    def section_signals(self, section_id):
        return set(self._section_signals_list(section_id))

    def _section_signals_list(self, section_id):
        retval = []
        distinct = set()
        for ss in sorted(self._data["section_signal"], key=lambda x: x["id"]):
            if ss["section_id"] == section_id:
                if ss["signal_id"] not in distinct:
                    retval.append(ss["signal_id"])
                    distinct.add(ss["signal_id"])
        return retval

    def section_signals_with_children(self, section_id):
        retval = set()
        section_with_children = self.all_section_children(section_id)
        section_with_children.add(section_id)
        for child in section_with_children:
            retval = retval.union(self.section_signals(child))
        return retval

    @classmethod
    def _pulse_info_fields(cls):
        return [
            "id",
            "function",
            "length",
            "amplitude",
            "amplitude_param",
            "play_mode",
            "samples",
        ]

    def pulses(self):
        return [
            {k: p.get(k) for k in self._pulse_info_fields()}
            for p in self._data["pulse"].values()
        ]

    def pulses_dict(self):
        return {pulse["id"]: pulse for pulse in self.pulses()}

    def pulse(self, pulse_id):
        pulse = self._data["pulse"].get(pulse_id)
        if pulse is None:
            return None
        return {k: pulse.get(k) for k in self._pulse_info_fields()}

    @classmethod
    def _oscillator_info_fields(cls):
        return ["id", "frequency", "frequency_param", "hardware"]

    def oscillator_info(self, oscillator_id):
        oscillator = self._data["oscillator"].get(oscillator_id)
        if oscillator is None:
            return None
        return {k: oscillator[k] for k in self._oscillator_info_fields()}

    def hardware_oscillators(self):
        oscillator_infos = []
        for device_oscillator in self._data["device_oscillator"]:
            oscillator = self._data["oscillator"].get(
                device_oscillator["oscillator_id"]
            )
            if oscillator is not None:
                if oscillator["hardware"]:
                    oscillator_info = {
                        k: oscillator[k] for k in self._oscillator_info_fields()
                    }
                    oscillator_info["device_id"] = device_oscillator["device_id"]
                    oscillator_infos.append(oscillator_info)

        return list(sorted(oscillator_infos, key=lambda x: (x["device_id"], x["id"])))

    def device_oscillators(self, device_id):
        return [
            do["oscillator_id"]
            for do in self._data["device_oscillator"]
            if do["device_id"] == device_id
        ]

    def oscillators(self):
        return list(self._data["oscillator"].keys())

    def signal_oscillator(self, signal_id):
        signal_oscillators = [
            so
            for so in self._data["signal_oscillator"]
            if so["signal_id"] == signal_id
            and so["oscillator_id"] in self._data["oscillator"]
        ]
        retval = [
            {k: oscillator[k] for k in self._oscillator_info_fields()}
            for oscillator in [
                self._data["oscillator"][so["oscillator_id"]]
                for so in signal_oscillators
            ]
        ]

        if len(retval) > 1:
            raise Exception(
                f"Multiple oscillators assigned to signal '{signal_id}'. This is deprecated, use multiple signals instead."
            )

        return retval[0] if len(retval) > 0 else None

    def mixer_calibration(self, signal_id):
        return next(
            sc for sc in self._data["signal_connection"] if sc["signal_id"] == signal_id
        )["mixer_calibration"]

    def lo_frequency(self, signal_id):
        return next(
            sc for sc in self._data["signal_connection"] if sc["signal_id"] == signal_id
        )["lo_frequency"]

    def signal_range(self, signal_id):
        return next(
            sc for sc in self._data["signal_connection"] if sc["signal_id"] == signal_id
        )["range"]

    def port_delay(self, signal_id):
        return next(
            sc for sc in self._data["signal_connection"] if sc["signal_id"] == signal_id
        )["port_delay"]

    def port_mode(self, signal_id):
        return next(
            sc for sc in self._data["signal_connection"] if sc["signal_id"] == signal_id
        )["port_mode"]

    def threshold(self, signal_id):
        return next(
            sc for sc in self._data["signal_connection"] if sc["signal_id"] == signal_id
        )["threshold"]

    def section_pulses(self, section_id, signal_id):
        retval = self._section_pulses_raw(section_id, signal_id)
        for sp in retval:
            pulse_id = sp.get("pulse_id")
            if pulse_id is not None:
                pulse_def = self.pulse(pulse_id)
                if pulse_def is not None:
                    if sp["length"] is None and sp["length_param"] is None:
                        sp["length"] = pulse_def.get("length")
                        if pulse_def.get("length_param") is not None:
                            sp["length_param"] = pulse_def["length_param"]
                            sp["length"] = None

        return retval

    def _section_pulses_raw(self, section_id, signal_id):
        try:
            section_signal = next(
                si
                for si in self._data["section_signal"]
                if si["section_id"] == section_id and si["signal_id"] == signal_id
            )
        except StopIteration:
            return []
        section_signal_id = section_signal["id"]
        section_signal_pulses = [
            ssp
            for ssp in self._data["section_signal_pulse"]
            if ssp["section_signal_id"] == section_signal_id
        ]

        retval = [
            {
                k: ssp.get(k)
                for k in [
                    "section_id",
                    "seq_nr",
                    "pulse_id",
                    "length",
                    "length_param",
                    "amplitude",
                    "amplitude_param",
                    "play_mode",
                    "signal_id",
                    "offset",
                    "offset_param",
                    "acquire_params",
                    "phase",
                    "phase_param",
                    "increment_oscillator_phase",
                    "increment_oscillator_phase_param",
                    "set_oscillator_phase",
                    "set_oscillator_phase_param",
                ]
            }
            for ssp in sorted(
                section_signal_pulses,
                key=lambda x: (x["section_id"], x["signal_id"], x["seq_nr"]),
            )
        ]
        return retval

    def section_parameters(self, section_id):
        return [
            {k: p.get(k) for k in ["id", "start", "step", "values", "axis_name"]}
            for p in self._data["section_parameter"]
            if p["section_id"] == section_id
        ]

    @staticmethod
    def experiment_json_schema():
        with open(
            os.path.join(Path(__file__).parent.absolute(), "qccs-schema_2_5_0.json")
        ) as schema_file:
            return json.load(schema_file)

    @staticmethod
    def schema_validator():
        if ExperimentDAO._validator is None:
            schema = ExperimentDAO.experiment_json_schema()
            validator_cls = validator_for(schema)
            validator_cls.check_schema(schema)
            validator = validator_cls(schema)
            ExperimentDAO._validator = validator
        return ExperimentDAO._validator

    @staticmethod
    def dump(experiment_dao: "ExperimentDAO"):
        retval = {
            "$schema": "../../schemas/qccs-schema_2_5_0.json",
            "metadata": {
                "version": "2.5.0",
                "unit": {"time": "s", "frequency": "Hz", "phase": "rad"},
                "epsilon": {"time": 1e-12},
                "line_endings": "unix",
            },
        }
        retval["servers"] = []
        for server_info in experiment_dao.server_infos():
            retval["servers"].append(
                {
                    "id": server_info["id"],
                    "host": server_info["host"],
                    "port": int(server_info["port"]),
                    "api_level": server_info["api_level"],
                }
            )

        device_entries = {}
        reference_clock = None
        for device in experiment_dao.devices():
            device_info = experiment_dao.device_info(device)

            device_entry = {}
            for key in ["id", "serial", "interface", "reference_clock_source"]:
                if device_info[key] is not None:
                    device_entry[key] = device_info[key]
            device_entry["driver"] = device_info["device_type"].lower()

            oscillator_ids = experiment_dao.device_oscillators(device)

            if len(oscillator_ids) > 0:
                device_entry["oscillators_list"] = [
                    {"$ref": oscillator_id} for oscillator_id in oscillator_ids
                ]
            if device_info["server"] is not None:
                device_entry["server"] = {"$ref": device_info["server"]}
            device_entries[device_entry["id"]] = device_entry
            reference_clock = experiment_dao.device_reference_clock(device)

        retval["devices"] = list(sorted(device_entries.values(), key=lambda x: x["id"]))

        connectivity_object = {}
        if experiment_dao.global_leader_device():
            connectivity_object = {
                "leader": {"$ref": experiment_dao.global_leader_device()}
            }

        if reference_clock is not None:
            connectivity_object["reference_clock"] = reference_clock
        dios = []
        for dio_connection in experiment_dao.dio_connections():
            dios.append(
                {
                    "leader": {"$ref": dio_connection[0]},
                    "follower": {"$ref": dio_connection[1]},
                }
            )
        if len(dios) > 0:
            connectivity_object["dios"] = dios

        pqscs = []

        for pqsc in experiment_dao.pqscs():
            pqsc_entry = {"device": {"$ref": pqsc}, "ports": []}

            for port_info in experiment_dao.pqsc_ports(pqsc):
                pqsc_entry["ports"].append(
                    {"device": {"$ref": port_info["device"]}, "port": port_info["port"]}
                )
            pqscs.append(pqsc_entry)
        if len(pqscs) > 0:
            connectivity_object["pqscs"] = pqscs

        if len(connectivity_object.keys()) > 0:
            retval["connectivity"] = connectivity_object
        oscillator_infos = [
            experiment_dao.oscillator_info(oscillator_id)
            for oscillator_id in experiment_dao.oscillators()
        ]
        out_oscillators = []
        for oscillator_info in oscillator_infos:
            frequency = oscillator_info["frequency"]
            if oscillator_info.get("frequency_param") is not None:
                frequency = {"$ref": oscillator_info.get("frequency_param")}
            out_oscillator_entry = {
                "id": oscillator_info["id"],
                "frequency": frequency,
                "hardware": oscillator_info["hardware"],
            }
            out_oscillators.append(out_oscillator_entry)
        if len(out_oscillators) > 0:
            retval["oscillators"] = out_oscillators

        retval["signals"] = []
        signal_infos = [
            experiment_dao.signal_info(signal_id)
            for signal_id in experiment_dao.signals()
        ]

        signal_connections = []
        for signal_info in signal_infos:
            signal_entry = {
                "id": signal_info["signal_id"],
                "signal_type": signal_info["signal_type"],
            }
            if signal_info["modulation"]:
                signal_entry["modulation"] = signal_info["modulation"]
            if signal_info.get("offset") is not None:
                signal_entry["offset"] = signal_info.get("offset")
            signal_oscillator = experiment_dao.signal_oscillator(
                signal_info["signal_id"]
            )
            if signal_oscillator is not None:
                signal_entry["oscillators_list"] = [{"$ref": signal_oscillator["id"]}]
            retval["signals"].append(signal_entry)

            device_id = experiment_dao.device_from_signal(signal_info["signal_id"])
            signal_connection = {
                "signal": {"$ref": signal_info["signal_id"]},
                "device": {"$ref": device_id},
                "connection": {
                    "type": signal_info["connection_type"],
                    "channels": signal_info["channels"],
                },
            }

            mixer_calibration = experiment_dao.mixer_calibration(
                signal_info["signal_id"]
            )
            if mixer_calibration is not None:
                mixer_calibration_object = {}
                if mixer_calibration.get("voltage_offsets") is not None:
                    mixer_calibration_object["voltage_offsets"] = mixer_calibration[
                        "voltage_offsets"
                    ]
                if mixer_calibration.get("correction_matrix") is not None:
                    mixer_calibration_object["correction_matrix"] = mixer_calibration[
                        "correction_matrix"
                    ]
                if len(mixer_calibration_object.keys()) > 0:
                    signal_connection["mixer_calibration"] = mixer_calibration_object

            lo_frequency = experiment_dao.lo_frequency(signal_info["signal_id"])
            if lo_frequency is not None:
                signal_connection["lo_frequency"] = lo_frequency

            port_mode = experiment_dao.port_mode(signal_info["signal_id"])
            if port_mode is not None:
                signal_connection["port_mode"] = port_mode

            signal_range = experiment_dao.signal_range(signal_info["signal_id"])
            if signal_range is not None:
                signal_connection["range"] = signal_range

            port_delay = experiment_dao.port_delay(signal_info["signal_id"])
            if port_delay is not None:
                signal_connection["port_delay"] = port_delay

            threshold = experiment_dao.threshold(signal_info["signal_id"])
            if threshold is not None:
                signal_connection["threshold"] = threshold

            delay_signal = signal_info["delay_signal"]
            if delay_signal is not None:
                signal_connection["delay_signal"] = delay_signal

            signal_connections.append(signal_connection)

        retval["signal_connections"] = signal_connections

        pulses_list = []

        for pulse in experiment_dao.pulses():
            pulse_entry = {"id": pulse["id"]}
            fields = [
                "function",
                "length",
                "samples",
                "amplitude",
                "play_mode",
            ]
            for field in fields:
                if pulse.get(field) is not None:
                    pulse_entry[field] = pulse[field]

            if pulse_entry.get("amplitude_param"):
                pulse_entry["amplitude"] = {"$ref": pulse_entry["amplitude_param"]}
            if pulse_entry.get("length_param"):
                pulse_entry["length"] = {"$ref": pulse_entry["length_param"]}

            pulses_list.append(pulse_entry)
        retval["pulses"] = pulses_list

        sections_list = []
        for section_id in experiment_dao.sections():
            section_info = experiment_dao.section_info(section_id)
            out_section = {"id": section_info["section_id"]}

            direct_children = experiment_dao.direct_section_children(section_id)
            if section_info["has_repeat"]:
                out_section["repeat"] = {
                    "execution_type": section_info["execution_type"],
                    "count": section_info["count"],
                }
                if section_info["averaging_type"] is not None:
                    out_section["repeat"]["averaging_type"] = section_info[
                        "averaging_type"
                    ]

                section_parameters = experiment_dao.section_parameters(section_id)
                if len(section_parameters) > 0:
                    out_section["repeat"]["parameters"] = []
                    for parameter in section_parameters:
                        param_object = {"id": parameter["id"]}
                        keys = ["start", "step", "values"]
                        for key in keys:
                            if parameter.get(key) is not None:
                                param_object[key] = parameter[key]

                        out_section["repeat"]["parameters"].append(param_object)

            if len(direct_children) > 0:
                if section_info["has_repeat"]:
                    out_section["repeat"]["sections_list"] = [
                        {"$ref": child} for child in direct_children
                    ]
                else:
                    out_section["sections_list"] = [
                        {"$ref": child} for child in direct_children
                    ]
            keys = [
                "align",
                "length",
                "trigger",
                "repetition_mode",
                "repetition_time",
                "averaging_mode",
                "play_after",
            ]
            for key in keys:
                if section_info.get(key) is not None:
                    out_section[key] = section_info[key]
            if section_info.get("reset_oscillator_phase", False):
                out_section["reset_oscillator_phase"] = section_info[
                    "reset_oscillator_phase"
                ]

            signals_list = []
            for signal_id in experiment_dao._section_signals_list(section_id):
                section_signal_object = {"signal": {"$ref": signal_id}}
                section_signal_pulses = []
                for section_pulse in experiment_dao._section_pulses_raw(
                    section_id, signal_id
                ):
                    section_signal_pulse_object = {}
                    if section_pulse["pulse_id"] is not None:
                        section_signal_pulse_object["pulse"] = {
                            "$ref": section_pulse["pulse_id"]
                        }
                    for key in [
                        "amplitude",
                        "offset",
                        "increment_oscillator_phase",
                        "phase",
                        "set_oscillator_phase",
                        "length",
                    ]:
                        if section_pulse.get(key) is not None:
                            section_signal_pulse_object[key] = section_pulse[key]
                        if section_pulse.get(key + "_param") is not None:
                            section_signal_pulse_object[key] = {
                                "$ref": section_pulse[key + "_param"]
                            }

                    section_signal_pulses.append(section_signal_pulse_object)

                if len(section_signal_pulses) > 0:
                    section_signal_object["pulses_list"] = section_signal_pulses
                signals_list.append(section_signal_object)
            if len(signals_list) > 0:
                out_section["signals_list"] = signals_list

            sections_list.append(out_section)

        retval["sections"] = list(sorted(sections_list, key=lambda x: x["id"]))

        retval["experiment"] = {
            "sections_list": [
                {"$ref": section} for section in experiment_dao.root_sections()
            ],
            "signals_list": [
                {"$ref": signal_id} for signal_id in experiment_dao.signals()
            ],
        }
        return retval

    def _load_from_core(self, device_setup, experiment):

        global_leader_device_id = None
        self._data["server"] = {}
        self._data["device"] = {}
        self._data["oscillator"] = {}
        self._data["device_oscillator"] = []
        self._data["signal"] = {}
        self._data["signal_oscillator"] = []
        self._data["section_signal_pulse"] = []
        self._data["section_parameter"] = []
        self._data["pulse"] = {}

        for server in device_setup.servers.values():
            if hasattr(server, "leader_uid"):
                global_leader_device_id = server.leader_uid
            self._data["server"][server.uid] = {
                k: getattr(server, k) for k in ["host", "port", "api_level"]
            }
            if self._data["server"][server.uid]["port"] is not None:
                self._data["server"][server.uid]["port"] = int(
                    self._data["server"][server.uid]["port"]
                )
            self._data["server"][server.uid]["id"] = server.uid

        dest_path_devices = {}

        reference_clock = None
        for device in device_setup.instruments:
            if hasattr(device, "reference_clock"):
                reference_clock = device.reference_clock

        multiplexed_signals = {}
        for device in sorted(device_setup.instruments, key=lambda x: x.uid):

            server = device.server_uid

            driver = type(device).__name__.lower()
            serial = device.address
            interface = device.interface
            is_global_leader = 0
            if global_leader_device_id == device.uid:
                is_global_leader = 1
            reference_clock_source = getattr(device, "reference_clock_source", None)

            self._data["device"][device.uid] = {
                "id": device.uid,
                "device_type": driver,
                "serial": serial,
                "server": server,
                "interface": interface,
                "is_global_leader": is_global_leader,
                "reference_clock": reference_clock,
                "reference_clock_source": None
                if reference_clock_source is None
                else reference_clock_source.value,
            }

            for connection in device.connections:
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

                if multiplex_key not in multiplexed_signals:
                    multiplexed_signals[multiplex_key] = []
                multiplexed_signals[multiplex_key].append(connection.remote_path)

        ls_map = {}
        modulated_paths = {}
        ls_mixer_calibrations = {}
        ls_lo_frequencies = {}
        ls_ranges = {}
        ls_port_delays = {}
        ls_delays_signal = {}
        ls_port_modes = {}
        ls_thresholds = {}

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
            for field in PHYSICAL_CHANNEL_CALIBRATION_FIELDS:
                if field == "mixer_calibration":
                    continue
                values = set()
                for exp_signal in exp_signals:
                    if not exp_signal.is_calibrated():
                        continue
                    value = getattr(exp_signal, field)
                    if value is not None:
                        values.add(value)
                if len(values) > 1:
                    conflicting_signals = [
                        exp_signal.uid
                        for exp_signal in exp_signals
                        if exp_signal.is_calibrated()
                        and getattr(exp_signal.calibration, field) is not None
                    ]
                    raise LabOneQException(
                        f"The experiment signals {', '.join(conflicting_signals)} all "
                        f"touch physical channel '{pc.uid}', but provide conflicting "
                        f"settings for calibration field '{field}'."
                    )
                if len(values) > 0:
                    # Make sure all the experiment signals agree.
                    value = values.pop()
                    for exp_signal in exp_signals:
                        if exp_signal.is_calibrated():
                            setattr(exp_signal.calibration, field, value)

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
                        except ValueError:
                            pass
                        except TypeError:
                            pass
                        if not isinstance(frequency, float) and not isinstance(
                            frequency, int
                        ):
                            if frequency is not None and hasattr(frequency, "uid"):
                                frequency_param = frequency.uid

                            frequency = None
                        modulated_paths[ls.path] = {
                            "oscillator_id": oscillator_uid,
                            "is_hardware": is_hardware,
                        }
                        known_oscillator = self.oscillator_info(oscillator_uid)
                        if known_oscillator is None:
                            self._data["oscillator"][oscillator_uid] = {
                                "id": oscillator_uid,
                                "frequency": frequency,
                                "frequency_param": frequency_param,
                                "hardware": is_hardware,
                            }

                            if is_hardware:
                                device_id = dest_path_devices[ls.path]["device"]
                                self._data["device_oscillator"].append(
                                    {
                                        "device_id": device_id,
                                        "oscillator_id": oscillator_uid,
                                    }
                                )
                        else:
                            if (
                                known_oscillator["frequency"],
                                known_oscillator["frequency_param"],
                                known_oscillator["hardware"],
                            ) != (frequency, frequency_param, is_hardware):
                                raise Exception(
                                    f"Duplicate oscillator uid {oscillator_uid} found in {ls.path}"
                                )
                if (
                    hasattr(calibration, "mixer_calibration")
                    and hasattr(calibration.mixer_calibration, "voltage_offsets")
                    and hasattr(calibration.mixer_calibration, "correction_matrix")
                ):
                    ls_mixer_calibrations[ls.path] = {
                        "voltage_offsets": calibration.mixer_calibration.voltage_offsets,
                        "correction_matrix": calibration.mixer_calibration.correction_matrix,
                    }

                ls_local_oscillator = getattr(calibration, "local_oscillator")
                if ls_local_oscillator is not None:
                    ls_lo_frequencies[ls.path] = getattr(
                        ls_local_oscillator, "frequency"
                    )
                ls_ranges[ls.path] = getattr(calibration, "range")

                if (
                    hasattr(calibration, "port_mode")
                    and calibration.port_mode is not None
                ):
                    ls_port_modes[ls.path] = calibration.port_mode.value

                ls_thresholds[ls.path] = getattr(calibration, "threshold", None)

        for seq_nr, signal in enumerate(
            sorted(experiment.signals.values(), key=lambda x: x.uid)
        ):
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

            self._data["signal"][signal.uid] = {
                "signal_id": signal.uid,
                "signal_type": signal_type,
                "modulation": signal.mapped_logical_signal_path in modulated_paths,
                "seq_nr": seq_nr,
            }

            if signal.mapped_logical_signal_path in modulated_paths:
                oscillator_id = modulated_paths[signal.mapped_logical_signal_path][
                    "oscillator_id"
                ]
                self._data["signal_oscillator"].append(
                    {"signal_id": signal.uid, "oscillator_id": oscillator_id}
                )

        self._data["signal_connection"] = []
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

            self._data["signal_connection"].append(
                {
                    "signal_id": signal,
                    "device_id": dest_path_devices[lsuid]["device"],
                    "connection_type": dest_path_devices[lsuid]["type"],
                    "channels": channels,
                    "mixer_calibration": ls_mixer_calibrations.get(lsuid),
                    "lo_frequency": ls_lo_frequencies.get(lsuid),
                    "range": ls_ranges.get(lsuid),
                    "port_delay": ls_port_delays.get(lsuid),
                    "delay_signal": ls_delays_signal.get(lsuid),
                    "port_mode": ls_port_modes.get(lsuid),
                    "threshold": ls_thresholds.get(lsuid),
                }
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
        from laboneq.core.types.enums import IOSignalType

        self._data["pqsc_port"] = []
        self._data["dio"] = []
        self._data["section_tree"] = []
        self._data["section_signal"] = []
        self._data["section"] = {}

        for syncing_connection in syncing_connections:
            signal_type = syncing_connection[2]
            assert type(syncing_connection[2]) == type(IOSignalType.DIO)
            if signal_type == IOSignalType.DIO:
                dio_leader = syncing_connection[0]
                dio_follower = syncing_connection[1]
                self._data["dio"].append((dio_leader, dio_follower))

            elif signal_type == IOSignalType.ZSYNC:
                zsync_leader = syncing_connection[0]
                zsync_follower = syncing_connection[1]
                port = syncing_connection[4].physical_port_ids[0]
                self._data["pqsc_port"].append(
                    (zsync_leader, zsync_follower, int(port))
                )

        seq_avg_section, sweep_sections = self.find_sequential_averaging(experiment)
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

        section_signal_id = {"current": 0}
        section_uid_map = {}
        acquisition_type_map = {}
        for section in experiment.sections:
            self.process_section(
                section, None, section_uid_map, acquisition_type_map, exchanger_map
            )

        pulse_uids = set()
        for section in section_uid_map.values():
            self.insert_section(
                experiment,
                section,
                section_signal_id,
                pulse_uids,
                acquisition_type_map[section.uid],
                exchanger_map,
            )

        self._acquisition_type = AcquisitionType(
            next(
                (at for at in acquisition_type_map.values() if at is not None),
                AcquisitionType.INTEGRATION,
            )
        )

    def find_sequential_averaging(self, section) -> Tuple[Any, Tuple]:
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

            child_avg_section, child_sweep_sections = self.find_sequential_averaging(
                child_section
            )
            if avg_section is not None and child_avg_section is not None:
                raise LabOneQException(
                    "Illegal nesting of sequential averaging loops detected."
                )
            sweep_sections = (*sweep_sections, *child_sweep_sections)

        return avg_section, sweep_sections

    def process_section(
        self,
        section,
        acquisition_type,
        section_uid_map,
        acquisition_type_map,
        exchanger_map,
    ):
        if section.uid is None:
            raise RuntimeError(f"Section uid must not be None: {section}")
        section_uid_map[section.uid] = section
        current_acquisition_type = acquisition_type

        if hasattr(section, "acquisition_type"):
            current_acquisition_type = section.acquisition_type

        acquisition_type_map[section.uid] = current_acquisition_type

        for index, child_section in enumerate(section.sections):
            self._data["section_tree"].append(
                {
                    "parent_section_id": exchanger_map(section).uid,
                    "child_section_id": exchanger_map(child_section).uid,
                    "index": index,
                }
            )

            self.process_section(
                child_section,
                current_acquisition_type,
                section_uid_map,
                acquisition_type_map,
                exchanger_map,
            )

    def insert_section(
        self,
        experiment,
        section,
        section_signal_id,
        pulse_uids,
        acquisition_type,
        exchanger_map,
    ):
        has_repeat = 0
        count = 1
        experiment_seq = None
        for index, experiment_section in enumerate(
            exchanger_map(s) for s in experiment.sections
        ):
            if experiment_section.uid == section.uid:
                experiment_seq = index
                break

        averaging_type = None
        if hasattr(section, "count"):
            has_repeat = 1
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
                has_repeat = 1
                self._data["section_parameter"].append(
                    {
                        "section_id": section.uid,
                        "id": parameter.uid,
                        "start": None,
                        "step": None,
                        "values": values_list,
                        "axis_name": axis_name,
                    }
                )
                if hasattr(parameter, "count"):
                    count = parameter.count
                elif hasattr(parameter, "values"):
                    count = len(parameter.values)

        execution_type = None
        if hasattr(section, "execution_type") and section.execution_type is not None:
            execution_type = section.execution_type.value

        align = "left"
        if exchanger_map(section).alignment is not None:
            align = exchanger_map(section).alignment.value

        offset = None
        offset_param = None
        if section.offset is not None:
            offset = section.offset
            if not isinstance(offset, float) and not isinstance(offset, int):
                offset_param = offset.uid
                offset = None

        length = None
        if hasattr(section, "length") and section.length is not None:
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

        reset_oscillator_phase = None
        if hasattr(section, "reset_oscillator_phase"):
            reset_oscillator_phase = section.reset_oscillator_phase

        trigger = None
        for operation in exchanger_map(section).operations:
            if hasattr(operation, "handle"):
                # an acquire event - add trigger
                trigger = [acquisition_type.value]

        play_after = None
        if hasattr(section, "play_after"):
            play_after = section.play_after
        play_after = play_after

        self._data["section"][section.uid] = dict(
            section_id=section.uid,
            experiment_seq=experiment_seq,
            has_repeat=has_repeat,
            execution_type=execution_type,
            count=count,
            trigger=trigger,
            averaging_type=averaging_type,
            align=align,
            offset=offset,
            offset_param=offset_param,
            length=length,
            averaging_mode=averaging_mode,
            repetition_mode=repetition_mode,
            repetition_time=repetition_time,
            play_after=play_after,
            reset_oscillator_phase=reset_oscillator_phase,
        )

        section_signals = {}
        for operation in exchanger_map(section).operations:
            if not hasattr(operation, "signal"):
                continue
            if hasattr(operation, "signal"):
                section_signals[operation.signal] = {
                    "section_uid": section.uid,
                    "signal_uid": operation.signal,
                }
        for signal_uid, section_signal in section_signals.items():

            section_signal["section_signal_id"] = section_signal_id["current"]
            self._data["section_signal"].append(
                {
                    "id": section_signal_id["current"],
                    "section_id": section_signal["section_uid"],
                    "signal_id": section_signal["signal_uid"],
                }
            )
            section_signal_id["current"] += 1

        seq_nr = 0
        for operation in exchanger_map(section).operations:
            if hasattr(operation, "signal"):
                section_signal = section_signals[operation.signal]
                pulse_offset = None
                pulse_offset_param = None

                if hasattr(operation, "time"):

                    pulse_offset = operation.time
                    if not isinstance(operation.time, float) and not isinstance(
                        operation.time, int
                    ):
                        pulse_offset = None
                        pulse_offset_param = operation.time.uid

                    self._data["section_signal_pulse"].append(
                        dict(
                            section_signal_id=section_signal["section_signal_id"],
                            section_id=section.uid,
                            signal_id=operation.signal,
                            pulse_id=None,
                            offset=pulse_offset,
                            offset_param=pulse_offset_param,
                            amplitude=None,
                            amplitude_param=None,
                            seq_nr=seq_nr,
                            acquire_params=None,
                            phase=None,
                            phase_param=None,
                            increment_oscillator_phase=None,
                            increment_oscillator_phase_param=None,
                            set_oscillator_phase=None,
                            set_oscillator_phase_param=None,
                        )
                    )
                    seq_nr += 1

                else:
                    pulse = None
                    operation_length = None
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

                        if pulse.uid not in pulse_uids:
                            samples = None
                            if hasattr(pulse, "function"):
                                function = pulse.function.value
                            if hasattr(pulse, "length"):
                                length = pulse.length

                            if hasattr(pulse, "samples"):
                                samples = pulse.samples

                            amplitude, amplitude_param = find_value_or_parameter_attr(
                                pulse, "amplitude", (float, int, complex)
                            )

                            self._data["pulse"][pulse.uid] = {
                                "id": pulse.uid,
                                "function": function,
                                "length": length,
                                "amplitude": amplitude,
                                "amplitude_param": amplitude_param,
                                "play_mode": None,
                                "samples": samples,
                            }

                            pulse_uids.add(pulse.uid)

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
                            acquire_params = {
                                "handle": operation.handle,
                                "acquisition_type": acquisition_type.name,
                            }

                        self._data["section_signal_pulse"].append(
                            dict(
                                section_signal_id=section_signal["section_signal_id"],
                                section_id=section.uid,
                                signal_id=operation.signal,
                                pulse_id=pulse.uid,
                                offset=pulse_offset,
                                offset_param=pulse_offset_param,
                                amplitude=pulse_amplitude,
                                amplitude_param=pulse_amplitude_param,
                                length=operation_length,
                                length_param=operation_length_param,
                                seq_nr=seq_nr,
                                acquire_params=acquire_params,
                                phase=pulse_phase,
                                phase_param=pulse_phase_param,
                                increment_oscillator_phase=pulse_increment_oscillator_phase,
                                increment_oscillator_phase_param=pulse_increment_oscillator_phase_param,
                                set_oscillator_phase=pulse_set_oscillator_phase,
                                set_oscillator_phase_param=pulse_set_oscillator_phase_param,
                            )
                        )

                        seq_nr += 1

    def validate_experiment(self):
        all_parameters = set()
        for section_id in self.sections():
            for parameter in self.section_parameters(section_id):
                all_parameters.add(parameter["id"])

        for section_id in self.sections():
            for signal_id in self.section_signals(section_id):
                for section_pulse in self.section_pulses(section_id, signal_id):
                    pulse_id = section_pulse.get("pulse_id")
                    if pulse_id is not None and self.pulse(pulse_id) is None:
                        raise RuntimeError(
                            f"Pulse {pulse_id} referenced in section {section_id} by a pulse on signal {signal_id} is not known."
                        )

                    for k in ["length_param", "amplitude_param", "offset_param"]:
                        param_name = section_pulse.get(k)
                        if param_name is not None:
                            if param_name not in all_parameters:
                                raise RuntimeError(
                                    f"Parameter {param_name} referenced in section {section_id} by a pulse on signal {signal_id} is not known."
                                )


class AttributeOverrider(object):
    def __init__(self, base, overrider):
        if base is None:
            raise RuntimeError("base must not be none")
        if overrider is None:
            raise RuntimeError("overrider must not be none")

        self._overrider = overrider
        self._base = base

    def __getattr__(self, attr):
        if hasattr(self._overrider, attr):
            overrider_value = getattr(self._overrider, attr)
            if overrider_value is not None:
                return overrider_value
        if hasattr(self._base, attr):
            return getattr(self._base, attr)
        raise AttributeError(
            f"Field {attr} not found on overrider {self._overrider} (type {type(self._overrider)}) nor on base {self._base}"
        )
