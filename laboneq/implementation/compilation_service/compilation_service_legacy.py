# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import logging
import time
import uuid

from laboneq.compiler import Compiler
from laboneq.core.types.compiled_experiment import (
    CompiledExperiment as CompiledExperimentDSL,
)
from laboneq.data.compilation_job import CompilationJob, SignalInfoType
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.interfaces.compilation_service import CompilationServiceAPI

_logger = logging.getLogger(__name__)


class CompilationServiceLegacy(CompilationServiceAPI):
    def __init__(self):
        self._job_queue = []
        self._job_results = {}

    def submit_compilation_job(self, job: CompilationJob):
        """
        Submit a compilation job.
        """
        job_id = len(self._job_queue)
        queue_entry = {"job_id": job_id, "job": job}

        experiment_json = convert_to_experiment_json(job)
        compiler = Compiler()
        compiler_output = compiler.run(experiment_json)

        self._job_results[job_id] = convert_compiler_output_to_scheduled_experiment(
            compiler_output
        )

        self._job_queue.append(queue_entry)
        return job_id

    def compilation_job_status(self, job_id: str):
        """
        Get the status of a compilation job.
        """
        return next(j for j in self._job_queue if j["job_id"] == job_id)

    def compilation_job_result(self, job_id: str) -> ScheduledExperiment:
        """
        Get the result of a compilation job. Blocks until the result is available.
        """
        num_tries = 10
        while True:
            result = self._job_results.get(job_id)
            if result:
                return result
            if num_tries == 0:
                break
            num_tries -= 1
            time.sleep(100e-3)


def convert_to_experiment_json(job: CompilationJob):
    retval = {
        "$schema": "../../schemas/qccs-schema_2_5_0.json",
        "metadata": {
            "version": "2.5.0",
            "unit": {"time": "s", "frequency": "Hz", "phase": "rad"},
            "epsilon": {"time": 1e-12},
            "line_endings": "unix",
        },
        "servers": [
            {
                "id": "zi_server",
                "host": "0.0.0.0",
                "port": 8004,
                "api_level": 6,
            }
        ],
    }

    devices_in_job = {}
    oscillators_in_job = {}
    device_oscillators = {}
    signal_connections = []

    for signal in job.experiment_info.signals:
        devices_in_job[signal.device.uid] = signal.device
        _logger.info(f"Added device {signal.device} to job")
        connection_dir = "in" if signal.type == SignalInfoType.INTEGRATION else "out"
        signal_connections.append(
            {
                "signal": {"$ref": signal.uid},
                "device": {"$ref": signal.device.uid},
                "connection": {
                    "type": connection_dir,
                    "channels": [int(c) for c in signal.channels],
                },
            }
        )
        for osc in signal.oscillators:
            oscillators_in_job[osc.uid] = osc
            if osc.is_hardware:
                device_oscillators.setdefault(signal.device.uid, []).append(osc)

    retval["devices"] = [
        {
            "id": d.uid,
            "server": {"$ref": "zi_server"},
            "serial": "DEV" + str(i),
            "interface": "1GbE",
            "driver": d.device_type.name.lower(),
        }
        for i, d in enumerate(devices_in_job.values())
    ]
    for d in retval["devices"]:
        if d["id"] in device_oscillators:
            d["oscillators_list"] = [
                {"$ref": o.uid} for o in device_oscillators[d["id"]]
            ]
    retval["oscillators"] = [
        {"id": o.uid, "frequency": o.frequency, "hardware": o.is_hardware}
        for o in oscillators_in_job.values()
    ]

    signal_type_mapping = {
        SignalInfoType.INTEGRATION: "integration",
        SignalInfoType.RF: "single",
        SignalInfoType.IQ: "iq",
    }

    retval["signals"] = [
        {
            "id": s.uid,
            "signal_type": signal_type_mapping[s.type],
            "oscillators_list": [{"$ref": o.uid} for o in s.oscillators],
        }
        for s in job.experiment_info.signals
    ]
    for s in retval["signals"]:
        if s["oscillators_list"] == []:
            del s["oscillators_list"]
        else:
            s["modulation"] = True

    retval["signal_connections"] = signal_connections

    retval["pulses"] = [
        not_none_fields_dict(
            p, ["uid", "length", "amplitude", "phase", "function"], {"uid": "id"}
        )
        for p in job.experiment_info.pulse_defs
    ]

    def walk_sections(section, visitor):
        visitor(section)
        for s in section.children:
            walk_sections(s, visitor)

    sections_flat = []

    def collector(section):
        sections_flat.append(section)

    for s in job.experiment_info.sections:
        walk_sections(s, collector)

    retval["sections"] = []
    for s in sections_flat:
        out_section = {
            "id": s.uid,
            "align": s.alignment.name.lower() if s.alignment else "left",
        }
        if s.count is not None:
            out_section["repeat"] = {
                "count": s.count,
                "sections_list": [{"$ref": c.uid} for c in s.children],
                "execution_type": s.execution_type,
                "averaging_type": s.averaging_type,
            }
        else:
            out_section["sections_list"] = [{"$ref": c.uid} for c in s.children]
        if out_section["sections_list"] == []:
            del out_section["sections_list"]
        retval["sections"].append(out_section)

    for ssp in job.experiment_info.section_signal_pulses:
        for s in retval["sections"]:
            if s["id"] == ssp.section.uid:
                if "signals_list" not in s:
                    s["signals_list"] = []
                signals_list_entry = next(
                    (
                        sle
                        for sle in s["signals_list"]
                        if sle["signal"]["$ref"] == ssp.signal.uid
                    ),
                    None,
                )
                if signals_list_entry is None:
                    signals_list_entry = {
                        "signal": {"$ref": ssp.signal.uid},
                        "pulses_list": [],
                    }
                    s["signals_list"].append(signals_list_entry)

                signals_list_entry["pulses_list"].append(
                    {"pulse": {"$ref": ssp.pulse_def.uid}}
                )
                break

    retval["experiment"] = {
        "sections_list": [{"$ref": s.uid} for s in job.experiment_info.sections],
        "signals_list": [{"$ref": s.uid} for s in job.experiment_info.signals],
    }

    _logger.info(f"Generated job: {json.dumps(retval, indent=2)}")

    return retval


def not_none_fields_dict(obj, fields, translator):
    return {
        translator.get(f, f): getattr(obj, f)
        for f in fields
        if getattr(obj, f) is not None
    }


def convert_compiler_output_to_scheduled_experiment(
    compiler_output: CompiledExperimentDSL,
) -> ScheduledExperiment:
    recipe = copy.deepcopy(compiler_output.recipe)
    src = copy.deepcopy(compiler_output.src)
    waves = copy.deepcopy(compiler_output.waves)
    wave_indices = copy.deepcopy(compiler_output.wave_indices)
    command_tables = copy.deepcopy(compiler_output.command_tables)
    pulse_map = copy.deepcopy(compiler_output.pulse_map)

    return ScheduledExperiment(
        uid=uuid.uuid4().hex,
        recipe=recipe,
        src=src,
        waves=waves,
        wave_indices=wave_indices,
        command_tables=command_tables,
        pulse_map=pulse_map,
    )
