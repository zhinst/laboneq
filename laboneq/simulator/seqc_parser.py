# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from types import SimpleNamespace

# Note: The simulator may be used as a testing tool, so it must be independent of the production code
# Do not add dependencies to the code being tested here (such as compiler, DSL asf.)
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import typing as npt
from pycparser.c_ast import (
    ID,
    Assignment,
    BinaryOp,
    Constant,
    Decl,
    DoWhile,
    FuncCall,
    FuncDef,
    If,
    Node,
    UnaryOp,
)
from pycparser.c_parser import CParser

if TYPE_CHECKING:
    from laboneq.core.types.compiled_experiment import CompiledExperiment


def precompensation_is_nonzero(precompensation):
    """Check whether the precompensation has any effect"""
    return precompensation is not None and (
        precompensation.get("exponential") is not None
        and any([e["amplitude"] != 0 for e in precompensation["exponential"]])
        or precompensation.get("high_pass") is not None
        or precompensation.get("bounce") is not None
        and precompensation["bounce"]["amplitude"] != 0
        or precompensation.get("FIR") is not None
        and any(c != 0 for c in precompensation["FIR"]["coefficients"])
    )


def precompensation_delay_samples(precompensation):
    """Compute the additional delay (in samples) caused by the precompensation"""
    if not precompensation_is_nonzero(precompensation):
        return 0
    delay = 72
    try:
        delay += 88 * len(precompensation["exponential"])
    except KeyError:
        pass
    if precompensation.get("high_pass") is not None:
        delay += 96
    if precompensation.get("bounce") is not None:
        delay += 32
    if precompensation.get("FIR") is not None:
        delay += 136
    return delay


def get_frequency(device_type):
    if device_type == "HDAWG":
        return 2.4e9
    elif device_type == "UHFQA":
        return 1.8e9
    elif device_type == "SHFQA":
        return 2e9
    elif device_type == "SHFSG":
        return 2e9
    elif device_type == "PQSC":
        return 0.0
    else:
        raise RuntimeError(f"Unsupported device type {device_type}")


def get_sample_multiple(device_type):
    if device_type in ["HDAWG", "SHFQA", "SHFSG"]:
        return 16
    elif device_type == "UHFQA":
        return 8
    else:
        return 16


def parse_seq_c(runtime: SimpleRuntime):
    if len(runtime.source) == 0:
        return

    parser = CParser()
    ast = parser.parse(runtime.source, runtime.descriptor.name)
    # ast_stream = StringIO()
    # ast.show(buf=ast_stream)
    # print(ast_stream.getvalue())

    item: Node
    for ext in ast.ext:
        if isinstance(ext, FuncDef) and ext.decl.name.startswith("set_"):
            parse_set_func(ext.decl.name[4:], ext.body.block_items[1], runtime)

    for item in ast.ext[-1].body.block_items:
        try:
            parse_item(item, runtime)
        except StopSimulation:
            break

    runtime.program_finished()


def parse_expression(item, runtime: SimpleRuntime):
    if isinstance(item, Constant):
        return parse_constant(item)
    if isinstance(item, ID):
        return runtime.resolve(item.name)
    if isinstance(item, BinaryOp):
        if item.op == "|":
            return parse_expression(item.left, runtime) | parse_expression(
                item.right, runtime
            )
        if item.op == "&":
            return parse_expression(item.left, runtime) & parse_expression(
                item.right, runtime
            )
        if item.op == "^":
            return parse_expression(item.left, runtime) ^ parse_expression(
                item.right, runtime
            )
    elif isinstance(item, UnaryOp):
        if item.op == "-":
            return -parse_expression(item.expr, runtime)
        if item.op == "++p":  # pre-increment
            identifier = item.expr
            if isinstance(identifier, ID) and identifier.name in runtime.variables:
                runtime.variables[identifier.name]["value"] += 1
                return runtime.variables[identifier.name]["value"]
            raise RuntimeError(f"Invalid lvalue in increment expression {identifier}")
        if item.op == "p++":  # post-increment
            identifier = item.expr
            if isinstance(identifier, ID) and identifier.name in runtime.variables:
                val = runtime.variables[identifier.name]["value"]
                runtime.variables[identifier.name]["value"] += 1
                return val
            raise RuntimeError(f"Invalid lvalue in increment expression {identifier}")


def parse_int_literal(s: str):
    try:
        l = len(s)
        while l > 1 and s[l - 1] in "uUlL":
            l -= 1
        return (
            int(s[:l], 8)
            if l > 1 and s[0] == "0" and s[1] in "01234567"
            else int(s[:l], 0)
        )
    except ValueError:
        raise ValueError(f"Invalid int literal: '{s}'")


@lru_cache(maxsize=None)
def parse_constant(constant: Constant):
    if constant.type == "int":
        r = parse_int_literal(constant.value)
    elif constant.type in ["float", "double"]:
        r = float(constant.value)
    else:
        r = constant.value
    return r


class StopSimulation(Exception):
    pass


def parse_set_func(param_name: str, stmt: Node, runtime: SimpleRuntime):
    def parse_step(lower_bound: int, stmt: Node):
        if isinstance(stmt, If):
            upper_bound = parse_int_literal(stmt.cond.right.value) + lower_bound
            parse_step(upper_bound, stmt.iftrue.block_items[0])
            parse_step(lower_bound, stmt.iffalse.block_items[0])
        else:
            value = parse_expression(stmt.args.exprs[1], runtime)
            runtime.setParamVal(param_name, lower_bound, value)

    parse_step(0, stmt)


def parse_item(item: Node, runtime: SimpleRuntime):
    if isinstance(item, Decl):
        runtime.declare(item.name)
        init_value = None
        if isinstance(item.init, Constant):
            init_value = parse_constant(item.init)
        elif isinstance(item.init, FuncCall):
            init_value = f'"{item.name[1:]}"'
        runtime.initialize(item.name, init_value)
    elif isinstance(item, FuncCall):
        func_name: str = item.name.name
        args = (
            tuple(parse_expression(arg, runtime) for arg in item.args)
            if item.args is not None
            else ()
        )
        if func_name in runtime.exposedFunctions:
            runtime.exposedFunctions[func_name](*args)
        elif func_name.startswith("set_"):
            runtime.setOscFreqByParam(func_name[4:], *args)
        else:
            pass  # Skipping unknown function

    elif isinstance(item, Assignment):
        variables = runtime.variables
        if item.op == "-=":
            variables[item.lvalue.name]["value"] = variables[item.lvalue.name][
                "value"
            ] - parse_expression(item.rvalue, runtime)
        if item.op == "+=":
            variables[item.lvalue.name]["value"] = variables[item.lvalue.name][
                "value"
            ] + parse_expression(item.rvalue, runtime)
        elif item.op == "=":
            variables[item.lvalue.name]["value"] = parse_expression(
                item.rvalue, runtime
            )

    elif isinstance(item, DoWhile):
        endless_guard = 10000
        while True:
            for subitem in item.stmt.children():
                parse_item(subitem[1], runtime)
            variable_value = 0
            condition_variable_name = "UNKNOWN"
            try:
                variables = runtime.variables
                condition_variable_name = item.cond.name
                variable_value = int(variables[condition_variable_name]["value"])
            except AttributeError:
                pass

            if variable_value <= 0:
                break
            endless_guard -= 1
            if endless_guard <= 0:
                raise RuntimeError("Endless guard triggered")
    if (
        runtime.max_time is not None
        and runtime._last_play_start_samples()[0] / runtime.descriptor.sampling_rate
        > runtime.max_time
    ):
        # Stop, once a time-span event encountered, that is entirely outside the simulation region.
        # This is important, as any point-in-time events before must be captured. This last time-span
        # event, however, must be dropped.
        runtime.seqc_simulation.events.pop(-1)
        raise StopSimulation


@dataclass
class SeqCDescriptor:
    name: str
    device_uid: str
    device_type: str
    awg_index: int
    measurement_delay_samples: int
    startup_delay: float
    sample_multiple: int
    sampling_rate: float
    output_port_delay: float
    source: str = None
    channels: List[int] = None
    wave_index: Dict[Any, Any] = None
    command_table: List[Any] = None


class Operation(Enum):
    PLAY_ZERO = auto()
    PLAY_WAVE = auto()
    START_QA = auto()
    SET_OSC_FREQ = auto()
    SET_TRIGGER = auto()
    SET_PRECOMP_CLEAR = auto()
    WAIT_WAVE = auto()


@dataclass
class SeqCEvent:
    start_samples: int
    length_samples: int
    operation: Operation
    args: List[Any]


@dataclass
class WaveRefInfo:
    assigned_index: int = -1
    wave_data_idx: List[int] = field(default_factory=list)
    length_samples: int = None


@dataclass
class SeqCSimulation:
    events: List[SeqCEvent] = field(default_factory=list)
    device_type: str = ""
    waves: List[Any] = field(default_factory=list)
    sampling_rate: float = field(default=2.0e9)
    startup_delay: float = field(default=0.0)
    output_port_delay: float = field(default=0.0)


class SimpleRuntime:
    def __init__(
        self,
        descriptor: SeqCDescriptor,
        waves,
        max_time: Optional[float],
    ):
        self.predefined_consts = {
            "QA_INT_0": 0b1,
            "QA_INT_1": 0b10,
            "QA_INT_2": 0b100,
            "QA_INT_3": 0b1000,
            "QA_INT_4": 0b10000,
            "QA_INT_5": 0b100000,
            "QA_INT_6": 0b1000000,
            "QA_INT_7": 0b10000000,
            "QA_INT_8": 0b100000000,
            "QA_INT_9": 0b1000000000,
            "QA_INT_10": 0b10000000000,
            "QA_INT_11": 0b100000000000,
            "QA_INT_12": 0b1000000000000,
            "QA_INT_13": 0b10000000000000,
            "QA_INT_14": 0b100000000000000,
            "QA_INT_15": 0b1000000000000000,
            "QA_GEN_0": 0b1,
            "QA_GEN_1": 0b10,
            "QA_GEN_2": 0b100,
            "QA_GEN_3": 0b1000,
            "QA_GEN_4": 0b10000,
            "QA_GEN_5": 0b100000,
            "QA_GEN_6": 0b1000000,
            "QA_GEN_7": 0b10000000,
            "QA_GEN_8": 0b100000000,
            "QA_GEN_9": 0b1000000000,
            "QA_GEN_10": 0b10000000000,
            "QA_GEN_11": 0b100000000000,
            "QA_GEN_12": 0b1000000000000,
            "QA_GEN_13": 0b10000000000000,
            "QA_GEN_14": 0b100000000000000,
            "QA_GEN_15": 0b1000000000000000,
            "QA_INT_ALL": 0b1111111111111111,
            "QA_INT_NONE": 0,
            "QA_GEN_ALL": 0b1111111111111111,
            "QA_GEN_NONE": 0,
            "QA_DATA_PROCESSED": 0b1000000000100
            if descriptor.device_type == "SHFSG"
            else 0b10000000100,
        }
        self.exposedFunctions = {
            "assignWaveIndex": self.assignWaveIndex,
            "playWave": self.playWave,
            "playZero": self.playZero,
            "executeTableEntry": self.executeTableEntry,
            "startQA": self.startQA,
            "startQAResult": self.startQAResult,
            "configFreqSweep": self.configFreqSweep,
            "setSweepStep": self.setSweepStep,
            "setOscFreq": self.setOscFreq,
            "setTrigger": self.setTrigger,
            "setPrecompClear": self.setPrecompClear,
            "waitWave": self.waitWave,
        }
        self.variables = {}
        self.seqc_simulation = SeqCSimulation()
        self.times = {}
        self.times_at_port = {}
        self.descriptor = descriptor
        self.waves = waves
        self.source = preprocess_source(descriptor.source)
        self.wave_lookup: Dict[Any, WaveRefInfo] = {}
        self.wave_data: List[Any] = []
        self.max_time: Optional[float] = max_time
        self._oscillator_sweep_config = {}
        self._oscillator_sweep_params: Dict[str, Dict[int, float]] = {}

    def _last_played_sample(self) -> int:
        ev = self.seqc_simulation.events
        return ev[-1].start_samples + ev[-1].length_samples if len(ev) > 0 else 0

    def _last_play_start_samples(self) -> Tuple[int, int]:
        # Returns a time in samples and an index where the next point-in-time event
        # has to be positioned relative to the last time-span event. This is to support
        # the SeqC rule of point-in-time events (like startQA) be aligned with the start
        # of the preceding time-span event (like playWave).
        ev = self.seqc_simulation.events
        if len(ev) == 0:
            return 0, 0
        if ev[-1].operation in [
            Operation.PLAY_WAVE,
            Operation.PLAY_ZERO,
            Operation.WAIT_WAVE,
        ]:
            return ev[-1].start_samples, -1
        return ev[-1].start_samples + ev[-1].length_samples, len(ev)

    def program_finished(self):
        self.seqc_simulation.device_type = self.descriptor.device_type
        self.seqc_simulation.waves = self.wave_data
        self.seqc_simulation.sampling_rate = self.descriptor.sampling_rate
        self.seqc_simulation.startup_delay = self.descriptor.startup_delay
        self.seqc_simulation.output_port_delay = self.descriptor.output_port_delay

    def declare(self, name):
        self.variables[name] = {"name": name}

    def initialize(self, name, value):
        self.variables[name]["value"] = value

    def resolve(self, name):
        if name in self.predefined_consts:
            return self.predefined_consts[name]
        return self.variables[name]["value"]

    def _args2key(self, args):
        return tuple(tuple(a.items()) if isinstance(a, dict) else a for a in args)

    def _update_wave_refs(self, wave_names: List[str], known_wave: WaveRefInfo):
        known_length = known_wave.length_samples  # make VSCode's code parser happy
        if known_length is not None:
            return
        for wave_name in wave_names:
            if wave_name is None:
                known_wave.wave_data_idx.append(None)
                continue
            wave_to_play = self.waves[wave_name]
            if np.ndim(wave_to_play) == 1:
                wave_len = len(wave_to_play)
                if np.iscomplexobj(wave_to_play):
                    known_wave.wave_data_idx.append(len(self.wave_data))
                    self.wave_data.append(wave_to_play.real)
                    known_wave.wave_data_idx.append(len(self.wave_data))
                    self.wave_data.append(wave_to_play.imag)
                else:
                    known_wave.wave_data_idx.append(len(self.wave_data))
                    self.wave_data.append(wave_to_play)
            elif np.ndim(wave_to_play) == 2:
                if len(wave_names) > 1:
                    raise Exception(
                        f"Multiple dual-channel waves in single 'playWave': {wave_names}"
                    )
                if np.iscomplexobj(wave_to_play):
                    raise Exception(
                        f"Multiple complex waves in single 'playWave': {wave_names}"
                    )
                # Assuming the waveform can't be 2 samples long, therefore size 2 maps to channels
                if np.shape(wave_to_play)[1] == 2:
                    wave_len = len(wave_to_play[:, 0])
                    known_wave.wave_data_idx.append(len(self.wave_data))
                    self.wave_data.append(wave_to_play[:, 0])
                    known_wave.wave_data_idx.append(len(self.wave_data))
                    self.wave_data.append(wave_to_play[:, 1])
                else:
                    wave_len = len(wave_to_play[0])
                    known_wave.wave_data_idx.append(len(self.wave_data))
                    self.wave_data.append(wave_to_play[0])
                    known_wave.wave_data_idx.append(len(self.wave_data))
                    self.wave_data.append(wave_to_play[1])
            else:
                raise Exception(
                    f"Unexpected waveform '{wave_name}' shape: {np.ndim(wave_to_play)}"
                )

            if wave_len % self.descriptor.sample_multiple != 0:
                raise Exception(
                    f"Wave {wave_name} has {wave_len} samples, which is not divisible by {self.descriptor.sample_multiple}"
                )
            if known_wave.length_samples is None:
                known_wave.length_samples = wave_len
            elif known_wave.length_samples != wave_len:
                raise Exception(
                    f"Inconsistent wave lengths {known_wave.length_samples} != {wave_len} in single 'playWave': {wave_names}"
                )

    def _append_wave_event(self, wave_names: List[str], known_wave: WaveRefInfo):
        self._update_wave_refs(wave_names, known_wave)

        time_samples = self._last_played_sample()
        self.seqc_simulation.events.append(
            SeqCEvent(
                start_samples=time_samples,
                length_samples=known_wave.length_samples,
                operation=Operation.PLAY_WAVE,
                args=[known_wave.wave_data_idx],
            )
        )

    def assignWaveIndex(self, *args):
        idx = args[-1]
        wave_key = self._args2key(args[:-1])
        known_wave = self.wave_lookup.get(wave_key)
        if known_wave is None:
            self.wave_lookup[wave_key] = WaveRefInfo(assigned_index=idx)
        else:
            if known_wave.assigned_index != idx:
                raise Exception(
                    f"Attempt to assign wave index {idx} to args {wave_key} having already index {known_wave.assigned_index}"
                )

    def playWave(self, *args):
        wave_key = self._args2key(args)
        known_wave = self.wave_lookup.get(wave_key)
        if known_wave is None:
            known_wave = WaveRefInfo()
            self.wave_lookup[wave_key] = known_wave

        wave_format = ".csv" if known_wave.assigned_index == -1 else ".wave"

        wave_names = []
        # Supporting only combinations emitted by L1Q compiler, not any possible SeqC
        for arg in args:
            if isinstance(arg, str):
                # handle also instructions like `playWave(1, "", 2, w1);`
                wave_names.append(
                    None if arg == '""' else (arg.strip('"') + wave_format)
                )

        if not wave_names or wave_names == [None]:
            raise RuntimeError(f"Couldn't determine wave name(s) from {args}")

        self._append_wave_event(wave_names, known_wave)

    def playZero(self, length):
        if length > 0:
            time_samples = self._last_played_sample()
            self.seqc_simulation.events.append(
                SeqCEvent(
                    start_samples=time_samples,
                    length_samples=length,
                    operation=Operation.PLAY_ZERO,
                    args=[],
                )
            )

    def executeTableEntry(self, ct_index):
        wave_key = self._args2key(["ct", ct_index])
        known_wave = self.wave_lookup.get(wave_key)
        if known_wave is None:
            known_wave = WaveRefInfo()
            self.wave_lookup[wave_key] = known_wave

        QA_DATA_PROCESSED_SG = 0b1000000000100
        if ct_index == QA_DATA_PROCESSED_SG:
            assert self.descriptor.device_type == "SHFSG"
            # todo(JL): Find a better index via the command table offset; take last for now
            ct_index = self.descriptor.command_table[-1]["index"]

        ct_entry = next(
            iter(i for i in self.descriptor.command_table if i["index"] == ct_index)
        )

        wave_index = ct_entry["waveform"]["index"]

        wave = self.descriptor.wave_index[wave_index]

        if wave["type"] not in ("iq", "multi"):
            raise RuntimeError(
                f"Command table execution in seqc parser only supports iq signals. Device type: {self.descriptor.device_type}, signal type: {wave['type']}"
            )

        wave_names = [wave["wave_name"] + suffix + ".wave" for suffix in ("_i", "_q")]

        self._append_wave_event(wave_names, known_wave)

    def startQA(self, *args):
        if self.descriptor.device_type == "SHFQA":
            self.startQA_SHFQA(*args)
        else:
            self.startQA_UHFQA(*args)

    def startQA_SHFQA(
        self,
        generators_mask=None,
        integrators_mask=None,
        input_monitor=0,
        result_addr=0,
        trigger=0,
    ):
        if generators_mask is None:
            generators_mask = self.resolve("QA_GEN_ALL")
        if integrators_mask is None:
            generators_mask = self.resolve("QA_INT_ALL")

        wave_data_idx = []
        event_length = 0
        for gen_index in range(16):
            if (generators_mask & (1 << gen_index)) != 0:
                wave_key = self._args2key(["gen", gen_index])
                known_wave = self.wave_lookup.get(wave_key)
                if known_wave is None:
                    known_wave = WaveRefInfo()
                    self.wave_lookup[wave_key] = known_wave
                wave = self.descriptor.wave_index[gen_index]
                wave_names = [wave["wave_name"] + ".wave"]
                wave_length_samples = len(self.waves[wave_names[0]])
                event_length = max(event_length, wave_length_samples)
                self._update_wave_refs(wave_names, known_wave)
                wave_data_idx = known_wave.wave_data_idx

        start_samples, insert_at = self._last_play_start_samples()
        self.seqc_simulation.events.insert(
            insert_at,
            SeqCEvent(
                start_samples=start_samples,
                length_samples=event_length,
                operation=Operation.START_QA,
                args=[
                    generators_mask,
                    integrators_mask,
                    self.descriptor.measurement_delay_samples,
                    input_monitor,
                    wave_data_idx,
                ],
            ),
        )

    def startQA_UHFQA(
        self,
        weighted_integrator_mask=None,
        monitor=False,
        result_address=0x0,
        trigger=0x0,
    ):
        if weighted_integrator_mask is None:
            weighted_integrator_mask = self.predefined_consts["QA_INT_ALL"]
        start_samples, insert_at = self._last_play_start_samples()
        self.seqc_simulation.events.insert(
            insert_at,
            SeqCEvent(
                start_samples=start_samples,
                length_samples=0,
                operation=Operation.START_QA,
                args=[
                    None,
                    weighted_integrator_mask,
                    self.descriptor.measurement_delay_samples,
                    monitor,
                    None,
                ],
            ),
        )

    def startQAResult(self):
        start_samples, insert_at = self._last_play_start_samples()
        self.seqc_simulation.events.insert(
            insert_at,
            SeqCEvent(
                start_samples=start_samples,
                length_samples=0,
                operation=Operation.START_QA,
                args=[
                    None,
                    0xFFFF,
                    self.descriptor.measurement_delay_samples,
                    None,
                    None,
                ],
            ),
        )

    def configFreqSweep(
        self, oscillator: int, freq_start: float, freq_increment: float
    ):
        self._oscillator_sweep_config[oscillator] = dict(
            start=freq_start, step=freq_increment
        )

    def setSweepStep(self, oscillator: int, index: int):
        try:
            d = self._oscillator_sweep_config[oscillator]
            start, step = d["start"], d["step"]
        except KeyError as e:
            raise RuntimeError(
                "setSweep() was called, but no sweep was set up with configSweep() earlier"
            ) from e

        freq = start + index * step
        self.setOscFreq(oscillator, freq)

    def setOscFreq(self, oscillator: int, frequency: float):
        start_samples, insert_at = self._last_play_start_samples()
        self.seqc_simulation.events.insert(
            insert_at,
            SeqCEvent(
                start_samples=start_samples,
                length_samples=0,
                operation=Operation.SET_OSC_FREQ,
                args=[oscillator, frequency],
            ),
        )

    def setParamVal(self, param_name: str, index: int, value: float):
        param_vals = self._oscillator_sweep_params.setdefault(param_name, {})
        param_vals[index] = value

    def setOscFreqByParam(self, param_name: str, index: int):
        self.setOscFreq(0, self._oscillator_sweep_params[param_name][index])

    def setTrigger(self, value: int):
        start_samples, insert_at = self._last_play_start_samples()
        self.seqc_simulation.events.insert(
            insert_at,
            SeqCEvent(
                start_samples=start_samples,
                length_samples=0,
                operation=Operation.SET_TRIGGER,
                args=[value],
            ),
        )

    def setPrecompClear(self, value: int):
        start_samples, insert_at = self._last_play_start_samples()
        self.seqc_simulation.events.insert(
            insert_at,
            SeqCEvent(
                start_samples=start_samples,
                length_samples=0,
                operation=Operation.SET_PRECOMP_CLEAR,
                args=[value],
            ),
        )

    def waitWave(self):
        time_samples = self._last_played_sample()
        self.seqc_simulation.events.append(
            SeqCEvent(
                start_samples=time_samples,
                length_samples=0,
                operation=Operation.WAIT_WAVE,
                args=[],
            )
        )


def find_device(recipe, device_uid):
    for device in recipe["devices"]:
        if device["device_uid"] == device_uid:
            return device
    return None


def analyze_recipe(
    recipe, sources, wave_indices, command_tables
) -> List[SeqCDescriptor]:
    outputs: Dict[str, List[int]] = {}
    seqc_descriptors_from_recipe: Dict[str, SeqCDescriptor] = {}
    for init in recipe["experiment"]["initializations"]:
        delay = 0
        if "measurements" in init and len(init["measurements"]) > 0:
            if "delay" in init["measurements"][0]:
                delay = init["measurements"][0]["delay"]

        device_uid = init["device_uid"]
        device = find_device(recipe, device_uid)
        device_type = device["driver"]
        sample_multiple = get_sample_multiple(device_type)
        try:
            sampling_rate = init["config"]["sampling_rate"]
        except KeyError:
            sampling_rate = None
        if sampling_rate is None or sampling_rate == 0:
            sampling_rate = get_frequency(device_type)
        startup_delay = -80e-9
        if device_type == "HDAWG" and "config" in init:
            if "dio_mode" in init["config"]:
                dio_mode = init["config"]["dio_mode"]
                if dio_mode == "hdawg_leader":
                    if sampling_rate == 2e9:
                        startup_delay = -24e-9
                    else:
                        startup_delay = -20e-9

        output_channel_delays: Dict[int, float] = {
            o["channel"]: o.get("port_delay", 0.0) for o in init.get("outputs", [])
        }

        output_channel_precompensation = {
            o["channel"]: o.get("precompensation", {}) for o in init.get("outputs", [])
        }

        awg_index = 0
        if "awgs" in init:
            for awg in init["awgs"]:
                seqc = awg["seqc"]
                awg_nr = awg["awg"]
                if device_type == "SHFSG" or device_type == "SHFQA":
                    output_channels = [awg_nr]
                else:
                    output_channels = [2 * awg_nr, 2 * awg_nr + 1]

                seqc_descriptors_from_recipe[seqc] = SeqCDescriptor(
                    name=seqc,
                    device_uid=device_uid,
                    device_type=device_type,
                    awg_index=awg_index,
                    measurement_delay_samples=delay,
                    startup_delay=startup_delay,
                    sample_multiple=sample_multiple,
                    sampling_rate=sampling_rate,
                    output_port_delay=output_channel_delays.get(
                        output_channels[0], 0.0
                    ),
                )

                precompensation_info = output_channel_precompensation.get(
                    output_channels[0]
                )
                if precompensation_info is not None:
                    precompensation_delay = (
                        precompensation_delay_samples(precompensation_info)
                        / sampling_rate
                    )
                    seqc_descriptors_from_recipe[
                        seqc
                    ].output_port_delay += precompensation_delay

                channels: List[int] = [
                    output["channel"]
                    for output in init["outputs"]
                    if output["channel"] in output_channels
                ]
                if len(channels) == 0:
                    channels.append(0)
                outputs[seqc] = channels

                awg_index += 1

    seq_c_wave_indices = {}
    for wave_index in wave_indices:
        wave_seq_c_filename = (
            wave_index["filename"][: -len("_waveindices.csv")] + ".seqc"
        )
        if len(wave_index["value"]) > 0:
            seq_c_wave_indices[wave_seq_c_filename] = {}
            for wave_name, index_value in wave_index["value"].items():
                seq_c_wave_indices[wave_seq_c_filename][index_value[0]] = {
                    "type": index_value[1],
                    "wave_name": wave_name,
                }

    seqc_descriptors = []
    for src in sources:
        name = src["filename"]
        command_table = next(
            (table["ct"] for table in command_tables if table["seqc"] == name), {}
        )
        seqc_descriptor = seqc_descriptors_from_recipe[name]
        seqc_descriptor.source = src["text"]
        seqc_descriptor.channels = outputs[name]
        seqc_descriptor.wave_index = seq_c_wave_indices.get(name, {})
        seqc_descriptor.command_table = command_table
        seqc_descriptors.append(seqc_descriptor)
    return seqc_descriptors


def run_single_source(descriptor: SeqCDescriptor, waves, max_time) -> SeqCSimulation:
    runtime = SimpleRuntime(
        descriptor=descriptor,
        waves=waves,
        max_time=max_time,
    )
    parse_seq_c(runtime)
    return runtime.seqc_simulation


def preprocess_source(text):
    parts = text.split("/* === END-OF-FUNCTION-DEFS === */\n")
    if len(parts) > 1:
        user_functions = parts[0]
        main = parts[1]
    else:
        user_functions = ""
        main = parts[0]
    # Strip-off comments
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    user_functions = regex.sub(_replacer, user_functions)
    main = regex.sub(_replacer, main)

    # Define SeqC built-ins and wrap program into function
    # to make the program syntactically correct for C parser.
    if len(main) > 0:
        source = (
            f"typedef int var;\n"
            f"typedef const char* wave;\n"
            f"typedef const char* string;\n"
            f"{user_functions}\n"
            f"void f(void){{\n{main}\n}}"
        )

    return source


def _analyze_compiled(
    compiled: CompiledExperiment,
) -> Tuple[List[SeqCDescriptor], Dict[str, npt.ArrayLike]]:
    if isinstance(compiled, dict):
        compiled = SimpleNamespace(
            recipe=compiled["recipe"],
            src=compiled["src"],
            waves=compiled["waves"],
            wave_indices=compiled["wave_indices"],
        )
    seqc_descriptors = analyze_recipe(
        compiled.recipe, compiled.src, compiled.wave_indices, compiled.command_tables
    )

    read_wave_bin = lambda w: w if w.ndim == 1 else np.array([[s] for s in w])
    waves = {w["filename"]: read_wave_bin(w["samples"]) for w in compiled.waves}
    return seqc_descriptors, waves


def simulate(compiled: CompiledExperiment, max_time=None) -> Dict[str, SeqCSimulation]:
    seqc_descriptors, waves = _analyze_compiled(compiled)
    results: Dict[str, SeqCSimulation] = {}
    for descriptor in seqc_descriptors:
        results[descriptor.name] = run_single_source(descriptor, waves, max_time)
    return results
