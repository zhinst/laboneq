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
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy import typing as npt
from pycparser.c_ast import (
    ID,
    Assignment,
    BinaryOp,
    Constant,
    Decl,
    DoWhile,
    For,
    FuncCall,
    FuncDef,
    If,
    Node,
    UnaryOp,
)
from pycparser.c_parser import CParser

from laboneq.compiler.common.compiler_settings import EXECUTETABLEENTRY_LATENCY
from laboneq.data.recipe import Recipe, TriggeringMode, RoutedOutput

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
    exponential = precompensation.get("exponential")
    if exponential is not None:
        delay += 88 * len(exponential)
    if precompensation.get("high_pass") is not None:
        delay += 96
    if precompensation.get("bounce") is not None:
        delay += 32
    if precompensation.get("FIR") is not None:
        delay += 136
    return delay


def calculate_output_router_delays(
    mapping: Mapping[int, Sequence[RoutedOutput]]
) -> dict[int, int]:
    """Calculate delays introduced from using Output router.

    Using output router introduces a constant delay of 52 samples.
    """
    key_to_delay = {}
    for uid, routing in mapping.items():
        if routing:
            key_to_delay[uid] = 52
            for output in routing:
                key_to_delay[output.from_channel] = 52
        else:
            if uid not in key_to_delay:
                key_to_delay[uid] = 0
    return key_to_delay


def get_frequency(device_type: str) -> float:
    try:
        return {
            "HDAWG": 2.4e9,
            "UHFQA": 1.8e9,
            "SHFQA": 2e9,
            "SHFSG": 2e9,
            "SHFQC": 2e9,
            "PQSC": 0.0,
            "SHFPPC": 0.0,
        }[device_type]
    except KeyError:
        raise RuntimeError(f"Unsupported device type {device_type}") from None


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

    try:
        for item in ast.ext[-1].body.block_items:
            parse_item(item, runtime)
    except StopSimulation:
        pass

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
        if item.op == "+":
            return parse_expression(item.left, runtime) + parse_expression(
                item.right, runtime
            )
        if item.op == "-":
            return parse_expression(item.left, runtime) - parse_expression(
                item.right, runtime
            )
        if item.op == "*":
            return parse_expression(item.left, runtime) * parse_expression(
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
        raise ValueError(f"Invalid int literal: '{s}'") from None


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
    elif isinstance(item, For):
        if item.cond is not None or item.next is not None:
            raise NotImplementedError("for loops not supported in the simulator")

        # if cond and next are None, assume the for loop is just encoding a seqc repeat
        n = int(parse_expression(item.init, runtime))
        for _i in range(n):
            for subitem in item.stmt.children():
                parse_item(subitem[1], runtime)
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
    channels: list[int] = None
    wave_index: dict[Any, Any] = None
    command_table: list[Any] = None
    is_spectroscopy: bool = False


class Operation(Enum):
    PLAY_ZERO = auto()
    PLAY_WAVE = auto()
    START_QA = auto()
    SET_OSC_FREQ = auto()
    SET_TRIGGER = auto()
    SET_PRECOMP_CLEAR = auto()
    WAIT_WAVE = auto()
    PLAY_HOLD = auto()


@dataclass
class SeqCEvent:
    start_samples: int
    length_samples: int
    operation: Operation
    args: list[Any]


@dataclass
class WaveRefInfo:
    assigned_index: int = -1
    wave_data_idx: list[int] = field(default_factory=list)
    length_samples: int = None


@dataclass
class CommandTableEntryInfo:
    abs_phase: float | None = None
    rel_phase: float | None = None
    abs_amplitude: float | None = None
    rel_amplitude: float | None = None
    amp_register: int | None = None

    @classmethod
    def from_ct_entry(cls, ct_entry: dict):
        d = {}
        if "phase" in ct_entry:
            # SHFSG
            incr = ct_entry["phase"].get("increment", False)
            if incr:
                d["rel_phase"] = ct_entry["phase"]["value"]
            else:
                d["abs_phase"] = ct_entry["phase"]["value"]

        if "phase0" in ct_entry:
            # HDAWG
            incr = ct_entry["phase0"].get("increment", False)
            assert ct_entry["phase1"].get("increment", False) == incr
            phase = ct_entry["phase0"]["value"]
            if not incr:
                phase -= 90
            assert (
                abs((phase - ct_entry["phase1"]["value"] + 180.0) % 360.0 - 180.0)
                < 1e-6
            )

            if incr:
                d["rel_phase"] = phase
            else:
                d["abs_phase"] = phase

        if "amplitude0" in ct_entry:
            # HDAWG
            increment = ct_entry["amplitude0"].get("increment", False)
            amp_register = ct_entry["amplitude0"].get("register")
            amplitude = ct_entry["amplitude0"]["value"]
            assert ct_entry["amplitude1"]["value"] == amplitude
            assert ct_entry["amplitude1"].get("register") == amp_register
            assert ct_entry["amplitude1"].get("increment", False) == increment
            if increment:
                d["rel_amplitude"] = amplitude
            else:
                d["abs_amplitude"] = amplitude
            d["amp_register"] = amp_register
        elif "amplitude00" in ct_entry:
            # SHFSG
            increment = ct_entry["amplitude00"].get("increment", False)
            amp_register = ct_entry["amplitude00"].get("register")
            amplitude = ct_entry["amplitude00"]["value"]
            for amp_field in ["amplitude01", "amplitude10", "amplitude11"]:
                ct_amp = ct_entry[amp_field]
                assert ct_amp.get("register") == amp_register
                assert ct_amp.get("increment", False) == increment
            assert ct_entry["amplitude01"]["value"] == -amplitude
            assert ct_entry["amplitude10"]["value"] == amplitude
            assert ct_entry["amplitude11"]["value"] == amplitude
            if increment:
                d["rel_amplitude"] = amplitude
            else:
                d["abs_amplitude"] = amplitude
            d["amp_register"] = amp_register

        return cls(**d)


@dataclass
class SeqCSimulation:
    events: list[SeqCEvent] = field(default_factory=list)
    device_type: str = ""
    waves: list[Any] = field(default_factory=list)
    sampling_rate: float = field(default=2.0e9)
    startup_delay: float = field(default=0.0)
    output_port_delay: float = field(default=0.0)
    is_spectroscopy: bool = False


class SimpleRuntime:
    def __init__(
        self,
        descriptor: SeqCDescriptor,
        waves,
        max_time: float | None,
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
            "ZSYNC_DATA_PQSC_REGISTER": 0b1000000000001
            if descriptor.device_type == "SHFSG"
            else 0b10000000001,
        }
        self.exposedFunctions = {
            "assignWaveIndex": self.assignWaveIndex,
            "playWave": self.playWave,
            "playZero": self.playZero,
            "playHold": self.playHold,
            "executeTableEntry": self.executeTableEntry,
            "startQA": self.startQA,
            "startQAResult": self.startQAResult,
            "configFreqSweep": self.configFreqSweep,
            "setSweepStep": self.setSweepStep,
            "setOscFreq": self.setOscFreq,
            "setTrigger": self.setTrigger,
            "setPrecompClear": self.setPrecompClear,
            "waitWave": self.waitWave,
            "waitDIOTrigger": self.waitDIOTrigger,
            "waitZSyncTrigger": self.waitZSyncTrigger,
        }
        self.variables = {}
        self.seqc_simulation = SeqCSimulation()
        self.seqc_simulation.is_spectroscopy = descriptor.is_spectroscopy
        self.times = {}
        self.times_at_port = {}
        self.descriptor = descriptor
        self.waves = waves
        self.source = preprocess_source(descriptor.source)
        self.wave_lookup_by_args: dict[Any, WaveRefInfo] = {}
        self.wave_names_by_index: dict[int, list[str]] = {}
        self.wave_data: list[Any] = []
        self.max_time: float | None = max_time
        self._oscillator_sweep_config = {}
        self._oscillator_sweep_params: dict[str, dict[int, float]] = {}
        self._command_table_by_index = {
            ct["index"]: ct for ct in self.descriptor.command_table
        }

    def _last_played_sample(self) -> int:
        ev = self.seqc_simulation.events
        return ev[-1].start_samples + ev[-1].length_samples if len(ev) > 0 else 0

    def _last_play_start_samples(self) -> tuple[int, int]:
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
            Operation.PLAY_HOLD,  # ?
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

    def _update_wave_refs(self, wave_names: list[str], known_wave: WaveRefInfo):
        known_length = known_wave.length_samples  # make VSCode's code parser happy
        if known_length is not None:
            return
        for wave_name in wave_names:
            if wave_name is None:
                known_wave.wave_data_idx.append(None)
                continue
            if wave_name.startswith("precomp_reset"):
                wave_to_play = np.zeros(32)
            else:
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

    def _append_wave_event(
        self,
        wave_names: list[str] | None,
        known_wave: WaveRefInfo | None,
        ct_info: CommandTableEntryInfo | None,
    ):
        if wave_names is not None:
            assert known_wave is not None
            self._update_wave_refs(wave_names, known_wave)

            uses_marker_1 = any(
                ["marker1" in wave for wave in wave_names if wave is not None]
            )
            uses_marker_2 = any(
                ["marker2" in wave for wave in wave_names if wave is not None]
            )
            length_samples = known_wave.length_samples
            wave_data_indices = known_wave.wave_data_idx
        else:
            uses_marker_1 = False
            uses_marker_2 = False
            length_samples = 0
            wave_data_indices = None

        time_samples = self._last_played_sample()
        self.seqc_simulation.events.append(
            SeqCEvent(
                start_samples=time_samples,
                length_samples=length_samples,
                operation=Operation.PLAY_WAVE,
                args=[
                    wave_data_indices,
                    ct_info,
                    {"marker1": uses_marker_1, "marker2": uses_marker_2},
                ],
            )
        )

    def assignWaveIndex(self, *args):
        idx = args[-1]
        wave_key = self._args2key(args[:-1])
        known_wave = self.wave_lookup_by_args.get(wave_key)
        if known_wave is None:
            wave_ref = WaveRefInfo(assigned_index=idx)
            self.wave_lookup_by_args[wave_key] = wave_ref
            wave_names = [
                name[1:-1] or None for name in wave_key if isinstance(name, str)
            ]
            self.wave_names_by_index[idx] = wave_names
        else:
            if known_wave.assigned_index != idx:
                raise Exception(
                    f"Attempt to assign wave index {idx} to args {wave_key} having already index {known_wave.assigned_index}"
                )

    def playWave(self, *args):
        wave_key = self._args2key(args)
        known_wave = self.wave_lookup_by_args.get(wave_key)
        if known_wave is None:
            known_wave = WaveRefInfo()
            self.wave_lookup_by_args[wave_key] = known_wave

        wave_format = ".csv" if known_wave.assigned_index == -1 else ".wave"

        # Supporting only combinations emitted by L1Q compiler, not any possible SeqC
        # handle also instructions like `playWave(1, "", 2, w1);`
        wave_names = [
            None if arg == '""' else (arg.strip('"') + wave_format)
            for arg in args
            if isinstance(arg, str)
        ]

        if not wave_names or wave_names == [None]:
            raise RuntimeError(f"Couldn't determine wave name(s) from {args}")

        self._append_wave_event(wave_names, known_wave, None)

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

    def playHold(self, length):
        if length > 0:
            time_samples = self._last_played_sample()
            self.seqc_simulation.events.append(
                SeqCEvent(
                    start_samples=time_samples,
                    length_samples=length,
                    operation=Operation.PLAY_HOLD,
                    args=[],
                )
            )

    def _waves_from_command_table_entry(self, ct_entry):
        if "waveform" not in ct_entry:
            return None, None
        if "index" not in ct_entry["waveform"]:
            return None, None
        wave_index = ct_entry["waveform"]["index"]
        known_wave = WaveRefInfo(assigned_index=wave_index)

        wave = self.descriptor.wave_index[wave_index]

        if wave["type"] in ("iq", "multi"):
            wave_names = [
                wave["wave_name"] + suffix + ".wave" for suffix in ("_i", "_q")
            ]
        elif wave["type"] in ["single", "double"]:
            wave_names = [
                name + ".wave" if name is not None else None
                for name in self.wave_names_by_index[wave_index]
            ]
        else:
            raise ValueError(f"Unknown signal type: {wave['type']}")

        for candidate_wave in self.waves.keys():
            if wave["wave_name"] in candidate_wave and "marker" in candidate_wave:
                wave_names.append(candidate_wave)

        return wave_names, known_wave

    def executeTableEntry(self, ct_index, latency=None):
        QA_DATA_PROCESSED_SG = 0b1000000000100
        ZSYNC_DATA_PQSC_REGISTER_SG = 0b1000000000001
        ZSYNC_DATA_PQSC_REGISTER_HD = 0b10000000001
        if ct_index == QA_DATA_PROCESSED_SG or ct_index == ZSYNC_DATA_PQSC_REGISTER_SG:
            assert self.descriptor.device_type == "SHFSG"
            # todo(JL): Find a better index via the command table offset; take last for now
            ct_index = self.descriptor.command_table[-1]["index"]
        elif ct_index == ZSYNC_DATA_PQSC_REGISTER_HD:
            assert self.descriptor.device_type == "HDAWG"
            # todo(JL): Find a better index via the command table offset; take last for now
            ct_index = self.descriptor.command_table[-1]["index"]

        ct_entry = self._command_table_by_index[ct_index]

        wave_names, known_wave = self._waves_from_command_table_entry(ct_entry)

        ct_info = CommandTableEntryInfo.from_ct_entry(ct_entry)

        if latency is not None:
            time_samples = self._last_played_sample()
            corrected_latency = latency + EXECUTETABLEENTRY_LATENCY
            if corrected_latency * 8 < time_samples:
                raise RuntimeError(
                    f"ExecuteTableEntry scheduled with latency {latency} before current time {time_samples}"
                )
            elif corrected_latency * 8 > time_samples:
                raise RuntimeError(
                    f"Play queue starved at current time {time_samples} for ExecuteTableEntry scheduled with latency {latency}"
                )

        if ct_entry.get("waveform", {}).get("precompClear", False):
            self.setPrecompClear(1)
            self._append_wave_event(wave_names, known_wave, ct_info)
            self.setPrecompClear(0)
        else:
            self._append_wave_event(wave_names, known_wave, ct_info)

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
        trigger=None,
    ):
        if generators_mask is None:
            generators_mask = self.resolve("QA_GEN_ALL")
        if integrators_mask is None:
            generators_mask = self.resolve("QA_INT_ALL")

        wave_data_idx = []
        event_length = 0

        def add_wave(gen_index, wave_data_idx, event_length):
            wave_key = self._args2key(["gen", gen_index])
            known_wave = self.wave_lookup_by_args.get(wave_key)
            if known_wave is None:
                known_wave = WaveRefInfo()
                self.wave_lookup_by_args[wave_key] = known_wave
            wave = self.descriptor.wave_index[gen_index]
            wave_names = [wave["wave_name"] + ".wave"]
            wave_length_samples = len(self.waves[wave_names[0]])
            event_length = max(event_length, wave_length_samples)
            self._update_wave_refs(wave_names, known_wave)
            wave_data_idx.append(known_wave.wave_data_idx)
            return wave_data_idx, event_length

        if self.descriptor.is_spectroscopy:
            assert generators_mask == self.predefined_consts["QA_GEN_NONE"]
            wave_data_idx, event_length = add_wave(0, wave_data_idx, event_length)
        else:
            for gen_index in range(16):
                if (generators_mask & (1 << gen_index)) != 0:
                    wave_data_idx, event_length = add_wave(
                        gen_index, wave_data_idx, event_length
                    )

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
                    trigger,
                ],
            ),
        )

    def startQA_UHFQA(
        self,
        weighted_integrator_mask=None,
        monitor=False,
        result_address=0x0,
        trigger=None,
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
                    trigger,
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

    def _start_trigger(self):
        # Here the assumption is that before the start trigger event, only playZeros
        # could potentially affect timings, so we only remove the effect of playZero
        # events. Besides, only one start trigger event is assumed.
        back_shift_samples = 0
        filtered_events = []
        for ev in self.seqc_simulation.events:
            if ev.operation == Operation.PLAY_ZERO:
                back_shift_samples += ev.length_samples
            else:
                ev.start_samples -= back_shift_samples
                filtered_events.append(ev)
        self.seqc_simulation.events = filtered_events

    def waitDIOTrigger(self):
        self._start_trigger()

    def waitZSyncTrigger(self):
        self._start_trigger()


def analyze_recipe(
    recipe: Recipe, sources, wave_indices, command_tables
) -> list[SeqCDescriptor]:
    outputs: dict[str, list[int]] = {}
    seqc_descriptors_from_recipe: dict[str, SeqCDescriptor] = {}
    for init in recipe.initializations:
        device_uid = init.device_uid
        device_type = init.device_type
        sample_multiple = get_sample_multiple(device_type)
        sampling_rate = init.config.sampling_rate
        if sampling_rate is None or sampling_rate == 0:
            sampling_rate = get_frequency(device_type)
        startup_delay = -80e-9
        if device_type == "HDAWG":
            triggering_mode = init.config.triggering_mode
            if triggering_mode == TriggeringMode.DESKTOP_LEADER:
                if sampling_rate == 2e9:
                    startup_delay = -24e-9
                else:
                    startup_delay = -20e-9

        # TODO(2K): input port_delay previously was not taken into account by the simulator
        # - keeping it as is for not breaking the tests. To be cleaned up.
        input_channel_delays: dict[int, float] = {
            i.channel: i.scheduler_port_delay  # + (0.0 if i.port_delay is None else i.port_delay)
            for i in init.inputs
        }

        output_channel_delays: dict[int, float] = {
            o.channel: o.scheduler_port_delay
            + (0.0 if o.port_delay is None else o.port_delay)
            for o in init.outputs
        }
        output_channel_precompensation = {
            o.channel: o.precompensation for o in init.outputs
        }
        output_channel_output_routers = {
            o.channel: o.routed_outputs for o in init.outputs
        }
        output_channel_output_router_delays = calculate_output_router_delays(
            output_channel_output_routers
        )
        awg_index = 0
        for awg in init.awgs:
            awg_nr = awg.awg
            rt_exec_step = next(
                r
                for r in recipe.realtime_execution_init
                if r.device_id == device_uid and r.awg_id == awg_nr
            )
            seqc = rt_exec_step.seqc_ref
            if device_type == "SHFSG" or device_type == "SHFQA":
                input_channel = awg_nr
                output_channels = [awg_nr]
            else:
                input_channel = 2 * awg_nr
                output_channels = [2 * awg_nr, 2 * awg_nr + 1]

            seqc_descriptors_from_recipe[seqc] = SeqCDescriptor(
                name=seqc,
                device_uid=device_uid,
                device_type=device_type,
                awg_index=awg_index,
                measurement_delay_samples=round(
                    input_channel_delays.get(input_channel, 0.0) * sampling_rate
                ),
                startup_delay=startup_delay,
                sample_multiple=sample_multiple,
                sampling_rate=sampling_rate,
                output_port_delay=output_channel_delays.get(output_channels[0], 0.0),
            )

            precompensation_info = output_channel_precompensation.get(
                output_channels[0]
            )
            if precompensation_info is not None:
                precompensation_delay = (
                    precompensation_delay_samples(precompensation_info) / sampling_rate
                )
                seqc_descriptors_from_recipe[
                    seqc
                ].output_port_delay += precompensation_delay
            delay_by_output_router = output_channel_output_router_delays.get(
                output_channels[0]
            )
            if delay_by_output_router is not None:
                seqc_descriptors_from_recipe[seqc].output_port_delay += (
                    delay_by_output_router / sampling_rate
                )
            channels: list[int] = [
                output.channel
                for output in init.outputs
                if output.channel in output_channels
            ]
            if len(channels) == 0:
                channels.append(0)
            outputs[seqc] = channels

            awg_index += 1

    seq_c_wave_indices = {}
    for wave_index in wave_indices:
        wave_seq_c_filename = wave_index["filename"]
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
        seqc_descriptor.is_spectroscopy = recipe.is_spectroscopy
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
        neartime_callbacks = parts[0]
        main = parts[1]
    else:
        neartime_callbacks = ""
        main = parts[0]
    # Strip-off comments
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    neartime_callbacks = regex.sub(_replacer, neartime_callbacks)
    main = regex.sub(_replacer, main)

    # repeat(n) {...} is not valid C, and topples the parser. Replace it with `for(n;;) {...}`.
    # Note that the result does not correspond to the usual C for-loop semantics.
    # This is fine though, as we do not emit 'regular' for loops from L1Q.
    pattern = r"repeat\s*\((\d+)\)\s*{"
    regex = re.compile(pattern)

    def replace_repeat(match_obj):
        if match_obj.group(1) is not None:
            n = match_obj.group(1)
            return f"for({n};;) {{"

    main = regex.sub(replace_repeat, main)

    # Constant definitions in SeqC omit the type (constants can only be int or float;
    # compile-time strings are instead defined via the `string` keyword). This is not
    # valid C, so we 'patch' statements like `const a = 5;` to `const int a = 5;`.
    main = re.sub(r"const(\s+[A-Za-z_]\w*\s+=)", r"const int\1", main)

    # Define SeqC built-ins and wrap program into function
    # to make the program syntactically correct for C parser.
    if len(main) > 0:
        source = (
            f"typedef int var;\n"
            f"typedef const char* wave;\n"
            f"typedef const char* string;\n"
            f"{neartime_callbacks}\n"
            f"void f(void){{\n{main}\n}}"
        )

    return source


def _analyze_compiled(
    compiled: CompiledExperiment,
) -> tuple[list[SeqCDescriptor], dict[str, npt.ArrayLike]]:
    if isinstance(compiled, dict):
        compiled = SimpleNamespace(
            scheduled_experiment=SimpleNamespace(
                recipe=compiled["recipe"],
                src=compiled["src"],
                waves=compiled["waves"],
                wave_indices=compiled["wave_indices"],
            )
        )
    seqc_descriptors = analyze_recipe(
        compiled.scheduled_experiment.recipe,
        compiled.scheduled_experiment.src,
        compiled.scheduled_experiment.wave_indices,
        compiled.scheduled_experiment.command_tables,
    )

    read_wave_bin = lambda w: w if w.ndim == 1 else np.array([[s] for s in w])
    waves = {
        w["filename"]: read_wave_bin(w["samples"])
        for w in compiled.scheduled_experiment.waves
    }
    return seqc_descriptors, waves


def simulate(compiled: CompiledExperiment, max_time=None) -> dict[str, SeqCSimulation]:
    seqc_descriptors, waves = _analyze_compiled(compiled)
    results: dict[str, SeqCSimulation] = {}
    for descriptor in seqc_descriptors:
        results[descriptor.name] = run_single_source(descriptor, waves, max_time)
    return results
