# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Dict

from pycparser.c_parser import CParser
from pycparser.c_ast import (
    Decl,
    FuncCall,
    Constant,
    ID,
    DoWhile,
    Assignment,
    BinaryOp,
    UnaryOp,
)
import numpy as np

import os
import fnmatch
import csv
import json

from types import SimpleNamespace

from io import StringIO

import logging
from engineering_notation import EngNumber
import re

from laboneq.compiler import CompilerSettings
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.compiled_experiment import CompiledExperiment

_logger = logging.getLogger(__name__)


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
        raise Exception(f"Unsupported device type {device_type}")


def get_sample_multiple(device_type):
    if device_type in ["HDAWG", "SHFQA", "SHFSG"]:
        return 16
    elif device_type == "UHFQA":
        return 8
    else:
        return 16


def read_wave_bin(bin_wave):
    if bin_wave.ndim == 1:
        return bin_wave
    return np.array([[s] for s in bin_wave])


def parse_seq_c(source, filename, runtime):

    if len(source) == 0:
        return
    c_code = "typedef int var;\ntypedef const char* wave;\nvoid f{\n" + source + "\n}"

    parser = CParser()
    ast = parser.parse(c_code, filename)
    ast_stream = StringIO()
    ast.show(buf=ast_stream)
    _logger.debug(ast_stream.getvalue())
    _logger.debug("***********")

    for item in ast.ext[2].body.block_items:
        try:
            parse_item(item, runtime)
        except StopSimulation:
            break

    runtime.program_finished()


def parse_expression(item, runtime):
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


def parse_int_literal(s):
    SUFFIX = r"[uU]?(ll|LL|l|L)?[uU]?$"
    PATTERN_BINARY = r"0[bB](?P<body>[01]+)" + SUFFIX
    PATTERN_OCTAL = r"(?P<body>0[0-7]+)" + SUFFIX
    PATTERN_DECIMAL = r"(?P<body>\d+)" + SUFFIX
    PATTERN_HEX = r"(0x|0X)(?P<body>[\da-fA-F]+)" + SUFFIX

    assert len(s)

    for pattern, base in zip(
        (PATTERN_BINARY, PATTERN_OCTAL, PATTERN_DECIMAL, PATTERN_HEX), (2, 8, 10, 16)
    ):
        match = re.match(pattern, s)
        if match:
            return int(match.group("body"), base)
    else:
        raise ValueError(f"Invalid int literal: {s}")


def parse_constant(constant: Constant):
    converter = {"int": parse_int_literal, "float": float, "double": float}
    if constant.type in converter:
        return converter[constant.type](constant.value)
    _logger.debug("Invalid type %s", constant.type)
    return constant.value


class StopSimulation(Exception):
    pass


def parse_item(item, runtime: "SimpleRuntime"):
    if isinstance(item, Decl):
        runtime.declare(item.name)
        init_value = None
        if isinstance(item.init, Constant):
            init_value = parse_constant(item.init)
        elif isinstance(item.init, FuncCall):
            init_value = f'"{item.name[1:]}"'
        runtime.initialize(item.name, init_value)
        _logger.debug("%s IS SET TO %s", item.name, init_value)
    elif isinstance(item, FuncCall):
        funcname = item.name.name
        _logger.debug("Call to %s", funcname)
        args = (
            tuple(parse_expression(arg, runtime) for arg in item.args)
            if item.args is not None
            else ()
        )
        if funcname in runtime.exposedFunctions:
            _logger.debug(
                "Call to %s with args %s item.args %s", funcname, args, item.args
            )
            runtime.exposedFunctions[funcname](*args)
        else:
            _logger.debug("Skipping unknown function %s", funcname)

    elif isinstance(item, Assignment):
        variables = runtime.variables
        if item.op == "-=":
            _logger.debug("** -=")
            variables[item.lvalue.name]["value"] = variables[item.lvalue.name][
                "value"
            ] - parse_expression(item.rvalue, runtime)
            _logger.debug(
                "Changed %s TO %s",
                item.lvalue.name,
                variables[item.lvalue.name]["value"],
            )
        if item.op == "+=":
            _logger.debug("** +=")
            variables[item.lvalue.name]["value"] = variables[item.lvalue.name][
                "value"
            ] + parse_expression(item.rvalue, runtime)
            _logger.debug(
                "Changed %s TO %s",
                item.lvalue.name,
                variables[item.lvalue.name]["value"],
            )
        elif item.op == "=":
            _logger.debug("** = , type of right hand side: %s", type(item.rvalue))
            variables[item.lvalue.name]["value"] = parse_expression(
                item.rvalue, runtime
            )
            _logger.debug(
                "Set %s TO %s", item.lvalue.name, variables[item.lvalue.name]["value"]
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
                _logger.debug("Variable value: %s", variables[item.cond.name]["value"])
                condition_variable_name = item.cond.name
                variable_value = int(variables[condition_variable_name]["value"])
            except AttributeError:
                _logger.debug("Cannot determine value of loop condition")

            if variable_value <= 0:
                _logger.debug(
                    "While loop on variable %s ends now", condition_variable_name
                )
                break
            else:
                _logger.debug(
                    "Continuing while loop on variable %s", condition_variable_name
                )
            endless_guard -= 1
            if endless_guard <= 0:
                raise "Endless guard triggered"
    if (
        runtime.max_time is not None
        and runtime.last_play_start_samples / runtime.sample_frequency
        > runtime.max_time
    ):
        _logger.info(
            "Output simulation truncated on %s at %d samples because of max_time=%f",
            runtime.device_uid,
            runtime.last_play_start_samples,
            runtime.max_time,
        )
        raise StopSimulation


@dataclasses.dataclass
class SeqCMetadata:
    name: str
    source: str
    channels: list
    device_uid: str
    device_type: str
    measurement_delay_samples: int
    startup_delay: float
    sampling_rate: float
    sample_multiple: int
    awg_index: int
    wave_index: dict
    output_port_delay: int


class SimpleRuntime:
    def __init__(
        self,
        waves,
        channels,
        device_uid,
        device_type,
        sample_frequency,
        delay,
        scale_factor,
        measurement_delay_samples,
        sample_multiple,
        awg_index,
        wave_index,
        max_time,
        output_port_delay,
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
            "QA_INT_ALL": 0b111111111111111,
            "QA_INT_NONE": 0,
            "QA_GEN_ALL": 0b111111111111111,
            "QA_GEN_NONE": 0,
        }
        self.exposedFunctions = {
            "assignWaveIndex": self.assignWaveIndex,
            "playWave": self.playWave,
            "playZero": self.playZero,
            "startQA": self.startQA,
            "startQAResult": self.startQAResult,
            "configFreqSweep": self.configFreqSweep,
            "setSweepStep": self.setSweepStep,
            "setOscFreq": self.setOscFreq,
        }
        self.variables = {}
        self.output: Dict[str, np.ndarray] = {}
        self.times = {}
        self.times_at_port = {}
        self.channel_mapping = {}
        self.scale_factor = scale_factor
        self.device_uid = device_uid
        self.device_type = device_type
        if self.device_type != "SHFQA":
            index = 1
            for i in channels:
                self.output[str(i)] = np.zeros(0)
                self.channel_mapping[str(index)] = str(i)
                index += 1
            _logger.debug(
                "Channels %s with mapping %s ", self.output, self.channel_mapping
            )

        if self.device_type == "SHFSG":
            self.output = {}
            self.channel_mapping = {
                "1": str(channels[0]) + "_I",
                "2": str(channels[0]) + "_Q",
            }
            for k in self.channel_mapping.values():
                self.output[k] = np.zeros(0)

        _logger.debug(
            "channels = %s , output %s with mapping %s ",
            channels,
            self.output,
            self.channel_mapping,
        )
        self.waves = waves

        self.sample_frequency = sample_frequency
        self.delay = delay
        self.measurement_delay_samples = measurement_delay_samples
        self.last_play_start_samples = 0
        self.last_play_zero_end = None
        self.sample_multiple = sample_multiple
        self.wave_index_lookup = {}
        self.awg_index = awg_index
        self.wave_index = wave_index
        self.max_time = max_time
        self.output_port_delay = output_port_delay
        self._oscillator_frequency = {}
        self._oscillator_sweep_config = {}

    def program_finished(self):
        for k, v in self.output.items():
            num_samples = len(v)
            _logger.debug(
                "Calculating times with frequency %f for channel %s, total time %f, the signal is leading by %s",
                self.sample_frequency,
                k,
                num_samples / self.sample_frequency,
                EngNumber(-self.delay),
            )
            self.times[k] = np.arange(num_samples) / self.sample_frequency + self.delay
            port_delay = self.output_port_delay
            if "QAResult" in k:
                port_delay = 0
            self.times_at_port[k] = (
                np.arange(num_samples) / self.sample_frequency + self.delay + port_delay
            )

    def declare(self, name):
        self.variables[name] = {"name": name}

    def initialize(self, name, value):
        self.variables[name]["value"] = value

    def resolve(self, name):
        if name in self.predefined_consts:
            return self.predefined_consts[name]
        return self.variables[name]["value"]

    def args2key(self, args):
        return tuple(tuple(a.items()) if isinstance(a, dict) else a for a in args)

    def assignWaveIndex(self, *args):
        idx = args[-1]
        wave_args = self.args2key(args[:-1])
        known_idx = self.wave_index_lookup.get(wave_args)
        if known_idx is None:
            self.wave_index_lookup[wave_args] = idx
        else:
            if known_idx != idx:
                raise Exception(
                    f"Attempt to assign wave index {idx} to args {wave_args} having already index {known_idx}"
                )

    def playWave(self, *args):
        _logger.debug("play wave called with " + str(args))
        known_idx = self.wave_index_lookup.get(self.args2key(args))
        is_single_play = False
        if len(args) == 2:
            if args[0] == '""' or args[1] == '""':
                is_single_play = True
        if len(args) == 1 or is_single_play:
            only_arg = args[0]
            if is_single_play:
                if args[0] == '""':
                    only_arg = args[1]
                key = tuple([tuple(only_arg.items())])
                known_idx = self.wave_index_lookup.get(key)
            filename = only_arg.strip('"') + (
                ".wave" if known_idx is not None else ".csv"
            )

            output_channel = "1"
            wave_to_play = self.waves[filename]

            if output_channel in self.channel_mapping:
                output_channel = self.channel_mapping[output_channel]

            self.last_play_start_samples = len(self.output[output_channel])
            if np.ndim(wave_to_play) == 2 and np.shape(wave_to_play)[1] == 2:
                second_channel = str(int(output_channel) + 1)
                if second_channel in self.channel_mapping:
                    second_channel = self.channel_mapping[second_channel]

                _logger.debug(
                    "playing %s (%d samples) on channel %s, channel currently has %d samples",
                    filename,
                    len(wave_to_play),
                    output_channel,
                    len(self.output[output_channel]),
                )
                if len(wave_to_play[:, 0]) % self.sample_multiple != 0:
                    raise Exception(
                        f"Wave {filename} has {len(wave_to_play)} samples, which is not divisible by {self.sample_multiple}"
                    )

                self.output[output_channel] = np.append(
                    self.output[output_channel], self.scale_factor * wave_to_play[:, 0]
                )
                self.output[second_channel] = np.append(
                    self.output[second_channel], self.scale_factor * wave_to_play[:, 1]
                )
                _logger.debug(
                    "Channel %s now has %d samples",
                    output_channel,
                    len(self.output[output_channel]),
                )
                _logger.debug(
                    "Channel %s now has %d samples",
                    second_channel,
                    len(self.output[second_channel]),
                )
            else:
                if len(wave_to_play) % self.sample_multiple != 0:
                    raise Exception(
                        f"Wave {filename} has {len(wave_to_play)} samples, which is not divisible by {self.sample_multiple}"
                    )
                _logger.debug(
                    "playing %s (%d samples) on channel %s, channel currently has %d samples",
                    filename,
                    len(wave_to_play),
                    output_channel,
                    len(self.output[output_channel]),
                )
                self.last_play_start_samples = len(self.output[output_channel])

                self.output[output_channel] = np.append(
                    self.output[output_channel], self.scale_factor * wave_to_play
                )
                _logger.debug(
                    "Channel %s now has %d samples",
                    output_channel,
                    len(self.output[output_channel]),
                )

        else:
            played_channels = {}
            channels = []
            awg_channel = 1
            for i in args:
                if isinstance(i, str):
                    if i == '""':
                        continue  # handle instructions like `playWave(1, "", 2, w1);`
                    filename = i.strip('"') + (
                        ".wave" if known_idx is not None else ".csv"
                    )
                    if len(channels) == 0:
                        channels = [awg_channel]
                    _logger.debug(
                        "playing %s on channel(s) %s (%d samples)",
                        filename,
                        channels,
                        len(self.waves[filename]),
                    )

                    for j in channels:
                        # Workaround, equivalent to the [[1, 0], [0, 1]] output gains matrix with "1,2" outputs assignment
                        # for both AWG channels, i.e. 1:1 awg->output channel mapping
                        # TODO(2k): do proper mixing using the actual output gains, and, potentially, simulating HW modulation
                        if j != awg_channel:
                            continue
                        mapped_channel = self.channel_mapping[str(j)]

                        self.last_play_start_samples = len(self.output[mapped_channel])
                        wave_to_play = self.waves[filename]
                        if len(wave_to_play) % self.sample_multiple != 0:
                            raise Exception(
                                f"Wave {filename} has {len(wave_to_play)} samples, which is not divisible by {self.sample_multiple}"
                            )

                        _logger.debug(
                            "Appending to %s, mapped to %s", j, mapped_channel
                        )

                        already_played_ch_len = played_channels.get(mapped_channel)
                        if already_played_ch_len is None:
                            # append wave to the output
                            self.output[mapped_channel] = np.append(
                                self.output[mapped_channel], wave_to_play
                            )
                            played_channels[mapped_channel] = len(wave_to_play)
                        else:
                            # mix-in wave to the output tail
                            assert len(wave_to_play) == already_played_ch_len
                            self.output[mapped_channel] = np.append(
                                self.output[mapped_channel][:-already_played_ch_len],
                                self.output[mapped_channel][-already_played_ch_len:]
                                + wave_to_play,
                            )
                    awg_channel += 1
                    channels = []
                else:
                    channels.append(i)

    def playZero(self, length):
        _logger.debug("playZero called with %d", length)
        _logger.debug("Output keys: %s", list(self.output.keys()))

        if self.device_type == "SHFQA":
            if "QAResult" not in self.output:
                self.output["QAResult"] = np.zeros(0)
            if self.last_play_zero_end is not None:
                last_play_start_samples = self.last_play_zero_end
            else:
                last_play_start_samples = len(self.output["QAResult"])

        if length > 0:
            if self.device_type != "SHFQA" and self.device_type != "SHFSG":
                for i in self.output.keys():
                    self.last_play_start_samples = len(self.output[i])

                    _logger.debug("playing %d zeroes on channel %s", length, i)
                    self.output[i] = np.append(self.output[i], np.zeros(length))

            elif self.device_type == "SHFQA":
                self.last_play_start_samples = last_play_start_samples
                self.last_play_zero_end = last_play_start_samples + length
                _logger.debug(
                    "SHFQA timing: last_play_start_samples=%d len of QA result = %d self.last_play_zero_end = %d",
                    last_play_start_samples,
                    len(self.output["QAResult"]),
                    self.last_play_zero_end,
                )
            elif self.device_type == "SHFSG":
                for i in self.channel_mapping.values():
                    self.last_play_start_samples = len(self.output[i])
                    _logger.debug("playing %d zeroes on channel %s", length, i)
                    self.output[i] = np.append(self.output[i], np.zeros(length))

    def startQA_SHFQA(
        self,
        generators_mask=None,
        integrators_mask=None,
        input_monitor=0,
        result_addr=0,
        trigger=0,
    ):
        _logger.debug(
            "startQA_SHFQA called with arguments (generators_mask=%s,integrators_mask=%s,input_monitor=%s,result_addr=%s,trigger=%s)",
            generators_mask,
            integrators_mask,
            input_monitor,
            result_addr,
            trigger,
        )
        if generators_mask is None:
            generators_mask = self.resolve("QA_GEN_ALL")
        if integrators_mask is None:
            generators_mask = self.resolve("QA_INT_ALL")
        active_generators = [
            generators_mask & self.resolve(f"QA_GEN_{i}") > 0 for i in range(16)
        ]
        active_integrators = [
            integrators_mask & self.resolve(f"QA_INT_{i}") > 0 for i in range(16)
        ]

        _logger.debug("active generators = %s", active_integrators)
        _logger.debug("active_integrators generators = %s", active_integrators)

        if "QAResult" not in self.output:
            self.output["QAResult"] = np.zeros(0)
        current_time = self.last_play_start_samples
        _logger.debug(
            "current time is  %s samples, last_play_zero_end=%d",
            current_time,
            self.last_play_zero_end,
        )
        qa_samples = len(self.output["QAResult"])
        if current_time > qa_samples:
            self.output["QAResult"] = np.append(
                self.output["QAResult"], np.zeros(current_time - qa_samples)
            )

        self.output["QAResult"] = np.append(self.output["QAResult"], np.array([1.0]))
        _logger.debug(
            "Added a 1 to QAResult at %d samples", len(self.output["QAResult"])
        )
        if self.last_play_zero_end is not None:
            if self.last_play_zero_end - 1 > current_time:
                missing_samples = self.last_play_zero_end - 1 - current_time
                self.output["QAResult"] = np.append(
                    self.output["QAResult"], np.zeros(missing_samples)
                )
                _logger.debug(
                    "SHFQA: Advanced QAResult by %d, is now at %d",
                    missing_samples,
                    len(self.output["QAResult"]),
                )

        for i, active in enumerate(active_generators):
            if active:
                output_channel = str(i)
                if output_channel not in self.output:
                    self.output[output_channel] = np.zeros(0) * 1j

                if current_time > len(self.output[output_channel]):
                    missing_samples = current_time - len(self.output[output_channel])

                    self.output[output_channel] = np.append(
                        self.output[output_channel], np.zeros(missing_samples)
                    )
                    _logger.debug(
                        "SHFQA: Advanced channel %s by %d, is now at %d",
                        output_channel,
                        missing_samples,
                        len(self.output[output_channel]),
                    )
                wave_index = self.wave_index[i]

                filename = wave_index["wave_name"] + ".wave"

                wave_to_play = self.waves[filename]

                if len(wave_to_play) % self.sample_multiple != 0:
                    raise Exception(
                        f"Wave {filename} has {len(wave_to_play)} samples, which is not divisible by 16"
                    )

                self.output[output_channel] = np.append(
                    self.output[output_channel], self.scale_factor * wave_to_play
                )
                _logger.debug(
                    "SHFQA: Played wave %s with %d samples on channel %s, is now at %d",
                    filename,
                    len(wave_to_play),
                    output_channel,
                    len(self.output[output_channel]),
                )

                for k, v in self.output.items():
                    _logger.debug("SHFQA:   Output %s has now %d samples", k, len(v))

                _logger.debug(
                    "Channel %s now has %d samples",
                    output_channel,
                    len(self.output[output_channel]),
                )

        for i, active in enumerate(active_integrators):
            if active:
                QA_result_name = "QAResult_" + str(i)
                if QA_result_name not in self.output:
                    self.output[QA_result_name] = np.zeros(0)
                if current_time > len(self.output[QA_result_name]):
                    missing_samples = current_time - len(self.output[QA_result_name])
                    self.output[QA_result_name] = np.append(
                        self.output[QA_result_name], np.zeros(missing_samples)
                    )

                _logger.debug("current time is  %s samples", current_time)
                analysis_time = current_time + self.measurement_delay_samples
                _logger.debug(
                    "analysis_time time is %f samples after delay is added",
                    analysis_time,
                )
                qa_samples = len(self.output[QA_result_name])
                if analysis_time > qa_samples:
                    self.output[QA_result_name] = np.append(
                        self.output[QA_result_name],
                        np.zeros(analysis_time - qa_samples),
                    )
                self.output[QA_result_name] = np.append(
                    self.output[QA_result_name], np.array([1.0])
                )
                _logger.debug(
                    "Added a 1 to %s at %d samples",
                    QA_result_name,
                    len(self.output[QA_result_name]),
                )

    def startQA(self, *args):
        _logger.debug("startQA called with args=%s", args)
        if self.device_type == "SHFQA":
            self.startQA_SHFQA(*args)
        else:
            self.startQA_UHFQA(*args)

    def startQA_UHFQA(
        self,
        weighted_integrator_mask=None,
        monitor=False,
        result_address=0x0,
        trigger=0x0,
    ):
        if weighted_integrator_mask is None:
            weighted_integrator_mask = self.predefined_consts["QA_INT_ALL"]
        _logger.debug(
            "startQA called with integrators mask = %s, input monitor = %s",
            weighted_integrator_mask,
            monitor,
        )
        self.startQAResult()

    def startQAResult(self):
        _logger.debug("startQAResult called")
        if "QAResult" not in self.output:
            self.output["QAResult"] = np.zeros(0)

        current_time = self.last_play_start_samples
        _logger.debug("current time is %f samples", current_time)
        current_time += self.measurement_delay_samples
        _logger.debug("current time is  %f samples after delay is added", current_time)
        qa_samples = len(self.output["QAResult"])
        if current_time > qa_samples:
            self.output["QAResult"] = np.append(
                self.output["QAResult"], np.zeros(current_time - qa_samples)
            )
        self.output["QAResult"] = np.append(self.output["QAResult"], np.array([1.0]))

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
            raise LabOneQException(
                "setSweep() was called, but no sweep was set up with configSweep() earlier"
            ) from e

        freq = start + index * step
        self.setOscFreq(oscillator, freq)

    def setOscFreq(self, oscillator: int, frequency: float):
        key = f"osc{oscillator}_freq"
        if key not in self.output.keys():
            self.output[key] = np.zeros(self.last_play_start_samples)
        else:
            old_frequency = self._oscillator_frequency.get(oscillator, 0.0)
            new_data = np.full(
                self.last_play_start_samples - len(self.output[key]), old_frequency
            )
            self.output[key] = np.append(self.output[key], new_data)

        self._oscillator_frequency[oscillator] = frequency


def find_device(recipe, device_uid):
    for device in recipe["devices"]:
        if device["device_uid"] == device_uid:
            return device
    return None


def analyze_recipe(recipe, waves, sources, wave_indices):

    outputs = {}
    seq_c_devices = {}
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
        compiler_settings = CompilerSettings()
        if "config" in init:
            if "dio_mode" in init["config"]:
                dio_mode = init["config"]["dio_mode"]
                if dio_mode == "zsync_dio":
                    if sampling_rate == 2e9:
                        startup_delay = -compiler_settings.HDAWG_LEAD_PQSC_2GHz
                    else:
                        startup_delay = -compiler_settings.HDAWG_LEAD_PQSC
                elif dio_mode == "hdawg_leader":
                    if sampling_rate == 2e9:
                        startup_delay = -compiler_settings.HDAWG_LEAD_DESKTOP_SETUP_2GHz
                    else:
                        startup_delay = -compiler_settings.HDAWG_LEAD_DESKTOP_SETUP
        if device_type == "UHFQA":
            startup_delay = -compiler_settings.UHFQA_LEAD_PQSC
        elif device_type == "SHFQA":
            startup_delay = -compiler_settings.SHFQA_LEAD_PQSC
        elif device_type == "SHFSG":
            startup_delay = -compiler_settings.SHFSG_LEAD_PQSC

        output_channel_delays = {
            o["channel"]: o.get("port_delay", 0) for o in init.get("outputs", [])
        }
        _logger.debug(
            "output_channel_delays=%s from otuptuts= %s",
            output_channel_delays,
            init.get("outputs", []),
        )

        awg_index = 0
        if "awgs" in init:
            for awg in init["awgs"]:
                seqc = awg["seqc"]
                awg_nr = awg["awg"]
                seq_c_devices[seqc] = {
                    "device_uid": device_uid,
                    "device_type": device_type,
                    "awg_index": awg_index,
                    "signal_type": awg["signal_type"],
                    "measurement_delay_samples": delay,
                    "startup_delay": startup_delay,
                    "sample_multiple": sample_multiple,
                    "sampling_rate": sampling_rate,
                }
                if device_type == "SHFSG" or device_type == "SHFQA":
                    output_channels = [awg_nr]
                else:
                    output_channels = [2 * awg_nr, 2 * awg_nr + 1]

                seq_c_devices[seqc]["output_port_delay"] = output_channel_delays.get(
                    output_channels[0], 0
                )

                channels = [
                    output["channel"] + 1
                    for output in init["outputs"]
                    if output["channel"] in output_channels
                ]
                if len(channels) == 0:
                    channels.append(1)
                outputs[seqc] = channels

                awg_index += 1

    _logger.debug("Outputs: %s", outputs)
    _logger.debug("seqC devices: %s", json.dumps(seq_c_devices))

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

    _logger.debug(seq_c_wave_indices)

    seq_c_descriptors = []
    for name, source in sources.items():
        seq_c_device = seq_c_devices[name]
        seq_c_wave_index = seq_c_wave_indices.get(name, {})
        _logger.debug("seq_c_wave_index for %s is %s", name, seq_c_wave_index)
        channels = outputs[name]
        _logger.debug("Processing descriptor for %s", name)
        if len(channels) == 1:
            _logger.debug("%s is single", name)

            descriptor = SeqCMetadata(
                name,
                source,
                channels,
                seq_c_device["device_uid"],
                seq_c_device["device_type"],
                seq_c_device["measurement_delay_samples"],
                startup_delay=seq_c_device["startup_delay"],
                sampling_rate=seq_c_device["sampling_rate"],
                sample_multiple=seq_c_device["sample_multiple"],
                awg_index=seq_c_device["awg_index"],
                wave_index=seq_c_wave_index,
                output_port_delay=seq_c_device["output_port_delay"],
            )
        else:
            _logger.debug("%s is NOT single; outputs: %s", name, channels)
            descriptor = SeqCMetadata(
                name,
                source,
                channels,
                seq_c_device["device_uid"],
                seq_c_device["device_type"],
                seq_c_device["measurement_delay_samples"],
                startup_delay=seq_c_device["startup_delay"],
                sampling_rate=seq_c_device["sampling_rate"],
                sample_multiple=seq_c_device["sample_multiple"],
                awg_index=seq_c_device["awg_index"],
                wave_index=seq_c_wave_index,
                output_port_delay=seq_c_device["output_port_delay"],
            )
        _logger.debug("Descriptor: %s", descriptor.__dict__)
        seq_c_descriptors.append(descriptor)

    return SimpleNamespace(descriptors=seq_c_descriptors, waves=waves)


def run_single_source(descriptor: SeqCMetadata, waves, max_time, scale_factor):
    _logger.debug("++++++++++++++++++++++++++++++++")
    scale_factor = 1.0
    _logger.debug(
        "Running source %s, using channels %s at frequency %s, scaled by factor %f with measurement delay %d samples",
        descriptor.name,
        descriptor.channels,
        EngNumber(descriptor.sampling_rate),
        scale_factor,
        descriptor.measurement_delay_samples,
    )
    _logger.debug(descriptor.source)

    runtime = SimpleRuntime(
        waves,
        descriptor.channels,
        descriptor.device_uid,
        descriptor.device_type,
        descriptor.sampling_rate,
        descriptor.startup_delay,
        scale_factor,
        descriptor.measurement_delay_samples,
        descriptor.sample_multiple,
        descriptor.awg_index,
        descriptor.wave_index,
        max_time=max_time,
        output_port_delay=descriptor.output_port_delay,
    )

    _logger.debug("-------------------------------")
    parse_seq_c(preprocess_source(descriptor.source), descriptor.name, runtime)
    _logger.debug(runtime.output)
    return runtime


def run_sources(descriptors, waves, max_time, scale_factors=None):
    runtimes = {}
    if scale_factors is None:
        scale_factors = {}
    for descriptor in descriptors:
        _logger.debug("++++++++++++++++++++++++++++++++")
        scale_factor = 1.0
        if descriptor.name in scale_factors:
            scale_factor = scale_factors[descriptor.name]
        runtime = run_single_source(descriptor, waves, max_time, scale_factor)
        runtimes[descriptor.name] = runtime
    return runtimes


def preprocess_source(text):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    return regex.sub(_replacer, text)


def convert_compiler_output_memory(data: CompiledExperiment):
    recipe = data.recipe
    wave_indices = data.wave_indices

    waves = {}
    for wave in data.waves:
        waves[wave["filename"]] = read_wave_bin(wave["samples"])

    sources = {}
    for src in data.src:
        sources[src["filename"]] = src["text"]

    return analyze_recipe(recipe, waves, sources, wave_indices)


def analyze_compiler_output_memory(
    data: CompiledExperiment, max_time=None, scale_factors=None
):
    if scale_factors is None:
        scale_factors = {}
    compiler_output = convert_compiler_output_memory(data)
    runtimes = run_sources(
        compiler_output.descriptors, compiler_output.waves, max_time, scale_factors
    )
    return runtimes


def plot_compiler_result(result, start_time=None, end_time=None):
    import matplotlib.pyplot as plt

    min_x = 1e99
    max_x = -1e99
    spans = {}
    for awgkey, awg in result.items():
        for k, v in awg.output.items():
            times = result[awgkey].times[k]
            start_index = 0
            end_index = len(times)
            if start_time is not None:
                for x in enumerate(result[awgkey].times[k]):
                    if x[1] >= start_time:
                        start_index = x[0]
                        break
            if end_time is not None:
                for x in reversed(list(enumerate(result[awgkey].times[k]))):
                    if x[1] <= end_time:
                        end_index = x[0]
                        break
            spans[(awgkey, k)] = (start_index, end_index)
            if len(times[start_index:end_index]) > 0:
                min_x = min(min_x, min(times[start_index:end_index]))
                max_x = max(max_x, max(times[start_index:end_index]))

    for awgkey, awg in sorted(result.items()):
        for k, v in sorted(awg.output.items()):
            span = spans[(awgkey, k)]
            x = result[awgkey].times[k][span[0] : span[1]]
            plt.rcParams["figure.dpi"] = 250
            fig = plt.figure(figsize=(500, 300), dpi=600, facecolor="w", edgecolor="k")
            fig, ax = plt.subplots()
            ax.set_xlim(min_x - (max_x - min_x) * 0.1, max_x + (max_x - min_x) * 0.1)
            ax.plot(x, v[span[0] : span[1]], "-", label=awgkey + " " + k)
            ax.legend()
            plt.grid(True)


def plot_compiler_output(
    output, start_time=None, end_time=None, max_time=10e-6, scale_factors=None
):
    if scale_factors is None:
        scale_factors = {}
    result = analyze_compiler_output_memory(output, max_time, scale_factors)
    plot_compiler_result(result, start_time=start_time, end_time=end_time)
