# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import hashlib
import logging
import re
import textwrap
from enum import Enum
from typing import Any, Dict, List, Optional

from laboneq.compiler.code_generator.compressor import Run, compressor_core
from laboneq.compiler.common.device_type import DeviceType
from laboneq.core.exceptions import LabOneQException

_logger = logging.getLogger(__name__)


MAX_PLAY_ZERO_HOLD_UHFQA = 131056
MAX_PLAY_ZERO_HOLD_HDAWG = 1048560

MIN_PLAY_ZERO_HOLD = 512 + 128


@functools.lru_cache()
def string_sanitize(input):
    """Sanitize the input string, so it can be safely used as (part of) an identifier
    in seqC."""

    # strip non-ascii characters
    s = input.encode("ascii", "ignore").decode()

    if s == "":
        s = "_"

    # only allowed characters are alphanumeric and underscore
    s = re.sub(r"[^\w\d]", "_", s)

    # names must not start with a digit
    if s[0].isdigit():
        s = "_" + s

    if s != input:
        s = f"{s}_{hashlib.md5(input.encode()).hexdigest()[:4]}"

    return s


SeqCStatement = Dict[str, Any]


class SeqCGenerator:
    def __init__(self):
        self._seq_c_text = ""
        self._statements: List[SeqCStatement] = []

    def num_statements(self):
        return len(self._statements)

    def num_noncomment_statements(self):
        retval = 0
        for statement in self._statements:
            if statement["type"] != "comment":
                retval += 1
        return retval

    def clear(self):
        self._statements.clear()

    def append_statements_from(self, seq_c_generator: SeqCGenerator):
        self._statements.extend(seq_c_generator._statements)

    def add_statement(self, statement: SeqCStatement):
        self._statements.append(statement)

    def add_comment(self, comment_text):
        statement = {"type": "comment", "text": comment_text}
        self.add_statement(statement)

    def add_function_call_statement(self, name, args=[], assign_to=None):
        statement = {"type": "generic_statement", "function": name}
        if args is not None and len(args) > 0:
            statement["args"] = args
        if assign_to is not None:
            statement["assign_to"] = assign_to
        self.add_statement(statement)

    def add_wave_declaration(
        self,
        device_type: DeviceType,
        signal_type,
        wave_id,
        length,
        has_marker1,
        has_marker2,
    ):
        self.add_statement(
            {
                "type": "wave_declaration",
                "device_type": device_type,
                "signal_type": signal_type,
                "wave_id": wave_id,
                "length": length,
                "has_marker1": has_marker1,
                "has_marker2": has_marker2,
            }
        )

    def estimate_complexity(self):
        """Calculate a rough estimate for the complexity (~nr of instructions)

        The point here is not to be accurate about every statement, but to correctly
        gauge the size of loops etc."""
        score = 0
        for s in self._statements:
            default = 1
            if s.get("type") in ["do_while", "repeat"]:
                default = 10
            score += s.get("complexity", default)
        return score

    def add_repeat(self, num_repeats, body: SeqCGenerator):
        assert body is not None
        complexity = body.estimate_complexity() + 2  # penalty for loop overhead
        self.add_statement(
            {
                "type": "repeat",
                "num_repeats": num_repeats,
                "body": body,
                "complexity": complexity,
            }
        )

    def add_do_while(self, condition, body: SeqCGenerator):
        assert body is not None
        complexity = body.estimate_complexity() + 5  # penalty for loop overhead
        self.add_statement(
            {
                "type": "do_while",
                "condition": condition,
                "body": body,
                "complexity": complexity,
            }
        )

    def add_function_def(self, text):
        self.add_statement({"type": "function_def", "text": text})

    def add_variable_declaration(self, variable_name, initial_value=None):
        statement = {"type": "variable_declaration", "variable_name": variable_name}
        if initial_value is not None:
            statement["initial_value"] = initial_value
        self.add_statement(statement)

    def add_variable_assignment(self, variable_name, value):
        statement = {"type": "variable_assignment", "variable_name": variable_name}
        statement["value"] = value
        self.add_statement(statement)

    def add_variable_increment(self, variable_name, value):
        statement = {"type": "variable_increment", "variable_name": variable_name}
        statement["value"] = value
        self.add_statement(statement)

    def add_assign_wave_index_statement(
        self, device_type: DeviceType, signal_type, wave_id, wave_index, channel
    ):
        self.add_statement(
            {
                "type": "assignWaveIndex",
                "device_type": device_type,
                "signal_type": signal_type,
                "wave_id": wave_id,
                "wave_index": wave_index,
                "channel": channel,
            }
        )

    def add_play_wave_statement(
        self, device_type: DeviceType, signal_type, wave_id, channel
    ):
        self.add_statement(
            {
                "type": "playWave",
                "device_type": device_type,
                "signal_type": signal_type,
                "wave_id": wave_id,
                "channel": channel,
            }
        )

    def add_command_table_execution(self, ct_index, latency=None, comment=""):
        self.add_statement(
            {
                "type": "executeTableEntry",
                "table_index": ct_index,
                "latency": latency,
                "comment": comment,
            }
        )

    def _add_play_zero_or_hold(
        self,
        num_samples,
        device_type,
        fname: str,
        deferred_calls: Optional[SeqCGenerator] = None,
    ):
        if deferred_calls is None:
            deferred_calls = SeqCGenerator()

        if isinstance(device_type, str):
            device_type = DeviceType(device_type)

        sample_multiple = device_type.sample_multiple
        if num_samples % sample_multiple != 0:
            raise Exception(
                f"Emitting {fname}({num_samples}), which is not divisible by {sample_multiple}, which it should be for {device_type}"
            )
        if num_samples < device_type.min_play_wave:
            raise LabOneQException(
                f"Attempting to emit {fname}({num_samples}), which is below the "
                f"minimum waveform length {device_type.min_play_wave} of device "
                f"'{device_type.value}' (sample multiple is {device_type.sample_multiple})"
            )
        max_play_fun = (
            MAX_PLAY_ZERO_HOLD_HDAWG
            if device_type == DeviceType.HDAWG
            else MAX_PLAY_ZERO_HOLD_UHFQA
        )

        def statement_factory(samples):
            self.add_statement(
                {
                    "type": f"{fname}",
                    "device_type": device_type,
                    "num_samples": samples,
                }
            )

        def flush_deferred_calls():
            self.append_statements_from(deferred_calls)
            deferred_calls.clear()

        if num_samples <= max_play_fun:
            statement_factory(num_samples)
            flush_deferred_calls()
        elif num_samples <= 2 * max_play_fun:
            # split in the middle
            half_samples = (num_samples // 2 // 16) * 16
            statement_factory(half_samples)
            flush_deferred_calls()
            statement_factory(num_samples - half_samples)
        else:  # non-unrolled loop
            num_segments, rest = divmod(num_samples, max_play_fun)
            if 0 < rest < MIN_PLAY_ZERO_HOLD:
                chunk = (max_play_fun // 2 // 16) * 16
                statement_factory(chunk)
                flush_deferred_calls()
                num_samples -= chunk
                num_segments, rest = divmod(num_samples, max_play_fun)
            if rest > 0:
                statement_factory(rest)
                flush_deferred_calls()
            if deferred_calls.num_statements() > 0:
                statement_factory(max_play_fun)
                flush_deferred_calls()
                num_segments -= 1
            if num_segments == 1:
                statement_factory(max_play_fun)
                return

            loop_body = SeqCGenerator()
            loop_body.add_statement(
                {
                    "type": f"{fname}",
                    "device_type": device_type,
                    "num_samples": max_play_fun,
                }
            )
            self.add_repeat(num_segments, loop_body)

    def add_play_zero_statement(
        self,
        num_samples,
        device_type,
        deferred_calls: Optional[SeqCGenerator] = None,
    ):
        """Add a playZero command

        If the requested number of samples exceeds the allowed number of samples for
        a single playZero, a tight loop of playZeros will be emitted.

        If deferred_calls is passed, the deferred function calls are cleared in the
        context of the added playZero(s). The passed list will be drained.
        """
        self._add_play_zero_or_hold(
            num_samples, device_type, "playZero", deferred_calls
        )

    def add_play_hold_statement(
        self,
        num_samples,
        device_type,
        deferred_calls: Optional[SeqCGenerator] = None,
    ):
        """Add a playHold command

        If the requested number of samples exceeds the allowed number of samples for
        a single playHold, a tight loop of playZeros will be emitted.

        If deferred_calls is passed, the deferred function calls are cleared in the
        context of the added playHold(s). The passed list will be drained.
        """
        self._add_play_zero_or_hold(
            num_samples, device_type, "playHold", deferred_calls
        )

    def generate_seq_c(self):
        self._seq_c_text = ""
        for statement in self._statements:
            _logger.debug("processing statement %s", statement)
            self.emit_statement(statement)
        return self._seq_c_text

    def emit_statement(self, statement: SeqCStatement):
        if statement["type"] == "generic_statement":
            if "assign_to" in statement:
                self._seq_c_text += f"{statement['assign_to']} = "
            self._seq_c_text += statement["function"] + "("
            if "args" in statement:
                is_first = True
                for arg in statement["args"]:
                    if not is_first:
                        self._seq_c_text += ","
                    else:
                        is_first = False
                    self._seq_c_text += str(arg)
            self._seq_c_text += ");\n"

        elif statement["type"] == "wave_declaration":
            if statement["device_type"].supports_binary_waves:
                self._seq_c_text += self._gen_wave_declaration_placeholder(statement)
        elif statement["type"] == "function_def":
            self._seq_c_text += statement["text"]
        elif statement["type"] == "variable_declaration":
            self._seq_c_text += "var " + statement["variable_name"]
            if "initial_value" in statement:
                self._seq_c_text += " = " + str(statement["initial_value"]) + ";\n"
            else:
                self._seq_c_text += ";\n"
        elif statement["type"] == "variable_assignment":
            self._seq_c_text += statement["variable_name"]
            self._seq_c_text += " = " + str(statement["value"]) + ";\n"
        elif statement["type"] == "variable_increment":
            self._seq_c_text += statement["variable_name"]
            self._seq_c_text += " += " + str(statement["value"]) + ";\n"

        elif statement["type"] == "do_while":
            self._seq_c_text += "do {\n"
            self._seq_c_text += textwrap.indent(
                statement["body"].generate_seq_c(), "  "
            )
            self._seq_c_text += "}\nwhile(" + statement["condition"] + ");\n"

        elif statement["type"] == "repeat":
            self._seq_c_text += f"repeat ({statement['num_repeats']}) {{\n"
            self._seq_c_text += textwrap.indent(
                statement["body"].generate_seq_c(), "  "
            )
            self._seq_c_text += "}\n"

        elif statement["type"] == "assignWaveIndex":
            wave_channels = self._build_wave_channel_assignment(statement)
            self._seq_c_text += (
                f'assignWaveIndex({wave_channels},{statement["wave_index"]});\n'
            )
        elif statement["type"] == "playWave":
            wave_channels = self._build_wave_channel_assignment(statement)
            self._seq_c_text += f"playWave({wave_channels});\n"
        elif statement["type"] == "executeTableEntry":
            self._seq_c_text += f"executeTableEntry({statement['table_index']}"
            latency = statement.get("latency", None)
            if latency is not None:
                self._seq_c_text += f", {latency}"
            self._seq_c_text += ");"
            if statement["comment"] != "":
                self._seq_c_text += f"  // {statement['comment']}"
            self._seq_c_text += "\n"
        elif statement["type"] == "comment":
            self._seq_c_text += "/* " + statement["text"] + " */\n"
        elif statement["type"] == "playZero":
            self._seq_c_text += f"playZero({statement['num_samples']});\n"
        elif statement["type"] == "playHold":
            self._seq_c_text += f"playHold({statement['num_samples']});\n"

    def _gen_wave_declaration_placeholder(self, statement: SeqCStatement) -> str:
        dual_channel = statement["signal_type"] in ["iq", "double", "multi"]
        sig_string = statement["wave_id"]
        length = statement["length"]
        device_type = statement["device_type"]
        assert length >= device_type.min_play_wave
        makers_declaration1 = ""
        makers_declaration2 = ""
        if statement["has_marker1"]:
            makers_declaration1 = ",true"
        if statement["has_marker2"]:
            makers_declaration2 = ",true"

        if dual_channel:

            return (
                f"wave w{sig_string}_i = placeholder({length}{makers_declaration1});\n"
                + f"wave w{sig_string}_q = placeholder({length}{makers_declaration2});\n"
            )
        else:
            return f"wave w{sig_string} = placeholder({length}{makers_declaration1});\n"

    def _build_wave_channel_assignment(self, statement) -> str:
        dual_channel = statement["signal_type"] in ["iq", "double", "multi"]
        sig_string = statement["wave_id"]
        channel = statement.get("channel")
        if dual_channel and statement["device_type"].supports_digital_iq_modulation:
            return f"1,2,w{sig_string}_i,1,2,w{sig_string}_q"
        elif dual_channel:
            return f"w{sig_string}_i,w{sig_string}_q"
        elif channel == 1:
            return f'1,"",2,w{sig_string}'
        else:
            return f"w{sig_string}"

    def _key(self):
        tuple_list = []
        for statement in self._statements:
            single_items = []
            for k, v in statement.items():
                if v is not None:
                    value = v
                    if isinstance(v, SeqCGenerator):
                        value = ("hash", hash(v))
                        single_items.append((k, value))
                    elif (
                        isinstance(v, int)
                        or isinstance(v, float)
                        or isinstance(v, str)
                        or isinstance(v, Enum)
                    ):
                        single_items.append((k, value))
                    else:
                        single_items.append((k, tuple(value)))
            tuple_list.append(tuple(single_items))
        return tuple(tuple_list)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if isinstance(other, SeqCGenerator):
            return self._key() == other._key()
        return NotImplemented

    def __repr__(self):
        retval = "SeqCGenerator("
        for statement in self._statements:
            retval += str(statement) + ","

        retval += ")"
        return retval

    def compressed(self):
        statement_hashes = [hash(k) for k in self._key()]
        statement_by_hash = {}
        for h, s in zip(statement_hashes, self._statements):
            if h in statement_by_hash and statement_by_hash[h] != s:
                _logger.warning("hash collision detected, skipping code compression")
                return self
            statement_by_hash[h] = s

        def cost_function(r: Run):
            complexity = sum(statement_by_hash[h].get("complexity", 1) for h in r.word)
            return -(r.count - 1) * complexity + 2

        compressed_statements = compressor_core(statement_hashes, cost_function)
        retval = SeqCGenerator()
        for cs in compressed_statements:
            if isinstance(cs, Run):
                body = SeqCGenerator()
                for statement_hash in cs.word:
                    body.add_statement(statement_by_hash[statement_hash])
                retval.add_repeat(cs.count, body)
            else:
                retval.add_statement(statement_by_hash[cs])

        return retval


def merge_generators(generators, compress=True) -> SeqCGenerator:
    generator_hashes = [hash(g) for g in generators]
    # todo: cannot check for hash collisions, SeqCGenerator.__eq__ also uses hash
    generator_by_hash = {h: g for h, g in zip(generator_hashes, generators)}

    retval = SeqCGenerator()
    if compress:

        def cost_function(r: Run):
            complexity = sum(generator_by_hash[h].estimate_complexity() for h in r.word)
            return -(r.count - 1) * complexity + 2

        compressed_generators = compressor_core(generator_hashes, cost_function)

        for cg in compressed_generators:
            if isinstance(cg, Run):
                if len(cg.word) == 1:
                    body = generator_by_hash[cg.word[0]]
                else:
                    body = SeqCGenerator()
                    for gen_hash in cg.word:
                        body.append_statements_from(generator_by_hash[gen_hash])
                retval.add_repeat(cg.count, body.compressed())
            else:
                retval.append_statements_from(generator_by_hash[cg])

        # optional: we might add a 2nd pass here on the merged generator, finding patterns
        # that partially span across multiple of the original parts.
        # retval = retval.compressed()
    else:
        for g in generators:
            retval.append_statements_from(g)

    return retval
