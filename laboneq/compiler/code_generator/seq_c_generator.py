# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
import logging
import re
from enum import Enum

from laboneq.compiler.common.device_type import DeviceType
from laboneq.core.exceptions import LabOneQException

_logger = logging.getLogger(__name__)


MAX_PLAY_ZERO_UHFQA = 131056
MAX_PLAY_ZERO_HDAWG = 1048560

MIN_PLAY_ZERO = 512 + 128
PLAY_ZERO_COUNTER_VARIABLE = "play_zero_count"


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


class SeqCGenerator:
    def __init__(self):
        self._seq_c_text = ""
        self._statements = []
        self._needs_play_zero_counter = False

    def num_statements(self):
        return len(self._statements)

    def num_noncomment_statements(self):
        retval = 0
        for statement in self._statements:
            if statement["type"] != "comment":
                retval += 1
        return retval

    def append_statements_from(self, seq_c_generator):
        self._statements.extend(seq_c_generator._statements)
        if seq_c_generator.needs_play_zero_counter():
            self._needs_play_zero_counter = True

    def add_statement(self, statement):
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

    def add_countdown_loop(self, variable_name, num_repeats, body):
        if body is None:
            raise Exception(f"Empty body for variable_name {variable_name}")
        self.add_statement(
            {
                "type": "countdown_loop",
                "variable_name": variable_name,
                "num_repeats": num_repeats,
                "body": body,
            }
        )

    def add_do_while(self, condition, body):
        if body is None:
            raise Exception("Empty body for while loop")
        self.add_statement({"type": "do_while", "condition": condition, "body": body})

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

    def add_command_table_execution(self, ct_index, comment=""):
        self.add_statement(
            {
                "type": "executeTableEntry",
                "table_index": ct_index,
                "comment": comment,
            }
        )

    def add_play_zero_statement(self, num_samples, device_type, deferred_calls=None):
        """Add a playZero command

        If the requested number of samples exceeds the allowed number of samples for
        a single playZero, a tight loop of playZeros will be emitted.

        If deferred_calls is passed, the deferred function calls are cleared in the
        context of the added playZero(s). The passed list will be drained.
        """
        if deferred_calls is None:
            deferred_calls = []

        assert isinstance(deferred_calls, list)

        if isinstance(device_type, str):
            device_type = DeviceType(device_type)

        sample_multiple = device_type.sample_multiple
        if num_samples % sample_multiple != 0:
            raise Exception(
                f"Emitting playZero({num_samples}), which is not divisble by {sample_multiple}, which it should be for {device_type}"
            )
        if num_samples < device_type.min_play_wave:
            raise LabOneQException(
                f"Attempting to emit playZero({num_samples}), which is below the "
                f"minimum waveform length {device_type.min_play_wave} of device "
                f"'{device_type.value}' (sample multiple is {device_type.sample_multiple})"
            )
        max_play_zero = (
            MAX_PLAY_ZERO_HDAWG
            if device_type == DeviceType.HDAWG
            else MAX_PLAY_ZERO_UHFQA
        )

        def statement_factory(samples):
            self.add_statement(
                {
                    "type": "playZero",
                    "device_type": device_type,
                    "num_samples": samples,
                }
            )

        def clear_deferred_calls():
            for call in deferred_calls:
                self.add_function_call_statement(call["name"], call["args"])
            del deferred_calls[:]

        if num_samples <= max_play_zero:
            statement_factory(num_samples)
            clear_deferred_calls()
        elif num_samples <= 2 * max_play_zero:
            # split in the middle
            half_samples = (num_samples // 2 // 16) * 16
            statement_factory(half_samples)
            clear_deferred_calls()
            statement_factory(num_samples - half_samples)
        else:  # non-unrolled loop
            self._needs_play_zero_counter = True
            num_segments, rest = divmod(num_samples, max_play_zero)
            if 0 < rest < MIN_PLAY_ZERO:
                chunk = (max_play_zero // 2 // 16) * 16
                statement_factory(chunk)
                clear_deferred_calls()
                num_samples -= chunk
                num_segments, rest = divmod(num_samples, max_play_zero)
            if rest > 0:
                statement_factory(rest)
                clear_deferred_calls()
            if deferred_calls:
                statement_factory(max_play_zero)
                clear_deferred_calls()
                num_segments -= 1
            if num_segments == 1:
                statement_factory(max_play_zero)
                return

            inner_loop = SeqCGenerator()
            inner_loop.add_statement(
                {
                    "type": "playZero",
                    "device_type": device_type,
                    "num_samples": max_play_zero,
                }
            )
            self.add_countdown_loop(
                self.play_zero_counter_variable_name(), num_segments, inner_loop
            )

    def generate_seq_c(self):
        self._seq_c_text = ""
        for statement in self._statements:
            logging.getLogger(__name__).debug("processing statement %s", statement)
            self.emit_statement(statement)
        return self._seq_c_text

    def emit_statement(self, statement):
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

        elif statement["type"] == "countdown_loop":
            self._seq_c_text += (
                f"{statement['variable_name']} = {statement['num_repeats']};\n"
            )
            self._seq_c_text += "do {\n"
            for line in statement["body"].generate_seq_c().splitlines():
                self._seq_c_text += "  " + line + "\n"
            self._seq_c_text += "  " + statement["variable_name"] + " -= 1;\n"
            self._seq_c_text += "}\nwhile(" + statement["variable_name"] + ");\n"

        elif statement["type"] == "do_while":
            self._seq_c_text += "do {\n"
            for line in statement["body"].generate_seq_c().splitlines():
                self._seq_c_text += "  " + line + "\n"
            self._seq_c_text += "}\nwhile(" + statement["condition"] + ");\n"

        elif statement["type"] == "assignWaveIndex":
            wave_channels = self._build_wave_channel_assignment(statement)
            self._seq_c_text += (
                f'assignWaveIndex({wave_channels},{statement["wave_index"]});\n'
            )
        elif statement["type"] == "playWave":
            wave_channels = self._build_wave_channel_assignment(statement)
            self._seq_c_text += f"playWave({wave_channels});\n"
        elif statement["type"] == "executeTableEntry":
            self._seq_c_text += f"executeTableEntry({statement['table_index']});"
            if statement["comment"] != "":
                self._seq_c_text += f"  // {statement['comment']}"
            self._seq_c_text += "\n"
        elif statement["type"] == "comment":
            self._seq_c_text += "/* " + statement["text"] + " */\n"
        elif statement["type"] == "playZero":
            self._seq_c_text += f"playZero({statement['num_samples']});\n"

    def _gen_wave_declaration_placeholder(self, statement) -> str:
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

    def __key(self):
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
                    elif "type" in v and v["type"] == "comment":
                        pass
                    else:
                        single_items.append((k, tuple(value)))
            if statement["type"] != "comment":
                tuple_list.append(tuple(single_items))
        return tuple(tuple_list)

    def needs_play_zero_counter(self):
        if self._needs_play_zero_counter:
            return True
        else:
            for statement in self._statements:
                for child in statement.values():
                    if isinstance(child, SeqCGenerator):
                        if child.needs_play_zero_counter():
                            return True
        return False

    def play_zero_counter_variable_name(self):
        return PLAY_ZERO_COUNTER_VARIABLE

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, SeqCGenerator):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        retval = "SeqCGenerator("
        for statement in self._statements:
            retval += str(statement) + ","

        retval += ")"
        return retval
