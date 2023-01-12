# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
import logging
import math
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
        self, device_type: DeviceType, signal_type, wave_id, length
    ):
        self.add_statement(
            {
                "type": "wave_declaration",
                "device_type": device_type,
                "signal_type": signal_type,
                "wave_id": wave_id,
                "length": length,
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
            raise Exception(f"Empty body for while loop")
        self.add_statement({"type": "do_while", "condition": condition, "body": body})

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

    def add_play_zero_statement(self, num_samples, device_type):
        if isinstance(device_type, str):
            device_type = DeviceType(device_type)

        sample_multiple = device_type.sample_multiple
        if num_samples % sample_multiple != 0:
            _logger.warning(
                "Emitting playZero(%d), which is not divisble by %d, which it should be for %s",
                num_samples,
                sample_multiple,
                device_type,
            )
            raise Exception(
                f"Emitting playZero({num_samples}), which is not divisble by {sample_multiple}, which it should be for {device_type}"
            )
        if num_samples < device_type.min_play_wave:
            raise LabOneQException(
                f"Attempting to emit playZero({num_samples}), which is below the "
                f"minimum waveform length {device_type.min_play_wave} of device "
                f"'{device_type.value}' (sample multiple is {device_type.sample_multiple})"
            )
        statement = {
            "type": "playZero",
            "device_type": device_type,
            "num_samples": int(num_samples),
        }

        max_play_zero = (
            MAX_PLAY_ZERO_HDAWG
            if device_type == DeviceType.HDAWG
            else MAX_PLAY_ZERO_UHFQA
        )
        if statement["num_samples"] > 2 * max_play_zero:
            self._needs_play_zero_counter = True

        statement["max_play_zero"] = max_play_zero
        self.add_statement(statement)

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
            play_zero_text = self._gen_play_zero(
                statement["num_samples"], statement["max_play_zero"]
            )
            self._seq_c_text += play_zero_text

    def _gen_play_zero(self, num_samples, max_play_zero):
        seq_c_text = ""
        _logger.debug(
            "Generating play zero for num_samples %d max_play_zero %d",
            num_samples,
            max_play_zero,
        )
        if num_samples <= max_play_zero:
            seq_c_text += "playZero(" + str(num_samples) + ");\n"
        elif num_samples <= (2 * max_play_zero):
            half_samples = round(num_samples / 2 / 16) * 16
            seq_c_text += "playZero(" + str(half_samples) + ");\n"
            seq_c_text += "playZero(" + str(num_samples - half_samples) + ");\n"
        else:
            num_segments = math.floor(num_samples / max_play_zero)
            rest = num_samples - max_play_zero * num_segments
            if rest < MIN_PLAY_ZERO:
                num_segments -= 1
                extended_rest = rest + max_play_zero
                half_samples = round(extended_rest / 2 / 16) * 16
                seq_c_text += "playZero(" + str(half_samples) + ");\n"
                seq_c_text += "playZero(" + str(extended_rest - half_samples) + ");\n"
            else:
                seq_c_text += "playZero(" + str(rest) + ");\n"
            if num_segments > 1:
                seq_c_text += (
                    self.play_zero_counter_variable_name()
                    + " = "
                    + str(num_segments)
                    + ";\n"
                )
                seq_c_text += "do {\n"
                seq_c_text += "  playZero(" + str(max_play_zero) + ");\n"
                seq_c_text += "  " + self.play_zero_counter_variable_name() + " -= 1;\n"
                seq_c_text += (
                    "} while(" + self.play_zero_counter_variable_name() + ");\n"
                )
            else:
                seq_c_text += "playZero(" + str(max_play_zero) + ");\n"

        return seq_c_text

    def _gen_wave_declaration_placeholder(self, statement) -> str:
        dual_channel = statement["signal_type"] in ["iq", "double", "multi"]
        sig_string = statement["wave_id"]
        length = statement["length"]
        device_type = statement["device_type"]
        assert length >= device_type.min_play_wave

        if dual_channel:
            return (
                f"wave w{sig_string}_i = placeholder({length});\n"
                + f"wave w{sig_string}_q = placeholder({length});\n"
            )
        else:
            return f"wave w{sig_string} = placeholder({length});\n"

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
