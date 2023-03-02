# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from laboneq.compiler.code_generator.compressor import compress_generators
from laboneq.compiler.code_generator.seq_c_generator import (
    SeqCGenerator,
    string_sanitize,
)
from laboneq.compiler.common.device_type import DeviceType
from laboneq.core.exceptions import LabOneQException

if TYPE_CHECKING:
    from laboneq.compiler.code_generator.command_table_tracker import (
        CommandTableTracker,
    )
    from laboneq.compiler.code_generator.seqc_tracker import SeqCTracker
    from laboneq.compiler.code_generator.signatures import WaveformSignature
    from laboneq.compiler.code_generator.wave_index_tracker import WaveIndexTracker
    from laboneq.compiler.common.awg_info import AWGInfo


_logger = logging.getLogger(__name__)


def sort_events(events):
    later = {
        "sequencer_start": -100,
        "initial_reset_phase": -4,
        "LOOP_STEP_START": -3,
        "PUSH_LOOP": -2,
        "reset_phase": -1,
        "reset_precompensation_filters": -1,
        "acquire": 1,
        "ITERATE": 2,
    }
    sampled_event_list = sorted(
        events,
        key=lambda x: later[x["signature"]] if x["signature"] in later else 0,
    )

    return sampled_event_list


@dataclass
class FeedbackConnection:
    acquire: Optional[str]
    drive: Set[str] = field(default_factory=set)


def generate_if_else_tree(
    iterations: int, variable: str, step_factory: Callable[[int], str]
) -> List[str]:
    def if_level(base, bit):
        if bit == 0:
            return [f"{step_factory(base)}"]
        n_bit = bit - 1
        n_base = base + (1 << n_bit)
        if n_base < iterations:
            return [
                f"if ({variable} & 0b{(1 << n_bit):b}) {{  // {variable} >= {n_base}",
                *[f"  {l}" for l in if_level(n_base, n_bit)],
                f"}} else {{  // {variable} < {n_base}",
                *[f"  {l}" for l in if_level(base, n_bit)],
                "}",
            ]
        else:
            return if_level(base, n_bit)

    if iterations == 0:
        return []
    start_bit = iterations.bit_length()
    return if_level(0, start_bit)


class SampledEventHandler:
    def __init__(
        self,
        seqc_tracker: SeqCTracker,
        command_table_tracker: CommandTableTracker,
        function_defs_generator: SeqCGenerator,
        declarations_generator: SeqCGenerator,
        wave_indices: WaveIndexTracker,
        feedback_connections: Dict[str, FeedbackConnection],
        awg: AWGInfo,
        device_type: DeviceType,
        channels: List[int],
        sampled_signatures: Dict[str, Dict[WaveformSignature, Dict]],
        use_command_table: bool,
        emit_timing_comments: bool,
    ):
        self.seqc_tracker = seqc_tracker
        self.command_table_tracker = command_table_tracker
        self.function_defs_generator = function_defs_generator
        self.declarations_generator = declarations_generator
        self.wave_indices = wave_indices
        self.feedback_connections = feedback_connections
        self.awg = awg
        self.device_type = device_type
        self.channels = channels
        self.sampled_signatures = sampled_signatures
        self.use_command_table = use_command_table
        self.emit_timing_comments = emit_timing_comments

        self.sampled_event_list = None
        self.declared_variables = set()
        self.loop_stack = []
        self.last_event = None
        self.match_parent_event = None
        self.command_table_match_offset = None
        self.match_command_table_entries = {}

    def handle_playwave(
        self,
        sampled_event,
    ):
        signature = sampled_event["playback_signature"]
        signal_id = sampled_event["signal_id"]
        state = signature.state

        match_statement_active = self.match_parent_event is not None
        handle = (
            self.match_parent_event["handle"]
            if self.match_parent_event is not None
            else None
        )

        assert (state is not None) == match_statement_active

        if not self.use_command_table and state is not None:
            raise LabOneQException(
                f"Found match/case statement for handle {handle} on unsupported device."
            )

        if signature.hw_oscillator is not None and not self.use_command_table:
            raise LabOneQException(
                "HW oscillator switching only possible in experimental "
                "command-table mode"
            )

        if not (
            signal_id in self.sampled_signatures
            and signature.waveform in self.sampled_signatures[signal_id]
        ):
            return False

        _logger.debug(
            "  Found matching signature %s for event %s",
            signature,
            sampled_event,
        )
        if state is None:
            # Playzeros were already added for match event
            self.seqc_tracker.add_required_playzeros(sampled_event)
            self.seqc_tracker.add_timing_comment(sampled_event["end"])

        play_wave_channel = None
        if len(self.channels) > 0:
            play_wave_channel = self.channels[0] % 2

        signal_type_for_wave_index = (
            self.awg.signal_type.value
            if self.device_type.supports_binary_waves
            else "csv"
            # Include CSV waves into the index to keep track of waves-AWG mapping
        )
        sig_string = signature.waveform.signature_string()
        if (
            all(p.pulse is None for p in signature.waveform.pulses)
            and self.use_command_table
        ):
            # all-zero pulse is played via play-zero command table entry
            wave_index = None
        else:
            wave_index = self.wave_indices.lookup_index_by_wave_id(sig_string)
            if wave_index is None:
                wave_index = self.wave_indices.create_index_for_wave(
                    sig_string, signal_type_for_wave_index
                )
                if wave_index is not None:
                    self.seqc_tracker.add_assign_wave_index_statement(
                        self.device_type,
                        self.awg.signal_type.value,
                        sig_string,
                        wave_index,
                        play_wave_channel,
                    )

        if not match_statement_active:
            if self.use_command_table:
                ct_index = self.command_table_tracker.lookup_index_by_signature(
                    signature
                )
                if ct_index is None:
                    ct_index = self.command_table_tracker.create_entry(
                        signature, wave_index
                    )
                comment = sig_string
                if signature.hw_oscillator is not None:
                    comment += f", osc={signature.hw_oscillator}"
                self.seqc_tracker.add_command_table_execution(ct_index, comment=comment)
            else:
                self.seqc_tracker.add_play_wave_statement(
                    self.device_type,
                    self.awg.signal_type.value,
                    sig_string,
                    play_wave_channel,
                )
            self.seqc_tracker.clear_deferred_function_calls()
            self.seqc_tracker.current_time = sampled_event["end"]
        else:
            assert self.use_command_table
            if state in self.match_command_table_entries:
                if self.match_command_table_entries[state] != (
                    signature,
                    wave_index,
                    sampled_event["start"] - self.match_parent_event["start"],
                ):
                    raise LabOneQException(
                        f"Duplicate state {state} with different pulses for handle "
                        f"{self.match_parent_event['handle']} found."
                    )
            else:
                self.match_command_table_entries[state] = (
                    signature,
                    wave_index,
                    sampled_event["start"] - self.match_parent_event["start"],
                )
            self.feedback_connections.setdefault(
                self.match_parent_event["handle"], FeedbackConnection(None)
            ).drive.add(signal_id)
        return True

    def handle_acquire(self, sampled_event):
        _logger.debug("  Processing ACQUIRE EVENT %s", sampled_event)

        args = [
            "QA_INT_ALL",
            "1" if "RAW" in sampled_event["acquisition_type"] else "0",
        ]
        start = sampled_event["start"]

        if start > self.seqc_tracker.current_time:
            self.seqc_tracker.add_required_playzeros(sampled_event)

            _logger.debug("  Deferring function call for %s", sampled_event)
            self.seqc_tracker.add_function_call_statement(
                name="startQA", args=args, deferred=True
            )
        else:
            skip = False
            if self.last_event is not None:
                if (
                    self.last_event["signature"] == "acquire"
                    and self.last_event["start"] == start
                ):
                    skip = True
                    _logger.debug(
                        "Skipping acquire event %s because last event was also acquire: %s",
                        sampled_event,
                        self.last_event,
                    )
            if not skip:
                self.seqc_tracker.add_function_call_statement(
                    "startQA", args, deferred=False
                )
        for h in sampled_event["acquire_handles"]:
            self._add_feedback_connection(h, sampled_event["signal_id"])

    def handle_qa_event(self, sampled_event):
        _logger.debug("  Processing QA_EVENT %s", sampled_event)
        generator_channels = set()
        for play_event in sampled_event["play_events"]:
            _logger.debug("  play_event %s", play_event)
            play_signature = play_event["playback_signature"].waveform
            if "signal_id" in play_event:
                signal_id = play_event["signal_id"]
                if play_signature in self.sampled_signatures[signal_id]:
                    _logger.debug(
                        "  Found matching signature %s for event %s",
                        play_signature,
                        play_event,
                    )
                    current_signal_obj = next(
                        signal_obj
                        for signal_obj in self.awg.signals
                        if signal_obj.id == signal_id
                    )
                    generator_channels.update(current_signal_obj.channels)
                    sig_string = play_signature.signature_string()

                    self.wave_indices.add_numbered_wave(
                        sig_string,
                        "complex",
                        current_signal_obj.channels[0],
                    )

        integration_channels = [
            event["channels"] for event in sampled_event["acquire_events"]
        ]

        integration_channels = [
            item for sublist in integration_channels for item in sublist
        ]

        if len(integration_channels) > 0:

            integrator_mask = "|".join(
                map(lambda x: "QA_INT_" + str(x), integration_channels)
            )
        else:
            integrator_mask = "QA_INT_NONE"

        if len(generator_channels) > 0:
            generator_mask = "|".join(
                map(lambda x: "QA_GEN_" + str(x), generator_channels)
            )
        else:
            generator_mask = "QA_GEN_NONE"

        if "spectroscopy" in sampled_event["acquisition_type"]:
            args = [0, 0, 0, 0, 1]
        else:
            args = [
                generator_mask,
                integrator_mask,
                "1" if "RAW" in sampled_event["acquisition_type"] else "0",
            ]

        self.seqc_tracker.add_required_playzeros(sampled_event)

        if sampled_event["end"] > self.seqc_tracker.current_time:
            self.seqc_tracker.add_timing_comment(sampled_event["end"])

        self.seqc_tracker.add_function_call_statement("startQA", args, deferred=True)
        if "spectroscopy" in sampled_event["acquisition_type"]:
            self.seqc_tracker.add_function_call_statement(
                "setTrigger", [0], deferred=True
            )

        for ev in sampled_event["acquire_events"]:
            for h in ev["acquire_handles"]:
                self._add_feedback_connection(h, ev["signal_id"])

    def handle_reset_precompensation_filters(self, sampled_event):
        if sampled_event["signal_id"] not in (s.id for s in self.awg.signals):
            return

        _logger.debug(
            "  Processing RESET PRECOMPENSATION FILTERS event %s",
            sampled_event,
        )
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.add_function_call_statement(
            name="setPrecompClear", args=[1], deferred=False
        )
        self.seqc_tracker.add_function_call_statement(
            name="setPrecompClear", args=[0], deferred=True
        )

    def handle_reset_phase(self, sampled_event):
        # If multiple phase reset events are scheduled at the same time,
        # only process the *last* one. This way, `reset_phase` takes
        # precedence.
        # TODO (PW): Remove this check, once we no longer force oscillator
        # resets at the start of the sequence.
        last_reset = [
            event
            for event in self.sampled_event_list
            if event["signature"] in ("reset_phase", "initial_reset_phase")
        ][-1]
        if last_reset is not sampled_event:
            return

        _logger.debug("  Processing RESET PHASE event %s", sampled_event)
        signature = sampled_event["signature"]
        start = sampled_event["start"]
        if signature == "initial_reset_phase":
            if start > self.seqc_tracker.current_time:
                self.seqc_tracker.add_required_playzeros(sampled_event)
            if self.awg.device_type.supports_reset_osc_phase:
                # Hack: we do not defer, and emit as early as possible.
                # This way it is hidden in the lead time.
                self.seqc_tracker.add_function_call_statement("resetOscPhase")
        elif (
            signature == "reset_phase" and self.awg.device_type.supports_reset_osc_phase
        ):
            self.seqc_tracker.add_required_playzeros(sampled_event)
            self.seqc_tracker.add_function_call_statement(
                "resetOscPhase", deferred=True
            )

    def handle_set_oscillator_frequency(self, sampled_event):
        if self.device_type == DeviceType.HDAWG:
            self._handle_set_oscillator_frequency_hdawg(sampled_event)
        else:
            self._handle_set_oscillator_frequency_shf(sampled_event)

    def _handle_set_oscillator_frequency_hdawg(self, sampled_event):
        iteration = sampled_event["iteration"]
        parameter_name = sampled_event["parameter_name"]
        param_stem = string_sanitize(parameter_name)
        counter_variable_name = string_sanitize(f"index_{parameter_name}")
        if iteration == 0:
            iterations = sampled_event["iterations"]
            if iterations > 512:
                raise LabOneQException(
                    "HDAWG can only handle RT frequency sweeps up to 512 steps."
                )
            steps = "\n  ".join(
                generate_if_else_tree(
                    iterations=iterations,
                    variable=f"arg_{param_stem}",
                    step_factory=lambda i: f"setDouble(osc_node_{param_stem}, {sampled_event['start_frequency'] + sampled_event['step_frequency'] * i});",
                )
            )
            self.function_defs_generator.add_function_def(
                f"void set_{param_stem}(var arg_{param_stem}) {{\n"
                f'  string osc_node_{param_stem} = "oscs/0/freq";\n'
                f"  {steps}\n"
                f"}}\n"
            )
            self.declarations_generator.add_variable_declaration(
                counter_variable_name, 0
            )
            self.seqc_tracker.add_variable_assignment(counter_variable_name, 0)
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.add_function_call_statement(
            f"set_{param_stem}",
            args=(f"{counter_variable_name}++",),
            deferred=True,
        )

    def _handle_set_oscillator_frequency_shf(self, sampled_event):
        iteration = sampled_event["iteration"]
        parameter_name = sampled_event["parameter_name"]
        counter_variable_name = string_sanitize(f"index_{parameter_name}")
        if iteration == 0:
            if counter_variable_name != f"index_{parameter_name}":
                _logger.warning(
                    "Parameter name '%s' has been sanitized in generated code.",
                    parameter_name,
                )
            self.declarations_generator.add_variable_declaration(
                counter_variable_name, 0
            )
            self.declarations_generator.add_function_call_statement(
                "configFreqSweep",
                (
                    0,
                    sampled_event["start_frequency"],
                    sampled_event["step_frequency"],
                ),
            )
            self.seqc_tracker.add_variable_assignment(counter_variable_name, 0)
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.add_function_call_statement(
            "setSweepStep",
            args=(0, f"{counter_variable_name}++"),
            deferred=True,
        )

    def handle_trigger_output(self, sampled_event):
        _logger.debug("  Processing trigger event %s", sampled_event)
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.add_function_call_statement(
            name="setTrigger", args=[sampled_event["state"]], deferred=True
        )

    def handle_loop_step_start(self, sampled_event):
        _logger.debug("  Processing LOOP_STEP_START EVENT %s", sampled_event)
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.append_loop_stack_generator()

    def handle_loop_step_end(self, sampled_event):
        _logger.debug("  Processing LOOP_STEP_END EVENT %s", sampled_event)
        self.seqc_tracker.add_required_playzeros(sampled_event)

    def handle_push_loop(self, sampled_event):
        _logger.debug(
            "  Processing PUSH_LOOP EVENT %s, top of stack is %s",
            sampled_event,
            self.seqc_tracker.current_loop_stack_generator(),
        )
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.append_loop_stack_generator(outer=True)

        if self.emit_timing_comments:
            self.seqc_tracker.add_comment(
                f"PUSH LOOP {sampled_event} current time = {self.seqc_tracker.current_time}"
            )

        self.loop_stack.append(sampled_event)

    def handle_iterate(self, sampled_event):
        if (
            self.seqc_tracker.current_loop_stack_generator().num_noncomment_statements()
            > 0
            or len(self.seqc_tracker.deferred_function_calls) > 0
        ):
            _logger.debug(
                "  Processing ITERATE EVENT %s, loop stack is %s",
                sampled_event,
                self.loop_stack,
            )
            if self.emit_timing_comments:
                self.seqc_tracker.add_comment(
                    f"ITERATE  {sampled_event}, current time = {self.seqc_tracker.current_time}"
                )
            self.seqc_tracker.add_required_playzeros(sampled_event)

            variable_name = string_sanitize(
                "repeat_count_" + str(sampled_event["loop_id"])
            )
            if variable_name not in self.declared_variables:
                self.declarations_generator.add_variable_declaration(variable_name)
                self.declared_variables.add(variable_name)

            loop_generator = SeqCGenerator()
            open_generators = self.seqc_tracker.pop_loop_stack_generators()
            _logger.debug(
                "  Popped %s, stack is now %s",
                open_generators,
                self.seqc_tracker.loop_stack_generators,
            )
            loop_body = compress_generators(
                open_generators, self.declarations_generator
            )
            loop_generator.add_countdown_loop(
                variable_name, sampled_event["num_repeats"], loop_body
            )
            if self.emit_timing_comments:
                loop_generator.add_comment(f"Loop for {sampled_event}")
            start_loop_event = self.loop_stack.pop()
            delta = sampled_event["start"] - start_loop_event["start"]
            self.seqc_tracker.current_time = (
                start_loop_event["start"] + sampled_event["num_repeats"] * delta
            )
            if self.emit_timing_comments:
                loop_generator.add_comment(
                    f"Delta: {delta} current time after loop: {self.seqc_tracker.current_time}, corresponding start event: {start_loop_event}"
                )
            self.seqc_tracker.append_loop_stack_generator(
                always=True, generator=loop_generator
            )
            self.seqc_tracker.append_loop_stack_generator(always=True)
        else:
            self.seqc_tracker.pop_loop_stack_generators()
            self.loop_stack.pop()

    def handle_match(self, sampled_event):
        self.match_parent_event = sampled_event

    def close_event_list(self):
        if self.match_parent_event is not None:
            handle = self.match_parent_event["handle"]
            sorted_ct_entries = sorted(self.match_command_table_entries.items())
            first = sorted_ct_entries[0][0]
            last = sorted_ct_entries[-1][0]
            if first != 0 or last - first + 1 != len(sorted_ct_entries):
                raise LabOneQException(
                    f"States missing in match statement with handle {handle}. First "
                    f"state: {first}, last state: {last}, number of states: "
                    f"{len(sorted_ct_entries)}, expected {last+1}, starting from 0."
                )

            # Check whether we already have the same states in the command table:
            if self.command_table_match_offset is not None:
                for idx, (signature, wave_index, _) in sorted_ct_entries:
                    current_ct_entry = self.command_table_tracker[
                        idx + self.command_table_match_offset
                    ]
                    current_wf_idx = current_ct_entry[1]["waveform"].get("index")
                    if current_ct_entry[0] != signature or wave_index != current_wf_idx:
                        raise LabOneQException(
                            "Multiple command table entry sets for feedback "
                            f"(handle {handle}), do you use the same pulses and states?"
                        )
            else:
                self.command_table_match_offset = len(self.command_table_tracker)
                for idx, (signature, wave_index, _) in sorted_ct_entries:
                    id2 = self.command_table_tracker.create_entry(signature, wave_index)
                    assert self.command_table_match_offset + idx == id2

            ev = self.match_parent_event
            if ev["local"]:
                if ev["start"] - self.seqc_tracker.current_time < 32:
                    _logger.warning(
                        f"Match section for handle {handle} too close to "
                        "previous play event, will be delayed by up to 32 samples."
                    )
                self.seqc_tracker.add_required_playzeros(
                    {"start": ev["start"] - 32, "signature": ev["signature"]}
                )
                # playZero anchor - make sure that the register readout of
                # executeTableEntry is happening at or after the requested time
                self.seqc_tracker.add_play_zero_statement(32)
                self.seqc_tracker.current_time += 32
                self.seqc_tracker.add_command_table_execution(
                    "QA_DATA_PROCESSED",
                    comment="Match handle " + handle,
                )
            else:
                assert ev["local"], "Global feedback not supported yet."
            self.seqc_tracker.add_timing_comment(ev["end"])
            self.seqc_tracker.clear_deferred_function_calls()
            self.seqc_tracker.current_time = self.match_parent_event["end"]
            self.match_parent_event = None

    def handle_sampled_event(self, sampled_event):
        signature = sampled_event["signature"]
        if signature == "playwave":
            if not self.handle_playwave(sampled_event):
                return
        elif signature == "acquire":
            self.handle_acquire(sampled_event)
        elif signature == "QA_EVENT":
            self.handle_qa_event(sampled_event)
        elif signature == "reset_precompensation_filters":
            self.handle_reset_precompensation_filters(sampled_event)
        elif signature in ("initial_reset_phase", "reset_phase"):
            self.handle_reset_phase(sampled_event)
        elif signature == "set_oscillator_frequency":
            self.handle_set_oscillator_frequency(sampled_event)
        elif signature == "trigger_output":
            self.handle_trigger_output(sampled_event)
        elif signature == "LOOP_STEP_START":
            self.handle_loop_step_start(sampled_event)
        elif signature == "LOOP_STEP_END":
            self.handle_loop_step_end(sampled_event)
        elif signature == "PUSH_LOOP":
            self.handle_push_loop(sampled_event)
        elif signature == "ITERATE":
            self.handle_iterate(sampled_event)
        elif signature == "match":
            self.handle_match(sampled_event)
        self.last_event = sampled_event

    def handle_sampled_event_list(self):
        for sampled_event in self.sampled_event_list:
            _logger.debug("  Processing event %s", sampled_event)
            self.handle_sampled_event(sampled_event)
        self.close_event_list()

    def handle_sampled_events(self, sampled_events):
        # Handle events grouped by start point
        for sampled_event_list in sampled_events.values():
            _logger.debug("EventListBeforeSort: %s", sampled_event_list)
            self.sampled_event_list = sort_events(sampled_event_list)
            _logger.debug("-Processing list:")
            for sampled_event_for_log in self.sampled_event_list:
                _logger.debug("       %s", sampled_event_for_log)
            _logger.debug("-End event list")
            self.handle_sampled_event_list()
        self.seqc_tracker.force_deferred_function_calls()

    def _add_feedback_connection(self, handle: str, acquire_signal: str):
        if handle is None:
            return
        try:
            fbc = self.feedback_connections[handle]
            if fbc.acquire is None:
                fbc.acquire = acquire_signal
            elif fbc.acquire != acquire_signal:
                raise LabOneQException(
                    f"Acquisition handle {handle} may not be "
                    f"reused with different signal {acquire_signal}"
                )
        except KeyError:
            self.feedback_connections[handle] = FeedbackConnection(acquire_signal)
