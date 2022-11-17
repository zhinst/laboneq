# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import List, Dict, TYPE_CHECKING

from laboneq.core.exceptions import LabOneQException
from laboneq.compiler.code_generator.compressor import compress_generators
from laboneq.compiler.code_generator.seq_c_generator import (
    SeqCGenerator,
    string_sanitize,
)
from laboneq.compiler.common.device_type import DeviceType

if TYPE_CHECKING:
    from laboneq.compiler.common.awg_info import AWGInfo
    from laboneq.compiler.code_generator.signatures import WaveformSignature
    from laboneq.compiler.code_generator.seqc_tracker import SeqCTracker
    from laboneq.compiler.code_generator.wave_index_tracker import WaveIndexTracker
    from laboneq.compiler.code_generator.command_table_tracker import (
        CommandTableTracker,
    )


_logger = logging.getLogger(__name__)


def sort_events(events):
    # For events that happen at the same sample, emit the play wave first, because it is asynchronous
    later = {
        "sequencer_start": -100,
        "initial_reset_phase": -4,
        "LOOP_STEP_START": -3,
        "PUSH_LOOP": -2,
        "reset_phase": -1,
        "acquire": 1,
        "ITERATE": 2,
    }
    sampled_event_list = sorted(
        events,
        key=lambda x: later[x["signature"]] if x["signature"] in later else 0,
    )

    return sampled_event_list


class SampledEventHandler:
    def __init__(
        self,
        seqc_tracker: SeqCTracker,
        command_table_tracker: CommandTableTracker,
        declarations_generator: SeqCGenerator,
        wave_indices: WaveIndexTracker,
        awg: AWGInfo,
        device_type: DeviceType,
        channels: List[int],
        sampled_signatures: Dict[str, Dict[WaveformSignature, Dict]],
        use_command_table: bool,
        emit_timing_comments: bool,
    ):
        self.sampled_event_list = None
        self.seqc_tracker = seqc_tracker
        self.command_table_tracker = command_table_tracker
        self.declarations_generator = declarations_generator
        self.declared_variables = set()
        self.loop_stack = []
        self.wave_indices = wave_indices
        self.awg = awg
        self.device_type = device_type
        self.channels = channels
        self.sampled_signatures = sampled_signatures
        self.use_command_table = use_command_table
        self.emit_timing_comments = emit_timing_comments
        self.last_playwave_event = None

    def handle_playwave(
        self,
        sampled_event,
    ):
        signature = sampled_event["playback_signature"]
        signal_id = sampled_event["signal_id"]

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
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.add_timing_comment(sampled_event["end"])
        generator = self.seqc_tracker.current_loop_stack_generator()

        play_wave_channel = None
        if len(self.channels) > 0:
            play_wave_channel = self.channels[0] % 2

        signal_type_for_wave_index = (
            self.awg.signal_type.value
            if self.device_type.supports_binary_waves
            else "csv"
            # Include CSV waves into the index to keep track of waves-AWG mapping
        )
        sig_string, _ = signature.waveform.signature_string()
        wave_index = self.wave_indices.lookup_index_by_wave_id(sig_string)
        if wave_index is None:
            wave_index = self.wave_indices.create_index_for_wave(
                sig_string, signal_type_for_wave_index
            )
            if wave_index is not None:
                generator.add_assign_wave_index_statement(
                    self.device_type,
                    self.awg.signal_type.value,
                    sig_string,
                    wave_index,
                    play_wave_channel,
                )
        if self.use_command_table:
            ct_index = self.command_table_tracker.lookup_index_by_signature(signature)
            if ct_index is None:
                ct_index = self.command_table_tracker.create_entry(
                    signature, wave_index
                )
            comment = sig_string
            if signature.hw_oscillator is not None:
                comment += f", osc={signature.hw_oscillator}"
            generator.add_command_table_execution(ct_index, comment)
        else:
            generator.add_play_wave_statement(
                self.device_type,
                self.awg.signal_type.value,
                sig_string,
                play_wave_channel,
            )
        self.seqc_tracker.clear_deferred_function_calls(sampled_event)

        self.seqc_tracker.current_time = sampled_event["end"]
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
            if self.last_playwave_event is not None:
                if (
                    self.last_playwave_event["signature"] == "acquire"
                    and self.last_playwave_event["start"] == start
                ):
                    skip = True
                    _logger.debug(
                        "Skipping acquire event %s because last event was also acquire: %s",
                        sampled_event,
                        self.last_playwave_event,
                    )
            if not skip:
                self.seqc_tracker.add_function_call_statement(
                    "startQA", args, deferred=False
                )

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
                    (
                        sig_string,
                        _,
                    ) = play_signature.signature_string()

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
            play_zero_after_qa = sampled_event["end"] - self.seqc_tracker.current_time
            self.seqc_tracker.add_timing_comment(sampled_event["end"])
            self.seqc_tracker.add_play_zero_statement(
                play_zero_after_qa, self.device_type
            )
        self.seqc_tracker.current_time = sampled_event["end"]

        self.seqc_tracker.add_function_call_statement("startQA", args)
        if "spectroscopy" in sampled_event["acquisition_type"]:
            self.seqc_tracker.add_function_call_statement("setTrigger", [0])

    def handle_reset_precompensation_filters(self, sampled_event):
        try:
            if sampled_event["signal_id"] in (s.id for s in self.awg.signals):
                _logger.debug(
                    "  Processing RESET PRECOMPENSATION FILTERS event %s",
                    sampled_event,
                )
                self.seqc_tracker.add_required_playzeros(sampled_event)
                self.seqc_tracker.add_function_call_statement(
                    name="setPrecompClear", args=[1]
                )
        except KeyError:
            pass

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
        assert not self.seqc_tracker.deferred_function_calls
        self.seqc_tracker.append_loop_stack_generator()

    def handle_push_loop(self, sampled_event):
        _logger.debug(
            "  Processing PUSH_LOOP EVENT %s, top of stack is %s",
            sampled_event,
            self.seqc_tracker.current_loop_stack_generator(),
        )
        self.seqc_tracker.add_required_playzeros(sampled_event)
        assert not self.seqc_tracker.deferred_function_calls
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
            assert not self.seqc_tracker.deferred_function_calls
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
        elif signature == "PUSH_LOOP":
            self.handle_push_loop(sampled_event)
        elif signature == "ITERATE":
            self.handle_iterate(sampled_event)
        self.last_playwave_event = sampled_event

    def handle_sampled_event_list(self):
        for sampled_event in self.sampled_event_list:
            _logger.debug("  Processing event %s", sampled_event)
            self.handle_sampled_event(sampled_event)

    def handle_sampled_events(self, sampled_events):
        for sampled_event_list in sampled_events.values():
            _logger.debug("EventListBeforeSort: %s", sampled_event_list)
            self.sampled_event_list = sort_events(sampled_event_list)
            _logger.debug("-Processing list:")
            for sampled_event_for_log in self.sampled_event_list:
                _logger.debug("       %s", sampled_event_for_log)
            _logger.debug("-End event list")
            self.handle_sampled_event_list()
