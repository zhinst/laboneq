# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from laboneq.compiler.code_generator.seq_c_generator import SeqCGenerator
from laboneq.compiler.common.awg_sampled_event import AWGEvent
from laboneq.compiler.common.device_type import DeviceType


class SeqCTracker:
    def __init__(
        self,
        init_generator: SeqCGenerator,
        deferred_function_calls: SeqCGenerator,
        sampling_rate: float,
        delay: float,
        device_type: DeviceType,
        emit_timing_comments: bool,
        logger,
    ) -> None:
        self.deferred_function_calls = deferred_function_calls
        self.loop_stack_generators: List[List[SeqCGenerator]] = [
            [init_generator, SeqCGenerator()]
        ]
        self.sampling_rate = sampling_rate
        self.delay = delay
        self.device_type = device_type
        self.emit_timing_comments = emit_timing_comments
        self.logger = logger
        self.current_time = 0

    def current_loop_stack_generator(self):
        return self.loop_stack_generators[-1][-1]

    def add_required_playzeros(self, sampled_event: AWGEvent):
        """If `current_time` precedes the scheduled start of the event, emit playZero to catch up.

        Also clears deferred function calls within the context of the new playZero."""
        start = sampled_event.start
        signature = sampled_event.type

        if start > self.current_time:
            play_zero_samples = start - self.current_time
            self.logger.debug(
                "  Emitting %d play zero samples before signature %s for event %s",
                play_zero_samples,
                signature,
                sampled_event,
            )

            self.add_timing_comment(self.current_time + play_zero_samples)
            self.current_loop_stack_generator().add_play_zero_statement(
                play_zero_samples, self.device_type, self.deferred_function_calls
            )
            self.current_time += play_zero_samples

        return self.current_time

    def flush_deferred_function_calls(self):
        """Emit the deferred function calls *now*."""
        if self.deferred_function_calls.num_statements() > 0:
            self.current_loop_stack_generator().append_statements_from(
                self.deferred_function_calls
            )
            self.deferred_function_calls.clear()

    def force_deferred_function_calls(self):
        """There may be deferred function calls issued *at the end of the sequence*.
        There will be no playWave or playZero that could flush them, so this function
        will force a flush.

        This function should not be needed anywhere but at the end of the sequence.
        (In the future, maybe at the end of a loop iteration?)
        """
        if self.deferred_function_calls.num_statements() > 0:
            # Flush any remaining deferred calls (ie those that should execute at the
            # end of the sequence)
            if self.device_type == DeviceType.SHFQA:
                # SHFQA does not support waitWave()
                self.add_play_zero_statement(32)
            else:
                self.add_function_call_statement("waitWave")
            self.flush_deferred_function_calls()

    def add_timing_comment(self, end_samples):
        if self.emit_timing_comments:
            start_time_ns = (
                round((self.current_time / self.sampling_rate - self.delay) * 1e10) / 10
            )
            end_time_ns = (
                round(((end_samples / self.sampling_rate) - self.delay) * 1e10) / 10
            )
            self.add_comment(
                f"{self.current_time} - {end_samples} , {start_time_ns} ns - {end_time_ns} ns "
            )

    def add_comment(self, comment):
        self.current_loop_stack_generator().add_comment(comment)

    def add_function_call_statement(
        self, name, args=None, assign_to=None, deferred=False
    ):
        if deferred:
            self.deferred_function_calls.add_function_call_statement(name, args)
        else:
            self.current_loop_stack_generator().add_function_call_statement(
                name, args, assign_to
            )

    def add_play_zero_statement(self, num_samples):
        self.current_loop_stack_generator().add_play_zero_statement(
            num_samples, self.device_type, self.deferred_function_calls
        )

    def add_play_hold_statement(self, num_samples):
        self.current_loop_stack_generator().add_play_hold_statement(
            num_samples, self.device_type, self.deferred_function_calls
        )

    def add_play_wave_statement(
        self, device_type: DeviceType, signal_type, wave_id, channel
    ):
        self.current_loop_stack_generator().add_play_wave_statement(
            device_type, signal_type, wave_id, channel
        )

    def add_command_table_execution(self, ct_index, latency=None, comment=""):
        assert latency is None or not isinstance(latency, int) or latency >= 31
        self.current_loop_stack_generator().add_command_table_execution(
            ct_index=ct_index, latency=latency, comment=comment
        )

    def add_variable_assignment(self, variable_name, value):
        self.current_loop_stack_generator().add_variable_assignment(
            variable_name, value
        )

    def add_variable_increment(self, variable_name, value):
        self.current_loop_stack_generator().add_variable_increment(variable_name, value)

    # todo: remove `always` argument
    def append_loop_stack_generator(self, always=False, generator=None):
        if not generator:
            generator = SeqCGenerator()

        top_of_stack = self.loop_stack_generators[-1]
        if always or len(top_of_stack) == 0 or top_of_stack[-1].num_statements() > 0:
            top_of_stack.append(generator)

    def push_loop_stack_generator(self, generator=None):
        self.loop_stack_generators.append([])
        self.append_loop_stack_generator(generator)

    def pop_loop_stack_generators(self):
        top_of_stack = self.loop_stack_generators.pop()
        for i, gen in enumerate(top_of_stack):
            compressed = gen.compressed()
            top_of_stack[i] = compressed
        return top_of_stack
