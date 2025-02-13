# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from laboneq.compiler.seqc.prng_tracker import PRNGTracker
from laboneq.compiler.seqc.seqc_generator import SeqCGenerator
from laboneq.compiler.seqc.awg_sampled_event import AWGEvent
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.seqc.output_mute import OutputMute


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
        automute_playzeros_min_duration: float,
        automute_playzeros: bool = False,
    ) -> None:
        self.deferred_function_calls = deferred_function_calls
        self.deferred_phase_changes = SeqCGenerator()
        self.loop_stack_generators: List[List[SeqCGenerator]] = [
            [init_generator, SeqCGenerator()]
        ]
        self.sampling_rate = sampling_rate
        self.delay = delay
        self.device_type = device_type
        self.emit_timing_comments = emit_timing_comments
        self.logger = logger
        self.current_time = 0
        self._prng_tracker: PRNGTracker | None = None
        self._automute: OutputMute | None = None
        self._active_trigger_outputs: int = 0
        if automute_playzeros:
            self._automute = OutputMute(
                device_type=self.device_type,
                generator=self,
                duration_min=automute_playzeros_min_duration,
            )

    @property
    def automute_playzeros(self) -> bool:
        if self._automute:
            return True
        return False

    def current_loop_stack_generator(self):
        return self.loop_stack_generators[-1][-1]

    def add_required_playzeros(self, sampled_event: AWGEvent) -> int:
        """If `current_time` precedes the scheduled start of the event, emit playZero to catch up.
        If muting is enabled, the emitted playZeros are muted if possible.

        Also clears deferred function calls within the context of the new playZero.

        Returns:
            Current time
        """
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
            if self._automute:
                if self._automute.can_mute(play_zero_samples):
                    self._automute.mute_samples(play_zero_samples)
                else:
                    self.add_timing_comment(self.current_time + play_zero_samples)
                    self.current_loop_stack_generator().add_play_zero_statement(
                        play_zero_samples,
                        self.device_type,
                        self.deferred_function_calls,
                    )
            else:
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

    def flush_deferred_phase_changes(self):
        """Phase changes (i.e. command table entries that take zero time) are emitted
        as late as possible, for example, we exchange it with a playZero that comes
        immediately after. This allows the sequencer to add more work to the wave player,
        and avoid gaps in the playback more effectively."""
        if self.has_deferred_phase_changes():
            self.current_loop_stack_generator().append_statements_from(
                self.deferred_phase_changes
            )
            self.deferred_phase_changes.clear()

    def has_deferred_phase_changes(self):
        return self.deferred_phase_changes.num_statements() > 0

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

    def add_play_zero_statement(self, num_samples: int, increment_counter=False):
        """Add playZero statement.

        Arguments:
            num_samples: Number of playZero samples.
            increment_counter: Increment current time counter.
                This is handy when using `add_required_playzeros()` within the same event.
        """
        if increment_counter:
            self.current_time += num_samples
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

    def add_phase_change(self, ct_index, comment=""):
        self.deferred_phase_changes.add_command_table_execution(
            ct_index=ct_index, comment=comment
        )

    def add_variable_assignment(self, variable_name, value):
        self.current_loop_stack_generator().add_variable_assignment(
            variable_name, value
        )

    def add_variable_increment(self, variable_name, value):
        self.current_loop_stack_generator().add_variable_increment(variable_name, value)

    # todo: remove `always` argument
    def append_loop_stack_generator(
        self, always=False, generator=None
    ) -> SeqCGenerator:
        if not generator:
            generator = SeqCGenerator()

        top_of_stack = self.loop_stack_generators[-1]
        if always or len(top_of_stack) == 0 or top_of_stack[-1].num_statements() > 0:
            top_of_stack.append(generator)
        return self.current_loop_stack_generator()

    def push_loop_stack_generator(self, generator=None):
        self.loop_stack_generators.append([])
        self.append_loop_stack_generator(generator)

    def pop_loop_stack_generators(self):
        top_of_stack = self.loop_stack_generators.pop()
        for i, gen in enumerate(top_of_stack):
            compressed = gen.compressed()
            top_of_stack[i] = compressed
        return top_of_stack

    def setup_prng(self, seed=None, prng_range=None, offset=None):
        """Insert a placeholder for setting up the PRNG

        In particular the offset into the command table can be efficiently encoded in
        range of the PRNG, which we do not know yet.

        This function returns a reference to a PRNGTracker through which the code
        generator can adjust the values and eventually commit them.
        Once committed, they values cannot be changed, and the PRNG has to be setup anew.
        """

        assert self._prng_tracker is None

        seqc_gen_prng = self.append_loop_stack_generator()
        self.append_loop_stack_generator()  # continuation

        self._prng_tracker = PRNGTracker(seqc_gen_prng)

        if seed is not None:
            self._prng_tracker.seed = seed
        if prng_range is not None:
            self._prng_tracker.range = prng_range
        if offset is not None:
            self._prng_tracker.offset = offset

    def drop_prng(self):
        assert self._prng_tracker is not None
        self._prng_tracker = None

    def add_prng_match_command_table_execution(self, offset: int):
        assert self.prng_tracker().is_committed()
        if offset != 0:
            index = f"prng_value + {offset}"
        else:
            index = "prng_value"
        self.add_command_table_execution(index)

    def sample_prng(self, declarations_generator: SeqCGenerator):
        variable_name = "prng_value"
        if not declarations_generator.is_variable_declared("prng_value"):
            declarations_generator.add_variable_declaration("prng_value")

        self.add_function_call_statement(
            name="getPRNGValue", args=[], deferred=False, assign_to=variable_name
        )

    def prng_tracker(self):
        return self._prng_tracker

    def add_set_trigger_statement(self, value: int, deferred=True):
        self.add_function_call_statement(
            name="setTrigger", args=[value], deferred=deferred
        )
        self._active_trigger_outputs = value

    def add_startqa_shfqa_statement(
        self,
        generator_mask: str,
        integrator_mask: str,
        monitor: int | bool | None = None,
        feedback_register: int | None = None,
        trigger: int | None = None,
    ):
        args: list[str] = [generator_mask, integrator_mask]
        if monitor is not None:
            args.append(str(monitor))
        if feedback_register is not None:
            assert monitor is not None
            args.append(str(feedback_register))
        if trigger is not None:
            assert monitor is not None
            assert feedback_register is not None
            args.append(str(trigger))
        else:
            trigger = 0
        self.add_function_call_statement("startQA", args, deferred=True)

        self._active_trigger_outputs = trigger

    def trigger_output_state(self):
        return self._active_trigger_outputs
