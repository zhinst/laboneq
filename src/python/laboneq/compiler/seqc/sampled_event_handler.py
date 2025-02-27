# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from functools import cmp_to_key
from typing import TYPE_CHECKING, Callable, List, Optional


from laboneq._utils import flatten
from laboneq.compiler.common.feedback_connection import FeedbackConnection
from laboneq.compiler.common.feedback_register_config import FeedbackRegisterConfig
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.feedback_router.feedback_router import (
    FeedbackRegisterLayout,
    GlobalFeedbackRegister,
    LocalFeedbackRegister,
)
from laboneq.compiler.seqc.seqc_generator import (
    SeqCGenerator,
    merge_generators,
)
from laboneq.compiler.seqc.shfppc_sweeper_config_tracker import (
    SHFPPCSweeperConfigTracker,
)
from laboneq.compiler.seqc.signatures import (
    PlaybackSignature,
    WaveformSignature,
)
from laboneq.compiler.seqc.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.common.compiler_settings import EXECUTETABLEENTRY_LATENCY
from laboneq.compiler.common.device_type import DeviceType
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType
from laboneq.core.utilities.string_sanitize import string_sanitize

if TYPE_CHECKING:
    from laboneq.compiler.seqc.command_table_tracker import (
        CommandTableTracker,
    )
    from laboneq.compiler.seqc.seqc_tracker import SeqCTracker
    from laboneq.compiler.seqc.wave_index_tracker import WaveIndexTracker
    from laboneq.compiler.common.awg_info import AWGInfo

_logger = logging.getLogger(__name__)


def sort_events(events: List[AWGEvent]) -> List[AWGEvent]:
    KEEP_ORDER = -1
    REVERSE_ORDER = 1  # noqa F841
    DONT_CARE = 0  # noqa F841

    def fixed_priorities(event1, event2) -> int | None:
        """Extensible list of special cases for event ordering."""
        if (
            event1.type == AWGEventType.ACQUIRE
            and event2.type == AWGEventType.TRIGGER_OUTPUT
        ):
            return KEEP_ORDER
        else:
            return None

    def cmp(event1, event2):
        if (v := fixed_priorities(event1, event2)) is not None:
            return v
        if (v := fixed_priorities(event2, event1)) is not None:
            return -v

        try:
            # Some events do not have a priority, notably PLAY_WAVE.
            # This is required, play must happen after acquire for instance, something
            # that is not captured by the event list.
            return event1.priority - event2.priority  # @IgnoreException
        except TypeError:
            # legacy ordering based on type only
            later = {
                AWGEventType.SEQUENCER_START: -100,
                AWGEventType.INITIAL_RESET_PHASE: -5,
                AWGEventType.LOOP_STEP_START: -4,
                AWGEventType.LOOP_STEP_END: -4,
                AWGEventType.INIT_AMPLITUDE_REGISTER: -3,
                AWGEventType.PUSH_LOOP: -2,
                AWGEventType.RESET_PHASE: -1,
                AWGEventType.RESET_PRECOMPENSATION_FILTERS: -1,
                AWGEventType.ACQUIRE: -1,
                AWGEventType.ITERATE: 2,
            }
            return later.get(event1.type, 0) - later.get(event2.type, 0)

    sampled_event_list = sorted(events, key=cmp_to_key(cmp))

    return sampled_event_list


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
        shfppc_sweeper_config_tracker: SHFPPCSweeperConfigTracker,
        function_defs_generator: SeqCGenerator,
        declarations_generator: SeqCGenerator,
        wave_indices: WaveIndexTracker,
        qa_signal_by_handle: dict[str, SignalObj],
        feedback_connections: dict[str, FeedbackConnection],
        feedback_register_layout: FeedbackRegisterLayout,
        feedback_register_config: FeedbackRegisterConfig,
        awg: AWGInfo,
        device_type: DeviceType,
        channels: List[int],
        use_command_table: bool,
        emit_timing_comments: bool,
        use_current_sequencer_step: bool,
    ):
        self.seqc_tracker = seqc_tracker
        self.command_table_tracker = command_table_tracker
        self.shfppc_sweeper_config_tracker = shfppc_sweeper_config_tracker
        self.function_defs_generator = function_defs_generator
        self.declarations_generator = declarations_generator
        self.wave_indices = wave_indices
        self.qa_signal_by_handle = qa_signal_by_handle
        self.feedback_connections = feedback_connections
        self.feedback_register_layout = feedback_register_layout
        self.feedback_register_config = feedback_register_config
        self.awg = awg
        self.device_type = device_type
        self.channels = channels
        self.use_command_table = use_command_table
        self.emit_timing_comments = emit_timing_comments
        self.sampled_event_list: List[AWGEvent] = None  # type: ignore
        self.loop_stack: List[AWGEvent] = []
        self.last_event: Optional[AWGEvent] = None
        self.match_parent_event: Optional[AWGEvent] = None

        # If true, this AWG sources feedback data from Zsync. If False, it sources data
        # from the local bus. None means neither source is used. Using both is illegal.
        self.use_zsync_feedback: bool | None = None

        self.match_command_table_entries: dict[
            int, tuple
        ] = {}  # For feedback or prng match
        self.match_seqc_generators: dict[int, SeqCGenerator] = {}  # user_register match
        self.current_sequencer_step = 0 if use_current_sequencer_step else None
        self.sequencer_step = 8  # todo(JL): Is this always the case, and how to get it?

    def _increment_sequencer_step(self):
        if self.current_sequencer_step is not None:
            assert self.seqc_tracker.current_time % self.sequencer_step == 0
            seq_step = self.seqc_tracker.current_time // self.sequencer_step
            if seq_step != self.current_sequencer_step:
                self.seqc_tracker.add_variable_increment(
                    "current_seq_step", seq_step - self.current_sequencer_step
                )
            self.current_sequencer_step = seq_step

    def handle_playwave(
        self,
        sampled_event: AWGEvent,
    ):
        signature: PlaybackSignature = sampled_event.params["playback_signature"]
        state = signature.state

        match_statement_active = self.match_parent_event is not None
        assert (state is not None) == match_statement_active
        handle = (
            self.match_parent_event.params["handle"]
            if self.match_parent_event is not None
            else None
        )

        if not self.use_command_table and state is not None and handle is not None:
            raise LabOneQException(
                f"Found match/case statement for handle {handle} on unsupported device."
            )
        if not signature.waveform:
            return False

        _logger.debug(
            "  Found matching signature %s for event %s",
            signature,
            sampled_event,
        )
        if state is None:
            # Playzeros were already added for match event
            self.seqc_tracker.add_required_playzeros(sampled_event)
            self.seqc_tracker.flush_deferred_phase_changes()
            self.seqc_tracker.add_timing_comment(sampled_event.end)

        play_wave_channel = None
        if len(self.channels) > 0:
            play_wave_channel = self.channels[0] % 2

        sig_string = signature.waveform.signature_string()
        wave_index = self.get_wave_index(signature, sig_string, play_wave_channel)
        if not match_statement_active:
            self.handle_regular_playwave(
                sampled_event, signature, sig_string, wave_index, play_wave_channel
            )
        else:
            if handle is not None:
                self.handle_playwave_on_feedback(sampled_event, signature, wave_index)
                return True
            user_register = self.match_parent_event.params.get("user_register")
            if user_register is not None:
                self.handle_playwave_on_user_register(
                    signature, sig_string, wave_index, play_wave_channel
                )
                return True
            assert self.match_parent_event.params.get("prng_sample") is not None
            self.handle_playwave_on_prng(sampled_event, signature, wave_index)
            return True

    def get_wave_index(self, signature, sig_string, play_wave_channel):
        signal_type_for_wave_index = (
            self.awg.signal_type.value
            if self.device_type.supports_binary_waves
            else "csv"
            # Include CSV waves into the index to keep track of waves-AWG mapping
        )
        if (
            not signature.waveform.samples
            and all(p.pulse is None for p in signature.waveform.pulses)
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
                    self.declarations_generator.add_assign_wave_index_statement(
                        self.device_type,
                        self.awg.signal_type.value,
                        sig_string,
                        wave_index,
                        play_wave_channel,
                    )

        return wave_index

    def precomp_reset_ct_index(self) -> int:
        length = 32
        sig_string = "precomp_reset"
        wave_index = self.wave_indices.lookup_index_by_wave_id(sig_string)

        signature = PlaybackSignature(
            waveform=WaveformSignature(
                length=length,
                pulses=None,
                samples=None,  # not relevant for command table entry creation
            ),
            clear_precompensation=True,
            hw_oscillator=None,
            pulse_parameters=(),
        )

        if wave_index is None:
            assert self.device_type.supports_binary_waves
            signal_type_for_wave_index = self.awg.signal_type.value

            self.declarations_generator.add_zero_wave_declaration(
                self.device_type, signal_type_for_wave_index, sig_string, length
            )
            wave_index = self.wave_indices.create_index_for_wave(
                sig_string, signal_type_for_wave_index
            )
            play_wave_channel = None
            if len(self.channels) > 0:
                play_wave_channel = self.channels[0] % 2
            self.declarations_generator.add_assign_wave_index_statement(
                self.device_type,
                signal_type_for_wave_index,
                sig_string,
                wave_index,
                play_wave_channel,
            )

            ct_index = self.command_table_tracker.create_entry(signature, wave_index)
        else:
            ct_index = self.command_table_tracker.lookup_index_by_signature(signature)

        return ct_index

    def handle_regular_playwave(
        self,
        sampled_event: AWGEvent,
        signature: PlaybackSignature,
        sig_string: str,
        wave_index: int | None,
        play_wave_channel: int | None,
    ):
        assert signature.waveform is not None
        if self.use_command_table:
            ct_index = self.command_table_tracker.lookup_index_by_signature(signature)
            if ct_index is None:
                ct_index = self.command_table_tracker.create_entry(
                    signature, wave_index
                )
            comment = self._make_command_table_comment(signature)
            self.seqc_tracker.add_command_table_execution(ct_index, comment=comment)
        else:
            self.seqc_tracker.add_play_wave_statement(
                self.device_type,
                self.awg.signal_type.value,
                sig_string,
                play_wave_channel,
            )
        self.seqc_tracker.flush_deferred_function_calls()
        self.seqc_tracker.current_time = sampled_event.end

    def handle_playwave_on_feedback(
        self,
        sampled_event: AWGEvent,
        signature: PlaybackSignature,
        wave_index: int | None,
    ):
        assert self.use_command_table
        assert self.match_parent_event is not None
        state = signature.state

        if state in self.match_command_table_entries:
            if self.match_command_table_entries[state] != (
                signature,
                wave_index,
                sampled_event.start - self.match_parent_event.start,
            ):
                raise LabOneQException(
                    f"Duplicate state {state} with different pulses for handle "
                    f"{self.match_parent_event.params['handle']} found."
                )
        else:
            self.match_command_table_entries[state] = (
                signature,
                wave_index,
                sampled_event.start - self.match_parent_event.start,
            )

    def handle_playwave_on_user_register(
        self,
        signature: PlaybackSignature,
        sig_string: str,
        wave_index: int | None,
        play_wave_channel: int | None,
    ):
        assert self.match_parent_event is not None
        user_register = self.match_parent_event.params["user_register"]
        state = signature.state
        assert state is not None
        assert user_register is not None
        branch_generator = self.match_seqc_generators.setdefault(state, SeqCGenerator())
        if self.use_command_table:
            ct_index = self.command_table_tracker.lookup_index_by_signature(signature)
            if ct_index is None:
                ct_index = self.command_table_tracker.create_entry(
                    signature, wave_index
                )
            comment = self._make_command_table_comment(signature)
            branch_generator.add_command_table_execution(ct_index, comment=comment)
        else:
            branch_generator.add_play_wave_statement(
                self.device_type,
                self.awg.signal_type.value,
                sig_string,
                play_wave_channel,
            )

    def handle_playwave_on_prng(
        self,
        sampled_event,
        signature: PlaybackSignature,
        wave_index: int | None,
    ):
        assert self.use_command_table
        assert self.match_parent_event is not None
        state = signature.state

        if state in self.match_command_table_entries:
            if self.match_command_table_entries[state] != (
                signature,
                wave_index,
                sampled_event.start - self.match_parent_event.start,
            ):
                raise LabOneQException(
                    f"Duplicate state {state} with different pulses for PRNG found in section "
                    f"{self.match_parent_event.params['section_name']}."
                )
        else:
            self.match_command_table_entries[state] = (
                signature,
                wave_index,
                sampled_event.start - self.match_parent_event.start,
            )

    @staticmethod
    def _make_command_table_comment(signature: PlaybackSignature) -> str:
        parts = []

        if signature.hw_oscillator is not None:
            parts.append(f"osc={signature.hw_oscillator}")

        if signature.set_phase is not None:
            parts.append(f"phase={signature.set_phase:.2g}")
        elif signature.increment_phase is not None:
            parts.append(f"phase+={signature.increment_phase:.2g}")

        if signature.amplitude_register is not None:
            ampl_register = f"amp_{signature.amplitude_register}"
        else:
            ampl_register = "amp"

        if signature.set_amplitude is not None:
            parts.append(f"{ampl_register}={signature.set_amplitude:.2g}")
        elif signature.increment_amplitude is not None:
            increment = signature.increment_amplitude
            if increment >= 0:
                parts.append(f"{ampl_register}+={increment:.2g}")
            else:
                parts.append(f"{ampl_register}-={-increment:.2g}")
        elif signature.amplitude_register is not None:
            parts.append(ampl_register)

        if signature.waveform is not None:
            parts.append(signature.waveform.signature_string())

        return "; ".join(parts)

    def handle_playhold(
        self,
        sampled_event: AWGEvent,
    ):
        assert self.seqc_tracker.current_time == sampled_event.start

        # There cannot be any zero-length phase increments between the head playWave
        # and the playHold.
        assert not self.seqc_tracker.has_deferred_phase_changes()

        self.seqc_tracker.add_play_hold_statement(
            sampled_event.end - sampled_event.start
        )
        self.seqc_tracker.current_time = sampled_event.end

    def handle_amplitude_register_init(self, sampled_event):
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.flush_deferred_phase_changes()
        assert self.use_command_table
        signature = sampled_event.params["playback_signature"]
        ct_index = self.command_table_tracker.lookup_index_by_signature(signature)
        if ct_index is None:
            ct_index = self.command_table_tracker.create_entry(signature, None)
        comment = self._make_command_table_comment(signature)
        self.seqc_tracker.add_command_table_execution(ct_index, comment=comment)

    def handle_acquire(self, sampled_event: AWGEvent):
        _logger.debug("  Processing ACQUIRE EVENT %s", sampled_event)

        args = [
            "QA_INT_ALL",
            "1" if "RAW" in sampled_event.params["acquisition_type"] else "0",
        ]
        start = sampled_event.start

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
                    self.last_event.type == AWGEventType.ACQUIRE
                    and self.last_event.start == start
                ):
                    skip = True
                    _logger.debug(
                        "Skipping acquire event %s because last event was also acquire: %s",
                        sampled_event,
                        self.last_event,
                    )
            if not skip:
                self.seqc_tracker.add_function_call_statement(
                    "startQA", args, deferred=True
                )

    def handle_qa_event(self, sampled_event: AWGEvent):
        generator_channels = set()
        play_event: AWGEvent
        for play_event in sampled_event.params["play_events"]:
            signature: PlaybackSignature = play_event.params["playback_signature"]
            play_signature = signature.waveform
            signal_id = play_event.params.get("signal_id")
            if signal_id is not None and play_signature:
                current_signal_obj = next(
                    signal_obj
                    for signal_obj in self.awg.signals
                    if signal_obj.id == signal_id
                )
                generator_channels.update(current_signal_obj.channels)
                self.wave_indices.add_numbered_wave(
                    play_signature.signature_string(),
                    "complex",
                    current_signal_obj.channels[0],
                )

        integration_channels = list(
            flatten(
                event.params["channels"]
                for event in sampled_event.params["acquire_events"]
            )
        )

        if len(integration_channels) > 0:
            integrator_mask: str = "|".join(
                map(lambda x: "QA_INT_" + str(x), integration_channels)
            )
        else:
            integrator_mask = "QA_INT_NONE"

        if len(generator_channels) > 0:
            generator_mask: str = "|".join(
                map(lambda x: "QA_GEN_" + str(x), generator_channels)
            )
        else:
            generator_mask = "QA_GEN_NONE"

        is_spectroscopy = bool(
            set(sampled_event.params["acquisition_type"]).intersection(
                [
                    AcquisitionType.SPECTROSCOPY_IQ.value,
                    AcquisitionType.SPECTROSCOPY.value,
                    AcquisitionType.SPECTROSCOPY_PSD.value,
                ]
            )
        )
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.flush_deferred_phase_changes()
        if sampled_event.end > self.seqc_tracker.current_time:
            self.seqc_tracker.add_timing_comment(sampled_event.end)

        if is_spectroscopy:
            generator_mask = "0"

            # In spectroscopy mode, there are no distinct integrators that can be,
            # triggered, so the mask is ignored. By setting it to a non-null value
            # however, we ensure that the timestamp of the acquisition is correctly
            # latched.
            integrator_mask = "1"

            self.seqc_tracker.add_startqa_shfqa_statement(
                generator_mask,
                integrator_mask,
                monitor=0,
                feedback_register=0,
                trigger=0b10 | self.seqc_tracker.trigger_output_state(),
            )
        else:
            monitor = 1 if "RAW" in sampled_event.params["acquisition_type"] else 0
            feedback_register = sampled_event.params["feedback_register"] or 0
            self.seqc_tracker.add_startqa_shfqa_statement(
                generator_mask,
                integrator_mask,
                monitor=monitor,
                feedback_register=feedback_register,
                trigger=self.seqc_tracker.trigger_output_state(),
            )
        if is_spectroscopy:
            self.seqc_tracker.add_set_trigger_statement(
                value=self.seqc_tracker.trigger_output_state() & 0b1
            )
        if self.seqc_tracker.automute_playzeros:
            # playZero for the waveform, which must not be muted.
            self.seqc_tracker.add_play_zero_statement(
                sampled_event.end - sampled_event.start, increment_counter=True
            )
            self.seqc_tracker.flush_deferred_function_calls()

    def handle_reset_precompensation_filters(self, sampled_event: AWGEvent):
        if sampled_event.params["signal_id"] not in (s.id for s in self.awg.signals):
            return

        self.seqc_tracker.add_required_playzeros(sampled_event)

        if self.use_command_table:
            ct_index = self.precomp_reset_ct_index()
            self.seqc_tracker.add_command_table_execution(
                ct_index, comment="precomp_reset"
            )
            self.seqc_tracker.flush_deferred_function_calls()
            assert not self.seqc_tracker.has_deferred_phase_changes()
            self.seqc_tracker.current_time = sampled_event.end
            return

        _logger.debug(
            "  Processing RESET PRECOMPENSATION FILTERS event %s",
            sampled_event,
        )
        self.seqc_tracker.add_function_call_statement(
            name="setPrecompClear", args=[1], deferred=False
        )
        self.seqc_tracker.add_function_call_statement(
            name="setPrecompClear", args=[0], deferred=True
        )

    def handle_reset_phase(self, sampled_event: AWGEvent):
        if not self.awg.device_type.supports_reset_osc_phase:
            return

        # If multiple phase reset events are scheduled at the same time,
        # only process the *last* one. This way, RESET_PHASE takes
        # precedence over INITIAL_RESET_PHASE.
        last_reset = [
            event
            for event in self.sampled_event_list
            if event.type
            in (AWGEventType.RESET_PHASE, AWGEventType.INITIAL_RESET_PHASE)
        ][-1]
        if last_reset is not sampled_event:
            return

        if self.use_command_table:
            # `resetOscPhase()` resets the DDS phase, but it does not clear the phase
            # offset from the command table. We do that via a zero-length CT entry.
            signature = PlaybackSignature(
                waveform=None,
                hw_oscillator=None,
                pulse_parameters=(),
                set_phase=0.0,
            )

            ct_index = self.command_table_tracker.get_or_create_entry(
                signature, wave_index=None
            )
        else:
            ct_index = None

        _logger.debug("  Processing RESET PHASE event %s", sampled_event)
        start = sampled_event.start
        if sampled_event.type == AWGEventType.INITIAL_RESET_PHASE:
            if start > self.seqc_tracker.current_time:
                self.seqc_tracker.add_required_playzeros(sampled_event)
            # Hack: we do not defer, and emit as early as possible.
            # This way it is hidden in the lead time.
            self.seqc_tracker.add_function_call_statement("resetOscPhase")
            if ct_index is not None:
                assert not self.seqc_tracker.has_deferred_phase_changes()
                self.seqc_tracker.add_command_table_execution(ct_index)
        elif sampled_event.type == AWGEventType.RESET_PHASE:
            self.seqc_tracker.add_required_playzeros(sampled_event)
            self.seqc_tracker.add_function_call_statement(
                "resetOscPhase", deferred=True
            )
            if ct_index is not None:
                assert not self.seqc_tracker.has_deferred_phase_changes()
                self.seqc_tracker.add_command_table_execution(ct_index)

    def handle_set_oscillator_frequency(self, sampled_event: AWGEvent):
        if self.device_type == DeviceType.HDAWG:
            self._handle_set_oscillator_frequency_hdawg(sampled_event)
        else:
            self._handle_set_oscillator_frequency_shf(sampled_event)

    def _handle_set_oscillator_frequency_hdawg(self, sampled_event: AWGEvent):
        iteration = sampled_event.params["iteration"]
        parameter_name = sampled_event.params["parameter_name"]
        param_stem = string_sanitize(parameter_name)
        counter_variable_name = string_sanitize(f"index_{parameter_name}")
        if iteration == 0:
            iterations = sampled_event.params["iterations"]
            if iterations > 512:
                raise LabOneQException(
                    "HDAWG can only handle RT frequency sweeps up to 512 steps."
                )
            steps = "\n  ".join(
                generate_if_else_tree(
                    iterations=iterations,
                    variable=f"arg_{param_stem}",
                    step_factory=lambda i: f"setDouble(osc_node_{param_stem}, {sampled_event.params['start_frequency'] + sampled_event.params['step_frequency'] * i});",
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

    def _handle_set_oscillator_frequency_shf(self, sampled_event: AWGEvent):
        iteration = sampled_event.params["iteration"]
        parameter_name = sampled_event.params["parameter_name"]
        counter_variable_name = string_sanitize(f"index_{parameter_name}")
        osc_id_symbol = string_sanitize(sampled_event.params["oscillator_id"])

        if not self.declarations_generator.is_variable_declared(counter_variable_name):
            self.declarations_generator.add_variable_declaration(
                counter_variable_name, 0
            )
            self.declarations_generator.add_constant_definition(
                osc_id_symbol,
                0,
                "preliminary! will be updated by controller",
            )
            self.declarations_generator.add_function_call_statement(
                "configFreqSweep",
                (
                    osc_id_symbol,
                    sampled_event.params["start_frequency"],
                    sampled_event.params["step_frequency"],
                ),
            )

        if counter_variable_name != f"index_{parameter_name}":
            _logger.warning(
                "Parameter name '%s' has been sanitized in generated code.",
                parameter_name,
            )

        if iteration == 0:
            self.seqc_tracker.add_variable_assignment(counter_variable_name, 0)
        self.seqc_tracker.add_required_playzeros(sampled_event)

        self.seqc_tracker.add_function_call_statement(
            "setSweepStep",
            args=(osc_id_symbol, f"{counter_variable_name}++"),
            deferred=True,
        )

    def handle_trigger_output(self, sampled_event: AWGEvent):
        _logger.debug("  Processing trigger event %s", sampled_event)
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.add_set_trigger_statement(value=sampled_event.params["state"])

    def handle_loop_step_start(self, sampled_event: AWGEvent):
        _logger.debug("  Processing LOOP_STEP_START EVENT %s", sampled_event)
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.append_loop_stack_generator()

    def handle_loop_step_end(self, sampled_event: AWGEvent):
        _logger.debug("  Processing LOOP_STEP_END EVENT %s", sampled_event)
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self._increment_sequencer_step()

    def handle_push_loop(self, sampled_event: AWGEvent):
        _logger.debug(
            "  Processing PUSH_LOOP EVENT %s, top of stack is %s",
            sampled_event,
            self.seqc_tracker.current_loop_stack_generator(),
        )
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.flush_deferred_phase_changes()
        if self.current_sequencer_step is not None:
            assert self.seqc_tracker.current_time % self.sequencer_step == 0
            self.current_sequencer_step = (
                self.seqc_tracker.current_time // self.sequencer_step
            )
            self.seqc_tracker.add_variable_assignment(
                "current_seq_step", self.current_sequencer_step
            )
        self.seqc_tracker.push_loop_stack_generator()

        if self.emit_timing_comments:
            self.seqc_tracker.add_comment(
                f"PUSH LOOP {sampled_event.params} current time = {self.seqc_tracker.current_time}"
            )

        self.loop_stack.append(sampled_event)
        self.shfppc_sweeper_config_tracker.enter_loop(
            sampled_event.params["num_repeats"]
        )

    def handle_iterate(self, sampled_event: AWGEvent):
        if (
            any(
                cg.num_noncomment_statements() > 0
                for cg in self.seqc_tracker.loop_stack_generators[-1]
            )
            or self.seqc_tracker.deferred_function_calls.num_statements() > 0
            or self.seqc_tracker.deferred_phase_changes.num_statements() > 0
        ):
            _logger.debug(
                "  Processing ITERATE EVENT %s, loop stack is %s",
                sampled_event,
                self.loop_stack,
            )
            if self.emit_timing_comments:
                self.seqc_tracker.add_comment(
                    f"ITERATE  {sampled_event.params}, current time = {self.seqc_tracker.current_time}"
                )
            self.seqc_tracker.add_required_playzeros(sampled_event)
            self.seqc_tracker.flush_deferred_phase_changes()
            self._increment_sequencer_step()

            loop_generator = SeqCGenerator()
            open_generators = self.seqc_tracker.pop_loop_stack_generators()
            _logger.debug(
                "  Popped %s, stack is now %s",
                open_generators,
                self.seqc_tracker.loop_stack_generators,
            )
            loop_body = merge_generators(open_generators)
            loop_generator.add_repeat(sampled_event.params["num_repeats"], loop_body)
            if self.emit_timing_comments:
                loop_generator.add_comment(f"Loop for {sampled_event.params}")
            start_loop_event = self.loop_stack.pop()
            delta = sampled_event.start - start_loop_event.start
            self.seqc_tracker.current_time = (
                start_loop_event.start + sampled_event.params["num_repeats"] * delta
            )
            if self.emit_timing_comments:
                loop_generator.add_comment(
                    f"Delta: {delta} current time after loop: {self.seqc_tracker.current_time}, corresponding start event: {start_loop_event.params}"
                )
            self.seqc_tracker.append_loop_stack_generator(
                always=True, generator=loop_generator
            )
            self.seqc_tracker.append_loop_stack_generator(always=True)
        else:
            self.seqc_tracker.pop_loop_stack_generators()
            self.loop_stack.pop()

        self.shfppc_sweeper_config_tracker.exit_loop()

    def handle_match(self, sampled_event: AWGEvent):
        if (this_sample := sampled_event.params.get("prng_sample")) is not None:
            other_sample = self.seqc_tracker.prng_tracker().active_sample
            section = sampled_event.params["section_name"]
            if this_sample != other_sample:
                raise LabOneQException(
                    f"In section '{section}: cannot match PRNG sample '{this_sample}' here. The only available PRNG sample is '{other_sample}'."
                )

        if self.match_parent_event is not None:
            mpe_par = self.match_parent_event.params
            se_par = sampled_event.params
            raise LabOneQException(
                f"Simultaneous match events on the same physical AWG are not supported. "
                "Affected handles/user registers: '"
                f"{mpe_par['handle'] or mpe_par['user_register']}' and '"
                f"{se_par['handle'] or se_par['user_register']}'"
            )
        self.match_parent_event = sampled_event
        self.match_seqc_generators = {}

    def handle_change_oscillator_phase(self, sampled_event: AWGEvent):
        signature = PlaybackSignature(
            waveform=None,
            hw_oscillator=sampled_event.params["oscillator"],
            pulse_parameters=(),
            increment_phase=sampled_event.params["phase"],
            increment_phase_params=(sampled_event.params["parameter"],),
        )

        # The `phase_resolution_range` is irrelevant here; for the CT phase a fixed
        # precision is used.
        signature.quantize_phase(0)

        ct_index = self.command_table_tracker.get_or_create_entry(signature, None)
        self.seqc_tracker.add_phase_change(
            ct_index, comment=self._make_command_table_comment(signature)
        )

    def handle_setup_prng(self, sampled_event: AWGEvent):
        if not self.device_type.has_prng:
            return

        seed = sampled_event.params["seed"]
        prng_range = sampled_event.params["range"]
        section = sampled_event.params["section"]

        assert seed is not None and prng_range is not None

        if self.seqc_tracker.prng_tracker() is not None:
            raise LabOneQException(
                f"In section '{section}': Cannot seed PRNG, it is already allocated in this context"
            )

        self.seqc_tracker.setup_prng(seed=seed, prng_range=prng_range)

    def handle_drop_prng_setup(self, _sampled_event: AWGEvent):
        if not self.device_type.has_prng:
            return
        self.seqc_tracker.drop_prng()

    def handle_sample_prng(self, sampled_event: AWGEvent):
        if not self.device_type.has_prng:
            return

        prng_tracker = self.seqc_tracker.prng_tracker()
        if prng_tracker.active_sample is not None:
            section = sampled_event.params["section_name"]
            this_sample = sampled_event.params["sample_name"]
            other_sample = prng_tracker.active_sample
            raise LabOneQException(
                f"In section '{section}': Can't draw sample '{this_sample}' from PRNG,"
                f" when other sample '{other_sample}' is still required at the same time"
            )
        prng_tracker.active_sample = sampled_event.params["sample_name"]

        self.seqc_tracker.sample_prng(self.declarations_generator)

    def handle_drop_prng_sample(self, sampled_event: AWGEvent):
        if not self.device_type.has_prng:
            return
        prng_tracker = self.seqc_tracker.prng_tracker()
        assert prng_tracker.active_sample == sampled_event.params["sample_name"]
        prng_tracker.drop_sample()
        self.match_command_table_entries.clear()

    def handle_ppc_step_start(self, sampled_event: AWGEvent):
        assert sampled_event.type == AWGEventType.PPC_SWEEP_STEP_START
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.add_function_call_statement("setTrigger", [1], deferred=True)
        self.shfppc_sweeper_config_tracker.add_step(**sampled_event.params)

    def handle_ppc_step_end(self, sampled_event: AWGEvent):
        assert sampled_event.type == AWGEventType.PPC_SWEEP_STEP_END
        self.seqc_tracker.add_required_playzeros(sampled_event)
        self.seqc_tracker.add_function_call_statement("setTrigger", [0], deferred=True)

    def _register_bitshift(
        self,
        register: GlobalFeedbackRegister | LocalFeedbackRegister,
        qa_signal: str,
        force_local_alignment: bool = False,
    ):
        """Calculate offset and mask into register for given qa_signal"""
        register_bitshift = 0  # offset into the register
        for width, signal in self.feedback_register_layout[register]:
            if signal == qa_signal:
                break
            else:
                register_bitshift += width if not force_local_alignment else 2
        else:
            raise AssertionError(f"Signal {qa_signal} not found in register {register}")
        mask = (1 << width) - 1
        return register_bitshift, width, mask

    def add_feedback_config(self, handle: str, local: bool):
        qa_signal = self.qa_signal_by_handle[handle]

        self.feedback_connections.setdefault(
            handle, FeedbackConnection(tx=qa_signal.awg.key)
        ).rx.add(self.awg.key)

        if local:
            register = LocalFeedbackRegister(qa_signal.awg.device_id)
            codeword_bitshift, width, mask = self._register_bitshift(
                register, qa_signal=qa_signal.id, force_local_alignment=True
            )
            index_select = None
        else:
            register = GlobalFeedbackRegister(qa_signal.awg.key)
            qa_register_bitshift, width, mask = self._register_bitshift(
                register, qa_signal=qa_signal.id, force_local_alignment=False
            )

            # feedback through PQSC: assign index based on AWG number
            index_select = qa_register_bitshift // 2
            codeword_bitshift = 2 * self.awg.awg_id + qa_register_bitshift % 2

        if local:
            self.feedback_register_config.source_feedback_register = "local"

        if not local:
            path = "ZSYNC_DATA_PROCESSED_A"
        else:
            path = "QA_DATA_PROCESSED"
        self.declarations_generator.add_function_call_statement(
            "configureFeedbackProcessing",
            args=[
                path,
                codeword_bitshift,
                width,  # todo: According to docs, should be decremented
                self.feedback_register_config.command_table_offset,
            ],
        )

        self.feedback_register_config.codeword_bitshift = codeword_bitshift
        self.feedback_register_config.codeword_bitmask = mask
        self.feedback_register_config.register_index_select = index_select

    def close_event_list(self):
        if self.match_parent_event is not None:
            params = self.match_parent_event.params
            if params["handle"] is not None:
                self.close_event_list_for_handle()
            elif params["user_register"] is not None:
                self.close_event_list_for_user_register()
            elif params["prng_sample"]:
                self.close_event_list_for_prng_match()

    def close_event_list_for_handle(self):
        assert self.match_parent_event is not None
        handle = self.match_parent_event.params["handle"]
        sorted_ct_entries = sorted(self.match_command_table_entries.items())
        first = sorted_ct_entries[0][0]
        last = sorted_ct_entries[-1][0]
        if first != 0 or last - first + 1 != len(sorted_ct_entries):
            raise LabOneQException(
                f"States missing in match statement with handle {handle}. First "
                f"state: {first}, last state: {last}, number of states: "
                f"{len(sorted_ct_entries)}, expected {last + 1}, starting from 0."
            )

        # Check whether we already have the same states in the command table:
        if self.feedback_register_config.command_table_offset is not None:
            for idx, (signature, wave_index, _) in sorted_ct_entries:
                current_ct_entry = self.command_table_tracker[
                    idx + self.feedback_register_config.command_table_offset
                ]
                assert current_ct_entry is not None
                current_wf_idx = current_ct_entry[1]["waveform"].get("index")
                if current_ct_entry[0] != signature or wave_index != current_wf_idx:
                    raise LabOneQException(
                        "Multiple command table entry sets for feedback "
                        f"(handle {handle}), do you use the same pulses and states?"
                    )
        else:
            local = self.match_parent_event.params["local"]
            self.feedback_register_config.command_table_offset = len(
                self.command_table_tracker
            )
            self.add_feedback_config(handle, local)
            # Allocate command table entries
            for idx, (signature, wave_index, _) in sorted_ct_entries:
                id2 = self.command_table_tracker.create_entry(signature, wave_index)
                assert self.feedback_register_config.command_table_offset + idx == id2

        ev = self.match_parent_event
        start = ev.start
        assert start >= self.seqc_tracker.current_time
        assert start % self.sequencer_step == 0
        self.seqc_tracker.add_required_playzeros(ev)
        self.seqc_tracker.flush_deferred_phase_changes()
        # Subtract the 3 cycles that we added (see match_schedule.py for details)
        assert self.current_sequencer_step is not None
        latency = (
            start // self.sequencer_step
            - self.current_sequencer_step
            - EXECUTETABLEENTRY_LATENCY
        )

        ete_global_feedback_source = "ZSYNC_DATA_PROCESSED_A"
        self.seqc_tracker.add_command_table_execution(
            "QA_DATA_PROCESSED" if ev.params["local"] else ete_global_feedback_source,
            latency="current_seq_step "
            + (f"+ {latency}" if latency >= 0 else f"- {-latency}"),
            comment="Match handle " + handle,
        )
        use_zsync: bool = not ev.params["local"]
        if self.use_zsync_feedback is not None and self.use_zsync_feedback != use_zsync:
            raise LabOneQException(
                "Mixed feedback paths (global and local) are illegal"
            )
        self.use_zsync_feedback = use_zsync
        self.seqc_tracker.add_timing_comment(ev.end)
        self.seqc_tracker.flush_deferred_function_calls()
        self.seqc_tracker.current_time = self.match_parent_event.end
        self.match_parent_event = None

    def close_event_list_for_user_register(self):
        match_event = self.match_parent_event
        assert match_event is not None
        user_register = match_event.params["user_register"]
        if not 0 <= user_register <= 15:
            raise LabOneQException(
                f"Invalid user register {user_register} in match statement. User registers must be between 0 and 15."
            )
        var_name = f"_match_user_register_{user_register}"
        try:
            self.declarations_generator.add_variable_declaration(
                var_name, f"getUserReg({user_register})"
            )
        except LabOneQException:
            pass  # Already declared, this is fine
        self.seqc_tracker.add_required_playzeros(match_event)
        self.seqc_tracker.flush_deferred_phase_changes()
        if_generator = SeqCGenerator()
        conditions_bodies: list[tuple[str | None, SeqCGenerator]] = [
            (f"{var_name} == {state}", gen.compressed())
            for state, gen in self.match_seqc_generators.items()
            if gen.num_noncomment_statements() > 0
        ]
        # If there is no match, we just play zeros to keep the timing correct
        play_zero_body = SeqCGenerator()
        play_zero_body.add_play_zero_statement(
            match_event.end - self.seqc_tracker.current_time,
            self.device_type,
        )
        conditions_bodies.append((None, play_zero_body.compressed()))
        if_generator.add_if(*zip(*conditions_bodies))  # type: ignore
        self.seqc_tracker.append_loop_stack_generator(
            always=True, generator=if_generator
        )
        self.seqc_tracker.append_loop_stack_generator(always=True)
        self.seqc_tracker.add_timing_comment(match_event.end)
        self.seqc_tracker.flush_deferred_function_calls()
        self.seqc_tracker.current_time = match_event.end
        self.match_parent_event = None

    def close_event_list_for_prng_match(self):
        assert self.match_parent_event is not None
        section = self.match_parent_event.params["section_name"]
        sorted_ct_entries = sorted(self.match_command_table_entries.items())
        first = sorted_ct_entries[0][0]
        last = sorted_ct_entries[-1][0]
        if first != 0 or last - first + 1 != len(sorted_ct_entries):
            raise LabOneQException(
                f"States missing in match statement (section {section}). First "
                f"state: {first}, last state: {last}, number of states: "
                f"{len(sorted_ct_entries)}, expected {last + 1}, starting from 0."
            )

        command_table_match_offset = len(self.command_table_tracker)
        # Allocate command table entries
        for idx, (signature, wave_index, _) in sorted_ct_entries:
            id2 = self.command_table_tracker.create_entry(
                signature, wave_index, ignore_already_in_table=True
            )
            assert command_table_match_offset + idx == id2

        prng_tracker = self.seqc_tracker.prng_tracker()
        if not prng_tracker.is_committed():
            # use the PRNG output range to do the offset for us
            prng_tracker.offset = command_table_match_offset
            prng_tracker.commit()
        command_table_match_offset -= prng_tracker.offset

        ev = self.match_parent_event
        start = ev.start
        assert start >= self.seqc_tracker.current_time
        self.seqc_tracker.add_required_playzeros(ev)
        self.seqc_tracker.flush_deferred_phase_changes()
        self.seqc_tracker.add_prng_match_command_table_execution(
            command_table_match_offset
        )
        self.seqc_tracker.add_timing_comment(ev.end)
        self.seqc_tracker.flush_deferred_function_calls()
        self.seqc_tracker.current_time = self.match_parent_event.end

        self.match_parent_event = None

    def handle_sampled_event(self, sampled_event: AWGEvent):
        signature = sampled_event.type
        if signature == AWGEventType.PLAY_WAVE:
            if not self.handle_playwave(sampled_event):
                return
        if signature == AWGEventType.PLAY_HOLD:
            self.handle_playhold(sampled_event)
        elif signature == AWGEventType.INIT_AMPLITUDE_REGISTER:
            self.handle_amplitude_register_init(sampled_event)
        elif signature == AWGEventType.ACQUIRE:
            self.handle_acquire(sampled_event)
        elif signature == AWGEventType.QA_EVENT:
            self.handle_qa_event(sampled_event)
        elif signature == AWGEventType.RESET_PRECOMPENSATION_FILTERS:
            self.handle_reset_precompensation_filters(sampled_event)
        elif signature in (
            AWGEventType.INITIAL_RESET_PHASE,
            AWGEventType.RESET_PHASE,
        ):
            self.handle_reset_phase(sampled_event)
        elif signature == AWGEventType.SET_OSCILLATOR_FREQUENCY:
            self.handle_set_oscillator_frequency(sampled_event)
        elif signature == AWGEventType.TRIGGER_OUTPUT:
            self.handle_trigger_output(sampled_event)
        elif signature == AWGEventType.LOOP_STEP_START:
            self.handle_loop_step_start(sampled_event)
        elif signature == AWGEventType.LOOP_STEP_END:
            self.handle_loop_step_end(sampled_event)
        elif signature == AWGEventType.PUSH_LOOP:
            self.handle_push_loop(sampled_event)
        elif signature == AWGEventType.ITERATE:
            self.handle_iterate(sampled_event)
        elif signature == AWGEventType.MATCH:
            self.handle_match(sampled_event)
        elif signature == AWGEventType.CHANGE_OSCILLATOR_PHASE:
            self.handle_change_oscillator_phase(sampled_event)
        elif signature == AWGEventType.SETUP_PRNG:
            self.handle_setup_prng(sampled_event)
        elif signature == AWGEventType.DROP_PRNG_SETUP:
            self.handle_drop_prng_setup(sampled_event)
        elif signature == AWGEventType.PRNG_SAMPLE:
            self.handle_sample_prng(sampled_event)
        elif signature == AWGEventType.DROP_PRNG_SAMPLE:
            self.handle_drop_prng_sample(sampled_event)
        elif signature == AWGEventType.PPC_SWEEP_STEP_START:
            self.handle_ppc_step_start(sampled_event)
        elif signature == AWGEventType.PPC_SWEEP_STEP_END:
            self.handle_ppc_step_end(sampled_event)
        self.last_event = sampled_event

    def handle_sampled_event_list(self):
        for sampled_event in self.sampled_event_list:
            _logger.debug("  Processing event %s", sampled_event)
            self.handle_sampled_event(sampled_event)
        self.close_event_list()

    def handle_sampled_events(self, sampled_events: AWGSampledEventSequence):
        # Handle events grouped by start point
        for sampled_event_list in sampled_events.sequence.values():
            _logger.debug("EventListBeforeSort: %s", sampled_event_list)
            self.sampled_event_list = sort_events(sampled_event_list)
            _logger.debug("-Processing list:")
            for sampled_event_for_log in self.sampled_event_list:
                _logger.debug("       %s", sampled_event_for_log)
            _logger.debug("-End event list")
            self.handle_sampled_event_list()
        self.seqc_tracker.force_deferred_function_calls()
