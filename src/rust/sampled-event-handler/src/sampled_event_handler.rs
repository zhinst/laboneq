// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::awg_events::{
    AwgEvent, ChangeHwOscPhase, EventType, Iterate, MatchEvent, PlayWaveEvent, PrngSetup, PushLoop,
    QaEvent, TriggerOutput,
};
use crate::command_table_tracker::CommandTableTracker;
use crate::feedback_register_config::FeedbackRegisterConfig;
use crate::feedback_register_layout::{FeedbackRegister, FeedbackRegisterLayout};
use crate::shfppc_sweeper_config_tracker::SHFPPCSweeperConfigTracker;
use crate::{AwgEventList, FeedbackRegisterIndex, Result, Samples, SeqcResults};
use codegenerator::WaveDeclaration;
use codegenerator::device_traits::{self};
use codegenerator::ir::compilation_job::{AwgKey, AwgKind, ChannelIndex, DeviceKind, TriggerMode};
use codegenerator::ir::experiment::{AcquisitionType, Handle, SweepCommand};
use codegenerator::ir::{InitAmplitudeRegister, OscillatorFrequencySweepStep, ParameterOperation};
use codegenerator::signature::WaveformSignature;
use codegenerator::utils::normalize_phase;
use core::str;
use indexmap::IndexMap;
use seqc_tracker::awg::Awg;
use seqc_tracker::compressor::{compress_generator, merge_generators};
use seqc_tracker::seqc_generator::{SeqCGenerator, seqc_generator_from_device_traits};
use seqc_tracker::seqc_statements::SeqCVariant;
use seqc_tracker::seqc_tracker::{SeqCTracker, top_loop_stack_generators_have_statements};
use seqc_tracker::wave_index_tracker::{SignalType, WaveIndexTracker};
use std::collections::{HashMap, HashSet};

type WaveIndex = u32;
type UserRegister = u16;
type State = u16;
type CommandTableEntry<'a> = (u16, (&'a PlayWaveEvent, Option<u32>, i64));

fn find_last_phase_reset(sampled_event_list: &[AwgEvent]) -> Option<&AwgEvent> {
    sampled_event_list.iter().rev().find(|event| {
        matches!(
            event.kind,
            EventType::ResetPhase(..) | EventType::InitialResetPhase(..)
        )
    })
}

fn legacy_ordering(event: &AwgEvent) -> i32 {
    // Legacy ordering based on event type
    match event.kind {
        EventType::InitialResetPhase(..) => -5,
        EventType::LoopStepStart(..) => -4,
        EventType::LoopStepEnd(..) => -4,
        EventType::InitAmplitudeRegister(..) => -3,
        EventType::PushLoop(..) => -2,
        EventType::ResetPhase(..) => -1,
        EventType::ResetPrecompensationFilters(..) => -1,
        EventType::AcquireEvent(..) => -1,
        EventType::Iterate(..) => 2,
        _ => 0,
    }
}

fn has_priority(event: &AwgEvent) -> bool {
    match event.kind {
        EventType::InitAmplitudeRegister(..)
        | EventType::Match(..)
        | EventType::PlayHold(..)
        | EventType::SetOscillatorFrequency(..)
        | EventType::PlayWave(..) => false,
        EventType::AcquireEvent(..)
        | EventType::ChangeHwOscPhase(..)
        | EventType::InitialResetPhase(..)
        | EventType::Iterate(..)
        | EventType::LoopStepEnd(..)
        | EventType::LoopStepStart(..)
        | EventType::PpcSweepStepEnd(..)
        | EventType::PpcSweepStepStart(..)
        | EventType::PrngDropSample(..)
        | EventType::PrngSample(..)
        | EventType::PrngSetup(..)
        | EventType::PushLoop(..)
        | EventType::QaEvent(..)
        | EventType::ResetPhase(..)
        | EventType::ResetPrecompensationFilters(..)
        | EventType::TriggerOutput(..)
        | EventType::TriggerOutputBit(..) => true,
    }
}

fn cmp(e1: &AwgEvent, e2: &AwgEvent) -> std::cmp::Ordering {
    if matches!(e1.kind, EventType::AcquireEvent(..))
        && matches!(e2.kind, EventType::TriggerOutput(..))
    {
        return std::cmp::Ordering::Less;
    }
    // Some events do not have a priority, notably PLAY_WAVE.
    // This is required, play must happen after acquire for instance, something
    // that is not captured by the event list.
    if has_priority(e1) && has_priority(e2) {
        return std::cmp::Ordering::Greater;
    }
    legacy_ordering(e1).cmp(&legacy_ordering(e2))
}

fn sort_events(events: &mut [AwgEvent]) {
    // Sort the events based on their priorities and types.
    events.sort_by(cmp);
}

fn generate_if_else_tree(
    iterations: usize,
    variable: &str,
    step_factory: impl Fn(usize) -> String,
) -> Vec<String> {
    // Generate an if-else tree for the given number of iterations by
    // binary splitting the iteration space. Recursively go through the
    // binary representation of the iteration count.
    if iterations == 0 {
        return vec![];
    }
    let start_bit = iterations.ilog2() + 1;

    fn if_level(
        base: usize,
        bit: u32,
        iterations: usize,
        variable: &str,
        step_factory: &impl Fn(usize) -> String,
    ) -> Vec<String> {
        if bit == 0 {
            return vec![step_factory(base)];
        }
        let n_bit = bit - 1;
        let n_base = base + (1 << n_bit);
        if n_base < iterations {
            let mut result = vec![format!(
                "if ({variable} & 0b{:b}) {{  // {variable} >= {n_base}",
                1 << n_bit
            )];
            result.extend(
                if_level(n_base, n_bit, iterations, variable, step_factory)
                    .into_iter()
                    .map(|l| format!("  {l}")),
            );
            result.push(format!("}} else {{  // {variable} < {n_base}"));
            result.extend(
                if_level(base, n_bit, iterations, variable, step_factory)
                    .into_iter()
                    .map(|l| format!("  {l}")),
            );
            result.push("}".to_string());
            result
        } else {
            if_level(base, n_bit, iterations, variable, step_factory)
        }
    }

    if_level(0, start_bit, iterations, variable, &step_factory)
}

fn fmt_2_digits(value: f64) -> String {
    // Format a floating-point number with two decimal places, removing trailing zeros.
    let formatted = format!("{value:.2}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn make_command_table_comment(signature: &PlayWaveEvent) -> String {
    let mut parts = Vec::new();
    if let Some(hw_oscillator) = &signature.hw_oscillator {
        parts.push(format!("osc={}|{}", hw_oscillator.index, hw_oscillator.uid));
    }
    if let Some(phase) = signature.increment_phase {
        parts.push(format!("phase+={}", fmt_2_digits(phase)));
    }
    let ampl_register = format!("amp_{}", signature.amplitude_register);
    if let Some(ampl) = &signature.amplitude {
        match ampl {
            ParameterOperation::SET(value) => {
                parts.push(format!("{}={}", ampl_register, fmt_2_digits(*value)))
            }
            ParameterOperation::INCREMENT(value) => {
                if *value >= 0.0 {
                    parts.push(format!("{}+={}", ampl_register, fmt_2_digits(*value)));
                } else {
                    parts.push(format!("{}-={}", ampl_register, fmt_2_digits(-value)));
                }
            }
        }
    } else {
        parts.push(ampl_register);
    }
    if !signature.waveform.is_playzero() {
        parts.push(signature.waveform.signature_string());
    }
    parts.join("; ")
}

fn add_wait_trigger_statements(
    awg: &Awg,
    init_generator: &mut SeqCGenerator,
    deferred_function_calls: &mut SeqCGenerator,
) -> Result<()> {
    const DELAY_FIRST_AWG: f64 = 32.0 / device_traits::HDAWG_TRAITS.sampling_rate;
    const DELAY_OTHER_AWG: f64 = 32.0 / device_traits::HDAWG_TRAITS.sampling_rate;
    const DELAY_UHFQA: f64 = 128.0 / device_traits::UHFQA_TRAITS.sampling_rate;

    match awg.trigger_mode {
        TriggerMode::DioTrigger => {
            // HDAWG+UHFQA connected via DIO, no PQSC
            if awg.awg_key.index() == 0 {
                if awg.is_reference_clock_internal {
                    return Err(anyhow::anyhow!(
                        "HDAWG+UHFQA system can only be used with an external clock connected to HDAWG in order to prevent jitter."
                    ));
                }
                init_generator.add_function_call_statement(
                    "waitDigTrigger",
                    vec![SeqCVariant::Integer(1)],
                    None::<String>,
                );
                init_generator.add_function_call_statement(
                    "setDIO",
                    vec![SeqCVariant::String(String::from("0xffffffff"))],
                    None::<String>,
                );
                init_generator.add_function_call_statement(
                    "waitDIOTrigger",
                    vec![],
                    None::<String>,
                );
                let delay_first_awg_samples =
                    (awg.sampling_rate * DELAY_FIRST_AWG / 16.0).round() as Samples * 16;
                if delay_first_awg_samples > 0 {
                    deferred_function_calls.add_function_call_statement(
                        "playZero",
                        vec![SeqCVariant::Integer(delay_first_awg_samples)],
                        None::<String>,
                    );
                }
            } else {
                init_generator.add_function_call_statement(
                    "waitDIOTrigger",
                    vec![],
                    None::<String>,
                );
                let delay_other_awg_samples =
                    (awg.sampling_rate * DELAY_OTHER_AWG / 16.0).round() as Samples * 16;
                if delay_other_awg_samples > 0 {
                    deferred_function_calls.add_function_call_statement(
                        "playZero",
                        vec![SeqCVariant::Integer(delay_other_awg_samples)],
                        None::<String>,
                    );
                }
            }
        }
        TriggerMode::InternalReadyCheck => {
            // Standalone HDAWG
            // We don't need to do anything for alignment because ready check mechanism handles that.
        }
        TriggerMode::DioWait => {
            // UHFQA triggered by HDAWG
            init_generator.add_function_call_statement("waitDIOTrigger", vec![], None::<String>);
            let delay_uhfqa_samples =
                (awg.sampling_rate * DELAY_UHFQA / 8.0).round() as Samples * 8; // DELAY_UHFQA
            if delay_uhfqa_samples > 0 {
                init_generator.add_function_call_statement(
                    "playZero",
                    vec![SeqCVariant::Integer(delay_uhfqa_samples)],
                    None::<String>,
                );
            }
        }
        TriggerMode::InternalTriggerWait => {
            // SHFQC, internally triggered
            init_generator.add_function_call_statement(
                "waitDigTrigger",
                vec![SeqCVariant::Integer(1)],
                None::<String>,
            );
        }
        TriggerMode::ZSync => {
            if awg.device_kind.traits().supports_zsync {
                // Any instrument triggered directly via ZSync
                init_generator.add_function_call_statement(
                    "waitZSyncTrigger",
                    vec![],
                    None::<String>,
                );
            } else {
                // UHFQA triggered by PQSC (forwarded over DIO)
                init_generator.add_function_call_statement(
                    "waitDIOTrigger",
                    vec![],
                    None::<String>,
                );
            }
        }
    }
    Ok(())
}

fn is_spectroscopy_type(at: &AcquisitionType) -> bool {
    matches!(
        at,
        AcquisitionType::SPECTROSCOPY_IQ | AcquisitionType::SPECTROSCOPY_PSD
    )
}

/// Check that a match statement covers all states without gaps
fn check_state_coverage(
    sorted_ct_entries: &[CommandTableEntry],
    id_generator: impl Fn() -> String,
) -> Result<()> {
    let first = sorted_ct_entries
        .first()
        .expect("Internal error: No command table entries")
        .0;
    let last = sorted_ct_entries
        .last()
        .expect("Internal error: No last entry in command table")
        .0;
    if first != 0 || (last - first + 1) as usize != sorted_ct_entries.len() {
        return Err(anyhow::anyhow!(
            "States missing in match statement for {}. First state: {}, last state: {}, number of states: {}, expected {}, starting from 0.",
            id_generator(),
            first,
            last,
            sorted_ct_entries.len(),
            last + 1
        ));
    }
    Ok(())
}
struct SampledEventHandler<'a> {
    seqc_tracker: SeqCTracker,
    command_table_tracker: CommandTableTracker,
    shfppc_sweeper_config_tracker: SHFPPCSweeperConfigTracker,
    declarations_generator: SeqCGenerator,
    function_defs_generator: SeqCGenerator,
    wave_indices: WaveIndexTracker,
    qa_signals_by_handle: &'a HashMap<Handle, (String, AwgKey)>,
    feedback_register: Option<FeedbackRegisterIndex>,
    feedback_register_layout: &'a FeedbackRegisterLayout,
    feedback_register_config: FeedbackRegisterConfig,
    awg: &'a Awg,
    emit_timing_comments: bool,
    loop_stack: Vec<Samples>,
    last_event: Option<&'a AwgEvent>,
    match_parent_event: Option<&'a AwgEvent>,
    last_phase_reset: Option<&'a AwgEvent>,
    // If true, this AWG sources feedback data from Zsync. If False, it sources data
    // from the local bus. None means neither source is used. Using both is illegal.
    zsync_feedback_active: Option<bool>,
    match_command_table_entries: IndexMap<State, (&'a PlayWaveEvent, Option<WaveIndex>, i64)>,
    match_seqc_generators: IndexMap<State, SeqCGenerator>,
    current_sequencer_step: Option<Samples>,
    sequencer_step: Samples,
    acquisition_type: &'a AcquisitionType,
}

impl<'a> SampledEventHandler<'a> {
    pub fn new(
        awg: &'a Awg,
        qa_signals_by_handle: &'a HashMap<Handle, (String, AwgKey)>,
        feedback_register: Option<FeedbackRegisterIndex>,
        feedback_register_layout: &'a FeedbackRegisterLayout,
        emit_timing_comments: bool,
        use_current_sequencer_step: bool,
        acquisition_type: &'a AcquisitionType,
    ) -> Result<Self> {
        let dual_channel = !matches!(awg.signal_kind, AwgKind::SINGLE);
        let traits = awg.device_kind.traits();
        let mut init_generator = SeqCGenerator::new(traits, dual_channel);
        let mut deferred_function_calls = SeqCGenerator::new(traits, dual_channel);
        let function_defs_generator = SeqCGenerator::new(traits, dual_channel);
        let declarations_generator = SeqCGenerator::new(traits, dual_channel);
        add_wait_trigger_statements(awg, &mut init_generator, &mut deferred_function_calls)?;
        let seqc_tracker = SeqCTracker::new(
            init_generator,
            deferred_function_calls,
            awg,
            emit_timing_comments,
            awg.shf_output_mute_min_duration,
        )?;
        let current_sequencer_step = if use_current_sequencer_step {
            Some(0)
        } else {
            None
        };
        Ok(SampledEventHandler {
            command_table_tracker: CommandTableTracker::new(
                awg.device_kind.clone(),
                awg.signal_kind.clone(),
            ),
            shfppc_sweeper_config_tracker: SHFPPCSweeperConfigTracker::new(),
            seqc_tracker,
            declarations_generator,
            function_defs_generator,
            wave_indices: WaveIndexTracker::new(),
            qa_signals_by_handle,
            feedback_register_layout,
            feedback_register_config: FeedbackRegisterConfig::default(),
            awg,
            emit_timing_comments,
            loop_stack: Vec::new(),
            last_event: None,
            match_parent_event: None,
            last_phase_reset: None,
            match_command_table_entries: IndexMap::new(),
            match_seqc_generators: IndexMap::new(),
            current_sequencer_step,
            sequencer_step: 8,
            feedback_register,
            acquisition_type,
            zsync_feedback_active: None,
        })
    }

    fn create_seqc_generator(&self) -> SeqCGenerator {
        seqc_generator_from_device_traits(self.awg.device_kind.traits(), &self.awg.signal_kind)
    }

    fn increment_sequencer_step(&mut self) {
        if let Some(step) = self.current_sequencer_step {
            assert!(
                self.seqc_tracker.current_time() % self.sequencer_step == 0,
                "Internal error: Current time not aligned with sequencer grid."
            );
            let seq_step = self.seqc_tracker.current_time() / self.sequencer_step;
            if seq_step != step {
                self.seqc_tracker.add_variable_increment(
                    "current_seq_step",
                    SeqCVariant::Integer(seq_step - step),
                );
            }
            self.current_sequencer_step = Some(seq_step);
        }
    }

    fn assert_command_table(&self, msg: &str) -> Result<()> {
        if !self.awg.device_kind.traits().supports_command_table {
            return Err(anyhow::anyhow!(format!("Internal error: {}", msg)));
        }
        Ok(())
    }

    fn handle_playwave(
        &mut self,
        start: Samples,
        end: Samples,
        data: &'a PlayWaveEvent,
    ) -> Result<()> {
        let play_wave_channel = self.awg.play_channels.first().map(|c| c % 2);
        let sig_string = data.waveform.signature_string();
        let wave_index = self.get_wave_index(data, &sig_string, play_wave_channel)?;

        if let Some(match_parent_event) = self.match_parent_event
            && let EventType::Match(parent_data) = &match_parent_event.kind
        {
            let state = data
                .state
                .expect("Internal error: Parent match without state.");

            if let Some(handle) = &parent_data.handle {
                self.handle_playwave_on_feedback(
                    start,
                    handle,
                    state,
                    data,
                    wave_index,
                    match_parent_event.start,
                )?;
            } else if parent_data.user_register.is_some() {
                self.handle_playwave_for_user_register(
                    data,
                    &sig_string,
                    wave_index,
                    play_wave_channel,
                )?;
            } else if parent_data.prng_sample {
                self.handle_playwave_on_prng(
                    start,
                    state,
                    data,
                    wave_index,
                    &parent_data.section,
                    match_parent_event.start,
                )?;
            } else {
                return Err(anyhow::anyhow!(
                    "Internal error: Found match/case statement without handle, prng_sample or user_register."
                ));
            }
        } else {
            assert!(
                data.state.is_none(),
                "Internal error: State without parent match."
            );

            self.handle_regular_playwave(
                start,
                end,
                data,
                &sig_string,
                wave_index,
                play_wave_channel,
            )?;
        }
        Ok(())
    }

    fn handle_playwave_on_feedback(
        &mut self,
        start: Samples,
        handle: &Handle,
        state: State,
        signature: &'a PlayWaveEvent,
        wave_index: Option<WaveIndex>,
        parent_match_start: Samples,
    ) -> Result<()> {
        self.assert_command_table(&format!(
            "Found match/case statement for handle {handle} on unsupported device.",
        ))?;
        if let Some(previous_entry) = self.match_command_table_entries.get(&state) {
            if previous_entry != &(signature, wave_index, start - parent_match_start) {
                return Err(anyhow::anyhow!(
                    "Duplicate state {state} with different pulses for handle {handle} found.",
                ));
            }
        } else {
            self.match_command_table_entries
                .insert(state, (signature, wave_index, start - parent_match_start));
        }
        Ok(())
    }

    fn handle_playwave_for_user_register(
        &mut self,
        signature: &PlayWaveEvent,
        signature_string: &str,
        wave_index: Option<WaveIndex>,
        play_wave_channel: Option<ChannelIndex>,
    ) -> Result<()> {
        let state = signature
            .state
            .expect("Internal error: User register event without state.");
        if !self.match_seqc_generators.contains_key(&state) {
            self.match_seqc_generators
                .insert(state, self.create_seqc_generator());
        }
        let branch_generator = &mut self.match_seqc_generators.get_mut(&state).unwrap();
        if self.awg.device_kind.traits().supports_command_table {
            let ct_index = self
                .command_table_tracker
                .lookup_index_by_signature(signature)
                .unwrap_or(
                    self.command_table_tracker
                        .create_entry(signature, wave_index, false)?,
                );
            let comment = make_command_table_comment(signature);
            branch_generator.add_command_table_execution(
                SeqCVariant::Integer(ct_index as i64),
                None,
                Some(comment),
            );
        } else {
            branch_generator.add_play_wave_statement(signature_string, play_wave_channel);
        }
        Ok(())
    }

    fn handle_playwave_on_prng(
        &mut self,
        start: Samples,
        state: State,
        waveform: &'a PlayWaveEvent,
        wave_index: Option<u32>,
        section: &str,
        parent_match_start: Samples,
    ) -> Result<()> {
        self.assert_command_table("Found PRNG statement on unsupported device.")?;
        if let Some(entry) = self.match_command_table_entries.get(&state) {
            if entry != &(waveform, wave_index, start - parent_match_start) {
                return Err(anyhow::anyhow!(
                    "Duplicate state {} with different pulses for PRNG found in section {}.",
                    state,
                    section
                ));
            }
        } else {
            self.match_command_table_entries
                .insert(state, (waveform, wave_index, start - parent_match_start));
        }
        Ok(())
    }

    fn handle_regular_playwave(
        &mut self,
        start: Samples,
        end: Samples,
        signature: &PlayWaveEvent,
        signature_string: &str,
        wave_index: Option<WaveIndex>,
        play_wave_channel: Option<ChannelIndex>,
    ) -> Result<()> {
        // Playzeros were already added for match event, so only needed for the non-match case.
        self.seqc_tracker.add_required_playzeros(start)?;
        self.seqc_tracker.flush_deferred_phase_changes();
        self.seqc_tracker.add_timing_comment(end);

        if self.awg.device_kind.traits().supports_command_table {
            let ct_index = self
                .command_table_tracker
                .get_or_create_entry(signature, wave_index)?;
            let comment = make_command_table_comment(signature);
            self.seqc_tracker.add_command_table_execution(
                SeqCVariant::Integer(ct_index as i64),
                None,
                Some(comment),
            )?;
        } else {
            self.seqc_tracker
                .add_play_wave_statement(signature_string, play_wave_channel);
        }
        self.seqc_tracker.flush_deferred_function_calls();
        self.seqc_tracker.set_current_time(end);
        Ok(())
    }

    fn handle_playhold(&mut self, start: Samples, end: Samples) -> Result<()> {
        assert!(self.seqc_tracker.current_time() as Samples == start);
        // There cannot be any zero-length phase increments between the head playWave
        // and the playHold.
        assert!(!self.seqc_tracker.has_deferred_phase_changes());

        self.seqc_tracker.add_play_hold_statement(end - start)?;
        self.seqc_tracker.set_current_time(end);
        Ok(())
    }

    fn handle_amplitude_register_init(
        &mut self,
        start: Samples,
        data: &InitAmplitudeRegister,
    ) -> Result<()> {
        self.assert_command_table("Amplitude register init on unsupported device.")?;
        let signature = PlayWaveEvent {
            waveform: WaveformSignature::Pulses {
                length: 0,
                pulses: vec![],
            },
            state: None,
            hw_oscillator: None,
            amplitude_register: data.register,
            amplitude: Some(data.value.clone()),
            increment_phase: None,
            increment_phase_params: vec![],
            channels: vec![],
        };
        self.seqc_tracker.add_required_playzeros(start)?;
        self.seqc_tracker.flush_deferred_phase_changes();
        let ct_index = self
            .command_table_tracker
            .lookup_index_by_signature(&signature)
            .unwrap_or_else(|| {
                self.command_table_tracker
                    .create_entry(&signature, None, false)
                    .unwrap()
            });
        let comment = make_command_table_comment(&signature);
        self.seqc_tracker.add_command_table_execution(
            SeqCVariant::Integer(ct_index as i64),
            None,
            Some(comment),
        )?;
        Ok(())
    }

    fn handle_match(&mut self, data: &MatchEvent) -> Result<()> {
        if let Some(match_parent_event) = self.match_parent_event
            && let EventType::Match(parent_data) = &match_parent_event.kind
        {
            return Err(anyhow::anyhow!(
                "Simultaneous match events on the same physical AWG are not supported. \
                Affected handles/user registers: '{}' and '{}'",
                parent_data
                    .handle
                    .as_ref()
                    .map(|h| h.to_string())
                    .unwrap_or(
                        parent_data
                            .user_register
                            .map(|ur| ur.to_string())
                            .unwrap_or("PRNG".to_string())
                    ),
                data.handle.as_ref().map(|h| h.to_string()).unwrap_or(
                    data.user_register
                        .map(|ur| ur.to_string())
                        .unwrap_or("PRNG".to_string())
                ),
            ));
        } else {
            self.match_seqc_generators.clear();
        }
        Ok(())
    }

    fn handle_change_hw_oscillator_phase(&mut self, data: &ChangeHwOscPhase) -> Result<()> {
        // The `phase_resolution_range` is irrelevant here; for the CT phase a fixed
        // precision is used.
        const PHASE_RESOLUTION_CT: f64 = (1 << 24) as f64 / (2.0 * std::f64::consts::PI);
        let quantized_phase =
            normalize_phase((data.phase * PHASE_RESOLUTION_CT).round() / PHASE_RESOLUTION_CT);
        let signature = PlayWaveEvent {
            waveform: WaveformSignature::Pulses {
                length: 0,
                pulses: vec![],
            },
            state: None,
            hw_oscillator: data.hw_oscillator.clone(),
            amplitude_register: 0,
            amplitude: None,
            increment_phase: Some(quantized_phase),
            increment_phase_params: vec![data.parameter.clone()],
            channels: vec![],
        };

        let ct_index = self
            .command_table_tracker
            .get_or_create_entry(&signature, None)?;
        self.seqc_tracker.add_phase_change(
            SeqCVariant::Integer(ct_index as i64),
            Some(make_command_table_comment(&signature)),
        );
        Ok(())
    }

    fn handle_reset_precompensation_filters(
        &mut self,
        start: Samples,
        end: Samples,
        length: Samples,
    ) -> Result<()> {
        self.seqc_tracker.add_required_playzeros(start)?;
        if self.awg.device_kind.traits().supports_command_table {
            let signature_string = "precomp_reset";
            let ct_index = self.precomp_reset_ct_index(length)?;
            self.seqc_tracker.add_command_table_execution(
                SeqCVariant::Integer(ct_index as i64),
                None,
                Some(signature_string),
            )?;
            self.seqc_tracker.flush_deferred_function_calls();
            self.seqc_tracker.set_current_time(end);
        } else {
            self.seqc_tracker.add_function_call_statement(
                "setPrecompClear",
                vec![SeqCVariant::Integer(1)],
                None::<String>,
                false,
            );
            self.seqc_tracker.add_function_call_statement(
                "setPrecompClear",
                vec![SeqCVariant::Integer(0)],
                None::<String>,
                true,
            );
        }
        Ok(())
    }

    fn handle_acquire(&mut self, start: Samples) -> Result<()> {
        let args = vec![
            SeqCVariant::String(String::from("QA_INT_ALL")),
            if *self.acquisition_type == AcquisitionType::RAW {
                SeqCVariant::Integer(1)
            } else {
                SeqCVariant::Integer(0)
            },
        ];
        if start > self.seqc_tracker.current_time() as Samples {
            self.seqc_tracker.add_required_playzeros(start)?;
            self.seqc_tracker.add_function_call_statement(
                String::from("startQA"),
                args,
                None::<String>,
                true,
            );
        } else {
            let mut skip: bool = false;
            if let Some(last_event) = &self.last_event {
                if let EventType::AcquireEvent(..) = last_event.kind {
                    if last_event.start == start {
                        // If the last event was an acquire event, we skip this one
                        // if it has the same start time.
                        skip = true;
                    }
                }
            }
            if !skip {
                self.seqc_tracker.add_function_call_statement(
                    String::from("startQA"),
                    args,
                    None::<String>,
                    true,
                );
            }
        }
        Ok(())
    }

    fn handle_ppc_step_start(&mut self, start: Samples, data: &SweepCommand) -> Result<()> {
        // todo: Do we even need this here? Could be expressed via trigger commands
        self.seqc_tracker.add_required_playzeros(start)?;
        self.seqc_tracker.add_function_call_statement(
            "setTrigger",
            vec![SeqCVariant::Integer(1)],
            None::<String>,
            true,
        );
        self.shfppc_sweeper_config_tracker.add_step(data.clone());
        Ok(())
    }

    fn handle_ppc_step_end(&mut self, start: Samples) -> Result<()> {
        self.seqc_tracker.add_required_playzeros(start)?;
        self.seqc_tracker.add_function_call_statement(
            "setTrigger",
            vec![SeqCVariant::Integer(0)],
            None::<String>,
            true,
        );
        Ok(())
    }

    fn handle_set_oscillator_frequency(
        &mut self,
        start: Samples,
        data: &OscillatorFrequencySweepStep,
    ) -> Result<()> {
        self.seqc_tracker.add_required_playzeros(start)?;

        if matches!(self.awg.device_kind, DeviceKind::HDAWG) {
            self._handle_set_oscillator_frequency_hdawg(data)
        } else {
            self._handle_set_oscillator_frequency_shf(start, data)
        }
    }

    fn _handle_set_oscillator_frequency_hdawg(
        &mut self,
        data: &OscillatorFrequencySweepStep,
    ) -> Result<()> {
        let osc_id_symbol = format!("freq_osc_{}", data.osc_index);
        let counter_variable_name = format!("c_freq_osc_{}", data.osc_index);
        if data.iteration == 0 {
            let tree_variable_name = format!("arg_{osc_id_symbol}");
            let steps = generate_if_else_tree(data.parameter.count, &tree_variable_name, |i| {
                format!(
                    "setDouble(osc_node_{osc_id_symbol}, {});",
                    data.parameter.start + data.parameter.step * i as f64
                )
            })
            .join("\n  ");
            self.function_defs_generator.add_function_def(format!(
                "void set_{osc_id_symbol}(var {tree_variable_name}) {{\
                    \n  string osc_node_{osc_id_symbol} = \"oscs/{}/freq\";\n  {}\n}}\n",
                data.osc_index, steps
            ));
            self.declarations_generator
                .add_variable_declaration(&counter_variable_name, Some(SeqCVariant::Integer(0)))?;
            self.seqc_tracker
                .add_variable_assignment(&counter_variable_name, SeqCVariant::Integer(0));
        }
        self.seqc_tracker.add_function_call_statement(
            format!("set_{osc_id_symbol}"),
            vec![SeqCVariant::String(format!("{counter_variable_name}++"))],
            None::<String>,
            true,
        );
        Ok(())
    }

    fn _handle_set_oscillator_frequency_shf(
        &mut self,
        start: Samples,
        data: &OscillatorFrequencySweepStep,
    ) -> Result<()> {
        let osc_id_symbol = format!("freq_osc_{}", data.osc_index);
        let counter_variable_name = format!("c_freq_osc_{}", data.osc_index);
        if !self
            .declarations_generator
            .is_variable_declared(&counter_variable_name)
        {
            self.declarations_generator
                .add_variable_declaration(&counter_variable_name, Some(SeqCVariant::Integer(0)))?;
            self.declarations_generator.add_constant_definition(
                &osc_id_symbol,
                SeqCVariant::Integer(data.osc_index as i64),
                None::<String>,
            );
            self.declarations_generator.add_function_call_statement(
                "configFreqSweep",
                vec![
                    SeqCVariant::String(osc_id_symbol.clone()),
                    SeqCVariant::Float(data.parameter.start),
                    SeqCVariant::Float(data.parameter.step),
                ],
                None::<String>,
            );
        }
        if data.iteration == 0 {
            self.seqc_tracker
                .add_variable_assignment(&counter_variable_name, SeqCVariant::Integer(0));
        }
        self.seqc_tracker.add_function_call_statement(
            "setSweepStep",
            vec![
                SeqCVariant::String(osc_id_symbol),
                SeqCVariant::String(format!("{counter_variable_name}++")),
            ],
            None::<String>,
            true,
        );
        self.seqc_tracker.add_required_playzeros(start)?;
        Ok(())
    }

    fn handle_reset_phase(&mut self, start: Samples, initial: bool) -> Result<()> {
        self.seqc_tracker.discard_deferred_phase_changes();
        if initial {
            if start > self.seqc_tracker.current_time() as Samples {
                self.seqc_tracker.add_required_playzeros(start)?;
            }
        } else {
            self.seqc_tracker.add_required_playzeros(start)?;
        }
        self.seqc_tracker.add_function_call_statement(
            "resetOscPhase",
            vec![],
            None::<String>,
            !initial,
        );
        // Hack: we do not defer the initial phase reset, and emit as early as possible.
        // This way it is hidden in the lead time.

        if self.awg.device_kind.traits().supports_command_table {
            // `resetOscPhase()` resets the DDS phase, but it does not clear the phase
            // offset from the command table. We do that via a zero-length CT entry.
            let ct_index = self.command_table_tracker.create_phase_reset_entry();
            self.seqc_tracker.add_command_table_execution(
                SeqCVariant::Integer(ct_index as i64),
                None,
                None::<String>,
            )?;
        }
        Ok(())
    }

    fn handle_loop_step_start(&mut self, start: Samples) -> Result<()> {
        self.seqc_tracker.add_required_playzeros(start)?;
        self.seqc_tracker.append_loop_stack_generator(None, false)?;
        Ok(())
    }

    fn handle_loop_step_end(&mut self, start: Samples) -> Result<()> {
        self.seqc_tracker.add_required_playzeros(start)?;
        self.increment_sequencer_step();
        Ok(())
    }

    fn handle_push_loop(&mut self, start: Samples, data: &PushLoop) -> Result<()> {
        self.seqc_tracker.add_required_playzeros(start)?;
        self.seqc_tracker.flush_deferred_phase_changes();
        if self.current_sequencer_step.is_some() {
            assert!(self.seqc_tracker.current_time() % self.sequencer_step == 0);
            let new_step = self.seqc_tracker.current_time() / self.sequencer_step;
            self.current_sequencer_step = Some(new_step);
            self.seqc_tracker
                .add_variable_assignment("current_seq_step", SeqCVariant::Integer(new_step));
        }
        self.seqc_tracker.push_loop_stack_generator(None)?;
        if self.emit_timing_comments {
            let current_time = self.seqc_tracker.current_time();
            self.seqc_tracker
                .add_comment(format!("PUSH LOOP {data:?} current time = {current_time}",));
        }
        self.loop_stack.push(start);
        self.shfppc_sweeper_config_tracker
            .enter_loop(data.num_repeats);
        Ok(())
    }

    fn handle_iterate(&mut self, start: Samples, data: &Iterate) -> Result<()> {
        if top_loop_stack_generators_have_statements(&self.seqc_tracker) {
            if self.emit_timing_comments {
                let current_time = self.seqc_tracker.current_time();
                self.seqc_tracker
                    .add_comment(format!("ITERATE  {data:?}, current time = {current_time}",));
            }
            self.seqc_tracker.add_required_playzeros(start)?;
            self.seqc_tracker.flush_deferred_phase_changes();
            self.increment_sequencer_step();

            let mut loop_generator = seqc_generator_from_device_traits(
                self.awg.device_kind.traits(),
                &self.awg.signal_kind,
            );
            let open_generators = self
                .seqc_tracker
                .pop_loop_stack_generators()
                .expect("Internal error: No open loop generators found.");
            let loop_body = merge_generators(&open_generators.iter().collect::<Vec<_>>(), true);
            loop_generator.add_repeat(data.num_repeats, loop_body);
            if self.emit_timing_comments {
                loop_generator.add_comment(format!("Loop for {data:?}"))
            }
            let start_loop_event_start = self.loop_stack.pop().unwrap_or_else(|| {
                panic!("Internal error: No loop start event found for iteration at {start}.")
            });
            let delta = start - start_loop_event_start;
            self.seqc_tracker
                .set_current_time(start_loop_event_start + data.num_repeats as i64 * delta);
            if self.emit_timing_comments {
                loop_generator.add_comment(format!(
                    "Delta: {delta} current time after loop: {}, corresponding start event start: {}",
                    self.seqc_tracker.current_time(),
                    start_loop_event_start
                ));
            }
            self.seqc_tracker
                .append_loop_stack_generator(Some(loop_generator), true)?;
            self.seqc_tracker.append_loop_stack_generator(None, true)?;
        } else {
            self.seqc_tracker.pop_loop_stack_generators();
            self.loop_stack.pop();
        }

        self.shfppc_sweeper_config_tracker.exit_loop();
        Ok(())
    }

    fn handle_setup_prng(&mut self, data: &PrngSetup) -> Result<()> {
        if !self.awg.device_kind.traits().has_prng {
            return Ok(());
        }
        self.seqc_tracker.setup_prng(data.seed, data.range)?;
        Ok(())
    }

    fn handle_sample_prng(&mut self) -> Result<()> {
        if !self.awg.device_kind.traits().has_prng {
            return Ok(());
        }
        self.seqc_tracker
            .sample_prng(&mut self.declarations_generator)?;
        Ok(())
    }

    fn handle_drop_prng_sample(&mut self) -> Result<()> {
        if !self.awg.device_kind.traits().has_prng {
            return Ok(());
        }
        self.match_command_table_entries.clear();
        Ok(())
    }

    fn handle_trigger_output(&mut self, start: Samples, data: &TriggerOutput) -> Result<()> {
        self.seqc_tracker.add_required_playzeros(start)?;
        self.seqc_tracker
            .add_set_trigger_statement(data.state, true);
        Ok(())
    }

    fn handle_qa(&mut self, start: Samples, end: Samples, data: &QaEvent) -> Result<()> {
        let mut generator_channels: HashSet<u8> = HashSet::new();
        for play_event in &data.play_wave_events {
            generator_channels.extend(play_event.channels.iter());
            let waveform_signature = &play_event.waveform;
            self.wave_indices.add_numbered_wave(
                waveform_signature.signature_string(),
                SignalType::COMPLEX,
                *play_event.channels.first().unwrap_or(&0) as WaveIndex,
            );
        }
        let is_spectroscopy = is_spectroscopy_type(self.acquisition_type);

        let integration_channels = data
            .acquire_events
            .iter()
            .flat_map(|event| event.channels.iter())
            .collect::<Vec<_>>();
        let integrator_mask = if is_spectroscopy {
            // In spectroscopy mode, there are no distinct integrators that can be,
            // triggered, so the mask is ignored. By setting it to a non-null value
            // however, we ensure that the timestamp of the acquisition is correctly
            // latched.
            "1"
        } else if integration_channels.is_empty() {
            "QA_INT_NONE"
        } else {
            &integration_channels
                .iter()
                .map(|channel| format!("QA_INT_{channel}"))
                .collect::<Vec<String>>()
                .join("|")
        };
        let generator_mask = if is_spectroscopy {
            "0"
        } else if generator_channels.is_empty() {
            "QA_GEN_NONE"
        } else {
            let mut generator_channels = generator_channels.iter().collect::<Vec<_>>();
            generator_channels.sort();
            &generator_channels
                .iter()
                .map(|channel| format!("QA_GEN_{channel}"))
                .collect::<Vec<String>>()
                .join("|")
        };
        self.seqc_tracker.add_required_playzeros(start)?;
        self.seqc_tracker.flush_deferred_phase_changes();
        if end > self.seqc_tracker.current_time() {
            self.seqc_tracker.add_timing_comment(end);
        }

        if is_spectroscopy {
            self.seqc_tracker.add_startqa_shfqa_statement(
                generator_mask,
                integrator_mask,
                Some(0),
                Some(0),
                Some(0b10 | self.seqc_tracker.trigger_output_state()),
            );
            self.seqc_tracker
                .add_set_trigger_statement(self.seqc_tracker.trigger_output_state() & 0b1, true);
        } else {
            self.seqc_tracker.add_startqa_shfqa_statement(
                generator_mask,
                integrator_mask,
                Some(if matches!(self.acquisition_type, AcquisitionType::RAW) {
                    1
                } else {
                    0
                }),
                Some(self.feedback_register.unwrap_or(0)),
                Some(self.seqc_tracker.trigger_output_state()),
            );
        }
        if self.seqc_tracker.automute_playzeros() {
            self.seqc_tracker
                .add_play_zero_statement(end - start, true)?;
            self.seqc_tracker.flush_deferred_function_calls();
        }
        Ok(())
    }

    fn get_wave_index(
        &mut self,
        signature: &PlayWaveEvent,
        sig_string: &str,
        play_wave_channel: Option<ChannelIndex>,
    ) -> Result<Option<WaveIndex>> {
        if signature.waveform.is_playzero() && self.awg.device_kind.traits().supports_command_table
        {
            // all-zero pulse is played via play-zero command table entry
            Ok(None)
        } else {
            let wave_index = self.wave_indices.lookup_index_by_wave_id(sig_string);
            match wave_index {
                Some(index) => Ok(index),
                None => {
                    let signal_type = if self.awg.device_kind.traits().supports_binary_waves {
                        SignalType::SIGNAL(self.awg.signal_kind.clone())
                    } else {
                        SignalType::CSV
                    };
                    let new_wave_index = self
                        .wave_indices
                        .create_index_for_wave(sig_string, signal_type)?;
                    if let Some(new_wave_index) = new_wave_index {
                        self.declarations_generator.add_assign_wave_index_statement(
                            sig_string,
                            new_wave_index,
                            play_wave_channel,
                        );
                    }
                    Ok(new_wave_index)
                }
            }
        }
    }

    fn precomp_reset_ct_index(&mut self, length: Samples) -> Result<usize> {
        let signature = PlayWaveEvent {
            hw_oscillator: None,
            amplitude: None,
            amplitude_register: 0,
            increment_phase: None,
            increment_phase_params: vec![],
            waveform: WaveformSignature::Pulses {
                length,
                pulses: vec![],
            },
            state: None,
            channels: vec![],
        };
        let signature_string = "precomp_reset";
        let wave_index = self.wave_indices.lookup_index_by_wave_id(signature_string);
        Ok(if wave_index.is_none() {
            assert!(self.awg.device_kind.traits().supports_binary_waves);

            self.declarations_generator
                .add_zero_wave_declaration(signature_string, length)?;
            let wave_index = self
                .wave_indices
                .create_index_for_wave(
                    signature_string,
                    SignalType::SIGNAL(self.awg.signal_kind.clone()),
                )
                .map_err(|_| {
                    anyhow::anyhow!(
                        "Internal error: Could not create wave index for precompensation reset."
                    )
                })?
                .expect("Internal error: Wave index for precompensation reset not created.");
            let play_wave_channel = self.awg.play_channels.first().map(|c| c % 2);
            self.declarations_generator.add_assign_wave_index_statement(
                signature_string,
                wave_index,
                play_wave_channel,
            );

            self.command_table_tracker
                .create_precompensation_clear_entry(signature, wave_index)
        } else {
            self.command_table_tracker
                .lookup_index_by_signature(&signature)
                .expect("Internal error: Command table entry for precompensation reset not found.")
        })
    }

    // Calculate offset and mask into register for given qa_signal
    fn register_bitshift(
        &self,
        register: &FeedbackRegister,
        qa_signal: &str,
        force_local_alignment: bool, // default false
    ) -> (u8, u8, u16) {
        let mut register_bitshift = 0;
        let qa_signal_opt = Some(qa_signal.to_string());
        let mut found_width: Option<u8> = None;
        for register_item in &self.feedback_register_layout[register] {
            if register_item.signal == qa_signal_opt {
                found_width = Some(register_item.width);
                break;
            } else {
                register_bitshift += if force_local_alignment {
                    2
                } else {
                    register_item.width
                };
            }
        }
        let width = found_width.unwrap_or_else(|| {
            panic!("Internal error: Signal {qa_signal} not found in register {register:?}")
        });
        let mask = (1 << width) - 1;
        (register_bitshift, width, mask)
    }

    fn add_feedback_config(&mut self, handle: &Handle, local: bool) {
        let (qa_signal, qa_signal_awg) = &self.qa_signals_by_handle[handle];
        let mut index_select = None;
        let (codeword_bitshift, width, mask) = if local {
            let register = &FeedbackRegister::Local {
                device: qa_signal_awg.device_name().to_string(),
            };
            self.register_bitshift(register, qa_signal, true)
        } else {
            let register = &FeedbackRegister::Global {
                awg_key: qa_signal_awg.clone(),
            };
            let (qaregister_bitshift, width, mask) =
                self.register_bitshift(register, qa_signal, false);
            index_select = Some(qaregister_bitshift / 2);
            (
                (2 * self.awg.awg_key.index() as u8 + qaregister_bitshift % 2),
                width,
                mask,
            )
        };
        let path = if local {
            "QA_DATA_PROCESSED"
        } else {
            "ZSYNC_DATA_PROCESSED_A"
        };
        self.declarations_generator.add_function_call_statement(
            "configureFeedbackProcessing",
            vec![
                SeqCVariant::String(path.to_string()),
                SeqCVariant::Integer(codeword_bitshift as i64),
                SeqCVariant::Integer(width as i64), // todo: According to docs, should be decremented
                SeqCVariant::Integer(
                    self.feedback_register_config
                        .command_table_offset
                        .expect("Internal error: Command table offset not set.")
                        as i64,
                ),
            ],
            None::<String>,
        );
        self.feedback_register_config.codeword_bitshift = Some(codeword_bitshift);
        self.feedback_register_config.codeword_bitmask = Some(mask);
        self.feedback_register_config.register_index_select = index_select;
    }

    fn close_event_list(&mut self) -> Result<()> {
        if let Some(match_parent_event) = std::mem::take(&mut self.match_parent_event) {
            if let EventType::Match(params) = &match_parent_event.kind {
                if let Some(handle) = &params.handle {
                    self.close_event_list_for_handle(
                        handle,
                        params.local,
                        match_parent_event.start,
                        match_parent_event.end,
                    )?;
                } else if let Some(user_register) = params.user_register {
                    self.close_event_list_for_user_register(
                        user_register,
                        match_parent_event.start,
                        match_parent_event.end,
                    )?;
                } else if params.prng_sample {
                    self.close_event_list_for_prng_match(
                        params,
                        match_parent_event.start,
                        match_parent_event.end,
                    )?;
                }
            } else {
                panic!(
                    "Internal error: Unexpected event type for match parent event: {:?}",
                    match_parent_event.kind
                );
            }
        }
        Ok(())
    }

    fn close_event_list_for_handle(
        &mut self,
        handle: &Handle,
        local: bool,
        match_parent_event_start: Samples,
        match_parent_event_end: Samples,
    ) -> Result<()> {
        // Latency required for arrival of the waveform in the wave player.
        // See match_schedule.py for details
        const EXECUTETABLEENTRY_LATENCY: i64 = 3;

        let mut sorted_ct_entries: Vec<_> = self.match_command_table_entries.drain(..).collect();
        sorted_ct_entries.sort_by_key(|(idx, _)| *idx);
        assert!(
            !sorted_ct_entries.is_empty(),
            "Internal error: No command table entries found for handle {handle}",
        );
        check_state_coverage(&sorted_ct_entries, || format!("handle {handle}"))?;
        // Check whether we already have the same states in the command table:
        if let Some(command_table_offset) = self.feedback_register_config.command_table_offset {
            for (idx, (signature, wave_index, _)) in sorted_ct_entries {
                let current_ct_entry = self
                    .command_table_tracker
                    .get(idx as usize + command_table_offset as usize)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "Internal error: Error retrieving command table entry: {}",
                            e
                        )
                    })?;
                let current_wf_idx = current_ct_entry
                    .1
                    .as_object()
                    .expect("Internal error: Wrong datatype command table entry")
                    .get("waveform")
                    .expect("Internal error: No waveform in command table entry")
                    .get("index")
                    .map(|v| {
                        v.as_u64()
                            .expect("Internal error: Waveform index not a number")
                            as u32
                    });
                if current_ct_entry.0.waveform != signature.waveform || wave_index != current_wf_idx
                {
                    return Err(anyhow::anyhow!(
                        "Multiple command table entry sets for feedback (handle {}), do you use the same pulses and states?",
                        handle
                    ));
                }
            }
        } else {
            let offset = self.command_table_tracker.len() as u32;
            self.feedback_register_config.command_table_offset = Some(offset);
            self.add_feedback_config(handle, local);
            // Allocate command table entries
            for (idx, (signature, wave_index, _)) in sorted_ct_entries {
                let id2 = self
                    .command_table_tracker
                    .create_entry(signature, wave_index, false)?;
                assert!(offset as usize + idx as usize == id2);
            }
        }
        assert!(match_parent_event_start >= self.seqc_tracker.current_time());
        assert!(match_parent_event_start % self.sequencer_step == 0);
        self.seqc_tracker
            .add_required_playzeros(match_parent_event_start)?;
        self.seqc_tracker.flush_deferred_phase_changes();
        // Subtract the 3 cycles that we had added (see match_schedule.py)
        let latency = match_parent_event_start / self.sequencer_step
            - self.current_sequencer_step.expect("Sequencer step not set")
            - EXECUTETABLEENTRY_LATENCY;
        self.seqc_tracker.add_command_table_execution(
            SeqCVariant::String(String::from(if local {
                "QA_DATA_PROCESSED"
            } else {
                "ZSYNC_DATA_PROCESSED_A"
            })),
            Some(SeqCVariant::String(if latency >= 0 {
                format!("current_seq_step + {latency}")
            } else {
                format!("current_seq_step - {}", -latency)
            })),
            Some(format!("Match handle {handle}")),
        )?;
        let use_zsync = !local;
        if self.zsync_feedback_active.is_some() && self.zsync_feedback_active != Some(use_zsync) {
            return Err(anyhow::anyhow!(
                "Mixed feedback paths (global and local) are illegal"
            ));
        }
        self.zsync_feedback_active = Some(use_zsync);
        self.seqc_tracker.add_timing_comment(match_parent_event_end);
        self.seqc_tracker.flush_deferred_function_calls();
        self.seqc_tracker.set_current_time(match_parent_event_end);
        self.match_parent_event = None;
        Ok(())
    }

    fn close_event_list_for_user_register(
        &mut self,
        user_register: UserRegister,
        match_parent_event_start: Samples,
        match_parent_match_end: Samples,
    ) -> Result<()> {
        if user_register > 15 {
            return Err(anyhow::anyhow!(
                "Invalid user register {} in match statement. User registers must be between 0 and 15.",
                user_register
            ));
        }
        let var_name = format!("_match_user_register_{user_register}");
        let _ = self.declarations_generator.add_variable_declaration(
            &var_name,
            Some(SeqCVariant::String(format!("getUserReg({user_register})"))),
        ); // Ignore errors, it's fine if it's already declared
        self.seqc_tracker
            .add_required_playzeros(match_parent_event_start)?;
        self.seqc_tracker.flush_deferred_phase_changes();
        let mut if_generator = self.create_seqc_generator();
        let mut conditions_bodies: Vec<_> = self
            .match_seqc_generators
            .drain(..)
            .filter_map(|state_gen| {
                if state_gen.1.num_noncomment_statements() > 0 {
                    Some((
                        Some(format!("{var_name} == {}", state_gen.0)),
                        compress_generator(state_gen.1),
                    ))
                } else {
                    None
                }
            })
            .collect();
        // If there is no match, we just play zeros to keep the timing correct
        let mut play_zero_body = self.create_seqc_generator();
        play_zero_body.add_play_zero_statement(
            match_parent_match_end - self.seqc_tracker.current_time(),
            &mut None,
        )?;
        conditions_bodies.push((None, play_zero_body));
        let mut conditions = vec![];
        let mut bodies = vec![];
        for (cond, body) in conditions_bodies.into_iter() {
            if let Some(cond) = cond {
                conditions.push(cond);
            };
            bodies.push(body);
        }
        if_generator.add_if(conditions, bodies)?;
        self.seqc_tracker
            .append_loop_stack_generator(Some(if_generator), true)?;
        self.seqc_tracker.append_loop_stack_generator(None, true)?;
        self.seqc_tracker.add_timing_comment(match_parent_match_end);
        self.seqc_tracker.flush_deferred_function_calls();
        self.seqc_tracker.set_current_time(match_parent_match_end);
        self.match_parent_event = None;
        Ok(())
    }

    fn close_event_list_for_prng_match(
        &mut self,
        params: &MatchEvent,
        parent_match_start: Samples,
        parent_match_end: Samples,
    ) -> Result<()> {
        let mut sorted_ct_entries: Vec<_> = self.match_command_table_entries.drain(..).collect();
        sorted_ct_entries.sort_by_key(|(idx, _)| *idx);
        check_state_coverage(&sorted_ct_entries, || format!("section {}", params.section))?;
        let mut command_table_match_offset = self.command_table_tracker.len() as u32;
        // Allocate command table entries
        for (idx, (signature, wave_index, _)) in sorted_ct_entries.iter() {
            let id2 = self
                .command_table_tracker
                .create_entry(signature, *wave_index, true)? as u32;
            assert!(command_table_match_offset + *idx as u32 == id2);
        }

        let offset = self.seqc_tracker.commit_prng(command_table_match_offset);
        command_table_match_offset -= offset;

        assert!(parent_match_start >= self.seqc_tracker.current_time());

        self.seqc_tracker
            .add_required_playzeros(parent_match_start)?;
        self.seqc_tracker.flush_deferred_phase_changes();

        self.seqc_tracker
            .add_prng_match_command_table_execution(command_table_match_offset)?;
        self.seqc_tracker.add_timing_comment(parent_match_end);
        self.seqc_tracker.flush_deferred_function_calls();
        self.seqc_tracker.set_current_time(parent_match_end);
        self.match_parent_event = None;
        Ok(())
    }

    fn handle_sampled_event(&mut self, sampled_event: &'a AwgEvent) -> Result<()> {
        let start = sampled_event.start;
        let end = sampled_event.end;
        match sampled_event.kind {
            EventType::PlayWave(ref data) => {
                self.handle_playwave(start, end, data)?;
            }
            EventType::PlayHold() => self.handle_playhold(start, end)?,
            EventType::InitAmplitudeRegister(ref data) => {
                self.handle_amplitude_register_init(start, data)?
            }
            EventType::Match(ref data) => {
                self.handle_match(data)?;
                self.match_parent_event = Some(sampled_event);
            }
            EventType::ChangeHwOscPhase(ref data) => {
                self.handle_change_hw_oscillator_phase(data)?
            }
            EventType::ResetPrecompensationFilters(ref data) => {
                self.handle_reset_precompensation_filters(start, end, *data)?
            }
            EventType::AcquireEvent() => self.handle_acquire(start)?,
            EventType::PpcSweepStepStart(ref data) => self.handle_ppc_step_start(start, data)?,
            EventType::PpcSweepStepEnd() => self.handle_ppc_step_end(start)?,
            EventType::SetOscillatorFrequency(ref data) => {
                self.handle_set_oscillator_frequency(start, data)?
            }
            EventType::ResetPhase() => {
                // If multiple phase reset events are scheduled at the same time,
                // only process the *last* one. This way, RESET_PHASE takes
                // precedence over INITIAL_RESET_PHASE.
                if let Some(last_phase_reset) = self.last_phase_reset {
                    if std::ptr::eq(sampled_event as *const _, last_phase_reset) {
                        self.handle_reset_phase(start, false)?
                    }
                }
            }
            EventType::InitialResetPhase() => {
                if let Some(last_phase_reset) = self.last_phase_reset {
                    if std::ptr::eq(sampled_event as *const _, last_phase_reset) {
                        self.handle_reset_phase(start, true)?
                    }
                }
            }
            EventType::LoopStepStart() => self.handle_loop_step_start(start)?,
            EventType::LoopStepEnd() => self.handle_loop_step_end(start)?,
            EventType::PushLoop(ref data) => self.handle_push_loop(start, data)?,
            EventType::Iterate(ref data) => self.handle_iterate(start, data)?,
            EventType::PrngSetup(ref data) => self.handle_setup_prng(data)?,
            EventType::PrngSample() => self.handle_sample_prng()?,
            EventType::PrngDropSample() => self.handle_drop_prng_sample()?,
            EventType::TriggerOutput(ref data) => self.handle_trigger_output(start, data)?,
            EventType::TriggerOutputBit(_) => panic!(
                "Internal error: TriggerOutputBit should have been resolved in SampledEventHandler"
            ),
            EventType::QaEvent(ref data) => self.handle_qa(start, end, data)?,
        };
        self.last_event = Some(sampled_event);
        Ok(())
    }

    pub fn handle_sampled_events(&mut self, awg_events: &'a AwgEventList) -> Result<()> {
        for (_, awg_event_list) in awg_events.iter() {
            self.last_phase_reset = find_last_phase_reset(awg_event_list);
            for awg_event in awg_event_list.iter() {
                self.handle_sampled_event(awg_event)?;
            }
            self.close_event_list()?;
        }
        self.seqc_tracker.force_deferred_function_calls()?;
        Ok(())
    }

    pub fn handle_declarations(&mut self, wave_declarations: &[WaveDeclaration]) -> Result<()> {
        if self.emit_timing_comments {
            self.declarations_generator.add_comment(format!(
                "{:?}/{} sampling rate: {} Sa/s",
                self.awg.device_kind,
                self.awg.awg_key.index(),
                self.awg.sampling_rate
            ));
        }
        if self.current_sequencer_step.is_some() {
            self.declarations_generator.add_variable_declaration(
                String::from("current_seq_step"),
                Some(SeqCVariant::Integer(0)),
            )?;
        }

        let mut signature_infos: Vec<_> = wave_declarations
            .iter()
            .map(|wave_declaration| {
                (
                    (*wave_declaration.signature_string).clone(),
                    wave_declaration.length,
                    (wave_declaration.has_marker1, wave_declaration.has_marker2),
                )
            })
            .collect();
        signature_infos.sort();
        for (signature_string, length, (has_marker1, has_marker2)) in signature_infos.into_iter() {
            self.declarations_generator.add_wave_declaration(
                signature_string,
                length,
                has_marker1,
                has_marker2,
            )?;
        }
        Ok(())
    }

    fn assemble_seqc(&mut self) -> Result<String> {
        let mut seq_c_generators: Vec<SeqCGenerator> = Vec::new();
        while let Some(part) = self.seqc_tracker.pop_loop_stack_generators() {
            seq_c_generators.extend(part.into_iter().rev());
        }
        seq_c_generators.reverse();
        let main_generator = merge_generators(&seq_c_generators.iter().collect::<Vec<_>>(), true);
        let mut seq_c_generator = self.create_seqc_generator();
        if self.function_defs_generator.num_statements() > 0 {
            seq_c_generator.append_statements_from(&self.function_defs_generator);
            seq_c_generator.add_comment("=== END-OF-FUNCTION-DEFS ===");
        }
        seq_c_generator.append_statements_from(&self.declarations_generator);
        seq_c_generator.append_statements_from(&main_generator);
        Ok(seq_c_generator.generate_seq_c())
    }

    pub fn finish(mut self) -> Result<SeqcResults> {
        let seqc = self.assemble_seqc()?;
        let command_table = self.command_table_tracker.finish();
        let (command_table, parameter_phase_increment_map) = match command_table {
            Some(ct) => (
                Some(ct.command_table),
                Some(ct.parameter_phase_increment_map),
            ),
            None => (None, None),
        };
        Ok(SeqcResults {
            seqc,
            wave_indices: self.wave_indices.finish(),
            command_table,
            parameter_phase_increment_map,
            shf_sweeper_config: self.shfppc_sweeper_config_tracker.finish(),
            feedback_register_config: self.feedback_register_config,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn handle_sampled_events(
    awg_events: AwgEventList,
    awg: &Awg,
    qa_signals_by_handle: &HashMap<Handle, (String, AwgKey)>,
    wave_declarations: &[WaveDeclaration],
    feedback_register: Option<FeedbackRegisterIndex>,
    feedback_register_layout: &FeedbackRegisterLayout,
    emit_timing_comments: bool,
    has_readout_feedback: bool,
    acquisition_type: &AcquisitionType,
) -> Result<SeqcResults> {
    let mut handler = SampledEventHandler::new(
        awg,
        qa_signals_by_handle,
        feedback_register,
        feedback_register_layout,
        emit_timing_comments,
        has_readout_feedback,
        acquisition_type,
    )?;
    let mut awg_events = awg_events;
    for (_, event_list) in awg_events.iter_mut() {
        sort_events(event_list);
    }
    handler.handle_declarations(wave_declarations)?;
    handler.handle_sampled_events(&awg_events)?;
    handler.finish()
}
