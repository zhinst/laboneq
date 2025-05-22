// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::output_mute::OutputMute;
use crate::prng_tracker::PRNGTracker;
use crate::{
    compressor::compress_generator,
    seqc_generator::{SeqCGenerator, seqc_generator_from_device_and_signal_type},
    seqc_statements::SeqCVariant,
};
use anyhow::anyhow;
use codegenerator::Samples;
use std::ops::DerefMut;
use std::sync::RwLock;
use std::{fmt, sync::Arc};

pub struct SeqCTracker {
    deferred_function_calls: SeqCGenerator,
    deferred_phase_changes: SeqCGenerator,
    loop_stack_generators: Vec<Vec<Arc<RwLock<SeqCGenerator>>>>,
    sampling_rate: f64,
    delay: f64,
    device_type: String,
    signal_type: String,
    emit_timing_comments: bool,
    current_time: Samples,
    seqc_gen_prng: Option<Arc<RwLock<SeqCGenerator>>>,
    prng_tracker: Option<Arc<RwLock<PRNGTracker>>>,
    automute: Option<OutputMute>,
    active_trigger_outputs: u32,
}

impl SeqCTracker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        init_generator: SeqCGenerator,
        deferred_function_calls: SeqCGenerator,
        sampling_rate: f64,
        delay: f64,
        device_type: String,
        signal_type: String,
        emit_timing_comments: bool,
        automute_playzeros_min_duration: f64,
        automute_playzeros: bool,
    ) -> Result<Self> {
        let deferred_phase_changes =
            seqc_generator_from_device_and_signal_type(&device_type, &signal_type)?;

        let loop_stack_generators = vec![vec![
            Arc::new(RwLock::new(init_generator)),
            Arc::new(RwLock::new(seqc_generator_from_device_and_signal_type(
                &device_type,
                &signal_type,
            )?)),
        ]];

        let mut tracker = Self {
            deferred_function_calls,
            deferred_phase_changes,
            loop_stack_generators,
            sampling_rate,
            delay,
            device_type,
            signal_type,
            emit_timing_comments,
            current_time: 0,
            seqc_gen_prng: None,
            prng_tracker: None,
            automute: None,
            active_trigger_outputs: 0,
        };

        if automute_playzeros {
            tracker.automute = Some(OutputMute::new(
                &tracker.device_type,
                automute_playzeros_min_duration,
            )?);
        }

        Ok(tracker)
    }

    pub fn automute_playzeros(&self) -> bool {
        self.automute.is_some()
    }

    pub fn current_loop_stack_generator(&self) -> Arc<RwLock<SeqCGenerator>> {
        self.loop_stack_generators
            .last()
            .expect("Loop stack should not be empty")
            .last()
            .expect("Inner loop stack should not be empty")
            .clone()
    }

    /// If `current_time` precedes the scheduled start of the event, emit playZero to catch up.
    /// If muting is enabled, the emitted playZeros are muted if possible.
    /// Also clears deferred function calls within the context of the new playZero.
    /// Returns the updated current time.
    pub fn add_required_playzeros(&mut self, start: i64) -> Result<Samples> {
        if start > self.current_time as i64 {
            let play_zero_samples = (start - self.current_time as i64) as Samples;

            if !self.mute_samples(play_zero_samples)? {
                self.add_timing_comment(self.current_time + play_zero_samples);
                self.current_loop_stack_generator()
                    .write()
                    .expect("Failed to lock generator")
                    .add_play_zero_statement(
                        play_zero_samples,
                        &mut Some(&mut self.deferred_function_calls),
                    )?;
            }
            self.current_time += play_zero_samples;
        }
        Ok(self.current_time)
    }

    /// Mute samples to be played.
    ///
    /// If the length of samples exceeds `samples_min`, muting is applied,
    /// otherwise no action is done.
    ///
    fn mute_samples(&mut self, samples: Samples) -> Result<bool> {
        let automute = match &mut self.automute {
            Some(automute) => {
                if samples < automute.samples_min {
                    return Ok(false);
                } else {
                    automute
                }
            }
            None => return Ok(false),
        };
        if samples < automute.samples_min {
            return Ok(false);
        }
        let delay_engage = automute.delay_engage;
        let delay_disengage = automute.delay_disengage;

        self.add_play_zero_statement(delay_engage, false)?;
        self.add_function_call_statement(
            "setTrigger",
            vec![SeqCVariant::Integer(1)],
            None::<&str>,
            true, // deferred
        );
        self.add_play_zero_statement(samples - delay_engage - delay_disengage, false)?;
        self.add_function_call_statement(
            "setTrigger",
            vec![SeqCVariant::Integer(0)],
            None::<&str>,
            true, // deferred
        );
        self.add_play_zero_statement(delay_disengage, false)?;

        Ok(true)
    }

    /// Force flushing deferred function calls, typically needed only at the sequence end.
    ///
    /// There may be deferred function calls issued *at the end of the sequence*.
    /// There will be no playWave or playZero that could flush them, so this function
    /// will force a flush.
    ///
    /// This function should not be needed anywhere but at the end of the sequence.
    /// In the future, maybe at the end of a loop iteration?)
    pub fn force_deferred_function_calls(&mut self) -> Result<()> {
        if !self.has_deferred_function_calls() {
            return Ok(());
        }
        if self.device_type.to_uppercase() == "SHFQA" {
            // SHFQA does not support waitWave()
            self.add_play_zero_statement(32, false)?;
        } else {
            self.add_function_call_statement("waitWave", vec![], None::<&str>, false);
        }
        self.flush_deferred_function_calls();
        Ok(())
    }

    /// Emit the deferred function calls *now*.
    pub fn flush_deferred_function_calls(&mut self) {
        if self.has_deferred_function_calls() {
            self.current_loop_stack_generator()
                .write()
                .expect("Failed to lock generator")
                .merge_statements_from(&self.deferred_function_calls);
            self.deferred_function_calls.clear();
        }
    }

    /// Emit deferred phase changes (zero-time command table entries).
    ///
    /// Phase changes (i.e. command table entries that take zero time) are emitted
    /// as late as possible, for example, we exchange it with a playZero that comes
    /// immediately after. This allows the sequencer to add more work to the wave player,
    /// and avoid gaps in the playback more effectively.
    pub fn flush_deferred_phase_changes(&mut self) {
        if self.has_deferred_phase_changes() {
            self.current_loop_stack_generator()
                .write()
                .expect("Failed to lock generator")
                .merge_statements_from(&self.deferred_phase_changes);
            self.deferred_phase_changes.clear();
        }
    }

    /// Discard pending deferred phase changes without applying them.
    ///
    /// Sometimes, it is useful to discard pending phase changes without ever
    /// applying them.
    ///
    /// For example, when we've accumulated deferred phase increments, but then
    /// encounter a phase reset, the phase changes are nullified anyway.
    pub fn discard_deferred_phase_changes(&mut self) {
        self.deferred_phase_changes.clear();
    }

    /// Check if there are any pending deferred function calls.
    pub fn has_deferred_function_calls(&self) -> bool {
        self.deferred_function_calls.num_statements() > 0
    }

    /// Check if there are any pending deferred phase changes.
    pub fn has_deferred_phase_changes(&self) -> bool {
        self.deferred_phase_changes.num_statements() > 0
    }

    /// Add a timing comment to the current generator if enabled.
    pub fn add_timing_comment(&mut self, end_samples: Samples) {
        if !self.emit_timing_comments {
            return;
        }
        let start_time_s = self.current_time as f64 / self.sampling_rate - self.delay;
        let end_time_s = end_samples as f64 / self.sampling_rate - self.delay;
        let start_time_ns = (start_time_s * 1e10).round() / 10.0;
        let end_time_ns = (end_time_s * 1e10).round() / 10.0;

        let comment = format!(
            "{} - {} , {} ns - {} ns ",
            self.current_time, end_samples, start_time_ns, end_time_ns
        );
        self.add_comment(&comment);
    }

    /// Add a comment string to the current generator.
    pub fn add_comment<S: Into<String>>(&mut self, comment: S) {
        self.current_loop_stack_generator()
            .write()
            .expect("Failed to lock generator")
            .add_comment(comment);
    }

    /// Add a function call statement to the appropriate generator.
    pub fn add_function_call_statement<S1: Into<String>, S2: Into<String>>(
        &mut self,
        name: S1,
        args: Vec<SeqCVariant>, // default: Empty
        assign_to: Option<S2>,  // default: None
        deferred: bool,         // default: false
    ) {
        if deferred {
            self.deferred_function_calls
                .add_function_call_statement(name, args, assign_to);
        } else {
            self.current_loop_stack_generator()
                .write()
                .expect("Failed to lock generator")
                .add_function_call_statement(name, args, assign_to);
        }
    }

    /// Add playZero statement, potentially incrementing the time counter.
    ///
    /// # Arguments:
    ///
    /// * `num_samples`: Number of samples to play.
    /// * `increment_counter`: If true, increment the current time by `num_samples`.
    pub fn add_play_zero_statement(
        &mut self,
        num_samples: Samples,
        increment_counter: bool, // default: false
    ) -> Result<()> {
        if increment_counter {
            self.current_time += num_samples;
        }
        self.current_loop_stack_generator()
            .write()
            .expect("Failed to lock generator")
            .add_play_zero_statement(num_samples, &mut Some(&mut self.deferred_function_calls))
    }

    /// Add playHold statement.
    pub fn add_play_hold_statement(&mut self, num_samples: Samples) -> Result<()> {
        self.current_loop_stack_generator()
            .write()
            .expect("Failed to lock generator")
            .add_play_hold_statement(num_samples, &mut Some(&mut self.deferred_function_calls))
    }

    /// Add playWave statement.
    pub fn add_play_wave_statement<S: Into<String>>(&mut self, wave_id: S, channel: Option<u16>) {
        self.current_loop_stack_generator()
            .write()
            .expect("Failed to lock generator")
            .add_play_wave_statement(wave_id, channel);
    }

    /// Add command table execution statement.
    pub fn add_command_table_execution<S: Into<String>>(
        &mut self,
        ct_index: SeqCVariant,
        latency: Option<SeqCVariant>, // default: None
        comment: Option<S>,           // default: ""
    ) -> Result<()> {
        if let Some(SeqCVariant::Integer(lat)) = &latency {
            if *lat < 31 {
                return Err(anyhow!("Latency must be >= 31 if specified, was {}", lat).into());
            }
        }
        self.current_loop_stack_generator()
            .write()
            .expect("Failed to lock generator")
            .add_command_table_execution(ct_index, latency, comment);
        Ok(())
    }

    /// Add a deferred phase change (zero-time command table execution).
    pub fn add_phase_change<S: Into<String>>(&mut self, ct_index: SeqCVariant, comment: Option<S>) {
        self.deferred_phase_changes
            .add_command_table_execution(ct_index, None, comment);
    }

    /// Add variable assignment statement.
    pub fn add_variable_assignment<S: Into<String>>(
        &mut self,
        variable_name: S,
        value: SeqCVariant,
    ) {
        self.current_loop_stack_generator()
            .write()
            .expect("Failed to lock generator")
            .add_variable_assignment(variable_name, value);
    }

    /// Add variable increment statement.
    pub fn add_variable_increment<S: Into<String>>(
        &mut self,
        variable_name: S,
        value: SeqCVariant,
    ) {
        self.current_loop_stack_generator()
            .write()
            .expect("Failed to lock generator")
            .add_variable_increment(variable_name, value);
    }

    pub fn append_loop_stack_generator(
        &mut self,
        generator: Option<SeqCGenerator>, // default: None
        always: bool,                     // default: false
    ) -> Result<Arc<RwLock<SeqCGenerator>>> {
        let generator = match generator {
            Some(generator) => generator,
            None => {
                seqc_generator_from_device_and_signal_type(&self.device_type, &self.signal_type)?
            }
        };
        let top_of_stack = self.loop_stack_generators.last_mut().ok_or_else(|| {
            anyhow!("Loop stack should not be empty in append_loop_stack_generator")
        })?;
        if always
            || top_of_stack.is_empty()
            || top_of_stack
                .last()
                .is_some_and(|g| g.read().expect("Failed to lock generator").num_statements() > 0)
        {
            top_of_stack.push(Arc::new(RwLock::new(generator)));
        }
        top_of_stack.last().map(Arc::clone).ok_or_else(|| {
            anyhow!("Loop stack should not be empty after appending generator").into()
        })
    }

    /// Pushes a new level onto the loop stack, optionally with an initial generator.
    pub fn push_loop_stack_generator(&mut self, generator: Option<SeqCGenerator>) -> Result<()> {
        self.loop_stack_generators.push(vec![]);
        self.append_loop_stack_generator(generator, false)
            .map(|_| ())
    }

    /// Pops the top level of generators from the loop stack, compresses and returns them.
    /// This also removes all other generators from the object, in particular `seqc_gen_prng`.
    pub fn pop_loop_stack_generators(&mut self) -> Option<Vec<SeqCGenerator>> {
        if self.loop_stack_generators.is_empty() {
            return None;
        }
        self.seqc_gen_prng = None;
        Some(match self.loop_stack_generators.pop() {
            Some(generators) => generators
                .into_iter()
                .map(|generator| {
                    compress_generator(
                        Arc::try_unwrap(generator)
                            .expect("Generator still has references")
                            .into_inner()
                            .expect("Failed to unwrap RwLock inner value"),
                    )
                })
                .collect(),
            None => Vec::new(),
        })
    }

    pub fn top_loop_stack_generators(&self) -> Option<&Vec<Arc<RwLock<SeqCGenerator>>>> {
        self.loop_stack_generators.last()
    }

    /// Insert a placeholder for setting up the PRNG
    ///
    /// In particular the offset into the command table can be efficiently encoded in
    /// range of the PRNG, which we do not know yet.
    ///
    /// This function returns a reference to a PRNGTracker through which the code
    /// generator can adjust the values and eventually commit them.
    /// Once committed, they values cannot be changed, and the PRNG has to be setup anew.
    pub fn setup_prng(&mut self, seed: Option<u32>, prng_range: Option<u32>) -> Result<()> {
        // Check if PRNG is already set up
        if self.prng_tracker.is_some() {
            return Err(anyhow!("PRNG already set up, cannot set up again",).into());
        }
        self.seqc_gen_prng = Some(self.append_loop_stack_generator(None, false)?);
        let mut prng_tracker = PRNGTracker::new();
        self.append_loop_stack_generator(None, false)?; // Continuation

        if let Some(seed) = seed {
            prng_tracker.set_seed(seed);
        }
        if let Some(range) = prng_range {
            prng_tracker.set_range(range);
        }
        self.prng_tracker = Some(Arc::new(RwLock::new(prng_tracker)));
        Ok(())
    }

    /// Discards the current PRNG setup state.
    pub fn drop_prng(&mut self) -> Result<()> {
        if self.prng_tracker.is_none() {
            return Err(anyhow!("PRNG not set up, cannot drop").into());
        }
        self.prng_tracker = None;
        Ok(())
    }

    /// Adds a command table execution statement indexed by the PRNG value plus an offset.
    pub fn add_prng_match_command_table_execution(&mut self, offset: i64) -> Result<()> {
        assert!(
            self.prng_tracker
                .as_ref()
                .expect("PRNG not set up")
                .read()
                .expect("Failed to lock PRNG tracker")
                .is_committed(),
            "PRNG not committed"
        );
        let index_str = if offset != 0 {
            format!("prng_value + {}", offset)
        } else {
            String::from("prng_value")
        };
        self.add_command_table_execution::<String>(SeqCVariant::String(index_str), None, None)?;
        Ok(())
    }

    pub fn commit_prng(&mut self) {
        self.prng_tracker
            .as_mut()
            .expect("PRNG not set up")
            .write()
            .expect("Failed to lock PRNG tracker")
            .commit(
                self.seqc_gen_prng
                    .as_mut()
                    .unwrap()
                    .write()
                    .expect("Failed to lock generator")
                    .deref_mut(),
            );
    }

    /// Adds commands to sample the PRNG value into the 'prng_value' variable.
    pub fn sample_prng(&mut self, declarations_generator: &mut SeqCGenerator) -> Result<()> {
        let variable_name = "prng_value";
        if !declarations_generator.is_variable_declared(variable_name) {
            declarations_generator.add_variable_declaration(variable_name, None)?;
        }
        self.add_function_call_statement("getPRNGValue", Vec::new(), Some(variable_name), false);
        Ok(())
    }

    /// Returns an immutable reference to the PRNG tracker state, if it exists.
    pub fn prng_tracker(&self) -> Option<Arc<RwLock<PRNGTracker>>> {
        self.prng_tracker.as_ref().map(Arc::clone)
    }

    /// Add a setTrigger statement.
    pub fn add_set_trigger_statement(&mut self, value: u32, deferred: bool) {
        self.add_function_call_statement(
            "setTrigger",
            vec![SeqCVariant::Integer(value as i64)],
            None::<&str>,
            deferred,
        );
        self.active_trigger_outputs = value;
    }

    /// Add a startQA statement (SHFQA specific).
    /// Handles optional arguments and their dependencies.
    pub fn add_startqa_shfqa_statement<S1: Into<String>, S2: Into<String>>(
        &mut self,
        generator_mask: S1,
        integrator_mask: S2,
        monitor: Option<u8>,            // Default: None
        feedback_register: Option<u32>, // Default: None
        trigger: Option<u32>,           // Default: None
    ) {
        let mut args = vec![
            SeqCVariant::String(generator_mask.into()),
            SeqCVariant::String(integrator_mask.into()),
        ];
        let mut final_trigger_val = 0; // Default trigger value

        if let Some(mon) = monitor {
            assert!(mon == 0 || mon == 1);
            args.push(SeqCVariant::Integer(mon as i64));
        }
        if let Some(reg) = feedback_register {
            assert!(
                monitor.is_some(),
                "Feedback register specified with monitor off in startQA"
            );
            args.push(SeqCVariant::Integer(reg as i64));
        }
        if let Some(trig) = trigger {
            assert!(
                monitor.is_some(),
                "Trigger specified with monitor off in startQA"
            );
            assert!(
                feedback_register.is_some(),
                "Trigger specified with no feedback register in startQA"
            );
            args.push(SeqCVariant::Integer(trig as i64));
            final_trigger_val = trig;
        }

        self.add_function_call_statement("startQA", args, None::<&str>, true); // Deferred = true
        self.active_trigger_outputs = final_trigger_val;
    }

    /// Get the last set state of the trigger outputs.
    pub fn trigger_output_state(&self) -> u32 {
        self.active_trigger_outputs
    }

    /// Get the current time in samples.
    pub fn current_time(&self) -> Samples {
        self.current_time
    }

    /// Set the current time in samples.
    pub fn set_current_time(&mut self, time: Samples) {
        self.current_time = time;
    }
}

impl fmt::Debug for SeqCTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SeqCTracker")
            .field("sampling_rate", &self.sampling_rate)
            .field("delay", &self.delay)
            .field("device_type", &self.device_type)
            .field("signal_type", &self.signal_type)
            .field("emit_timing_comments", &self.emit_timing_comments)
            .field("current_time", &self.current_time)
            .field("active_trigger_outputs", &self.active_trigger_outputs)
            .field(
                "has_deferred_calls",
                &(self.deferred_function_calls.num_statements() > 0),
            )
            .field("has_deferred_phase", &self.has_deferred_phase_changes())
            .field("automute_enabled", &self.automute.is_some())
            .field("prng_enabled", &self.prng_tracker.is_some())
            .field("loop_stack_depth", &self.loop_stack_generators.len())
            // Avoid printing full generators unless necessary
            // .field("loop_stack_generators", &self.loop_stack_generators)
            .finish()
    }
}
