// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::seqc_statements::{SeqCStatement, SeqCVariant};
use crate::wave_index_tracker::WaveIndex;
use crate::{Result, Samples};
use anyhow::anyhow;
use codegenerator::device_traits::DeviceTraits;
use codegenerator::ir::compilation_job::{AwgKind, ChannelIndex, DeviceKind};

type WaveId = str;
type Variable = str;

type VariableInternal = String;

static MIN_PLAY_ZERO_HOLD: Samples = 512 + 128;

fn format_comment(comment: &Option<String>) -> String {
    if let Some(comment) = comment {
        if !comment.is_empty() {
            return format!("  // {comment}");
        }
    }
    String::new()
}

fn indent(s: &str, prefix: &str) -> String {
    s.lines()
        .flat_map(|line| [prefix, line, "\n"].into_iter())
        .collect()
}

fn gen_wave_declaration_placeholder(
    dual_channel: bool,
    wave_id: &WaveId,
    length: Samples,
    has_marker1: bool,
    has_marker2: bool,
) -> String {
    let markers_declaration1 = if has_marker1 { ",true" } else { "" };
    let markers_declaration2 = if has_marker2 { ",true" } else { "" };
    if dual_channel {
        format!(
            "wave w{wave_id}_i = placeholder({length}{markers_declaration1});\nwave w{wave_id}_q = placeholder({length}{markers_declaration2});\n"
        )
    } else {
        format!(
            "wave w{wave_id} = placeholder({length}{markers_declaration1}{markers_declaration2});\n"
        )
    }
}

fn gen_zero_wave_declaration_placeholder(
    dual_channel: bool,
    wave_id: &WaveId,
    length: Samples,
) -> String {
    if dual_channel {
        format!("wave w{wave_id}_i = zeros({length});\nwave w{wave_id}_q = w{wave_id}_i;\n")
    } else {
        format!("wave w{wave_id} = zeros({length});\n")
    }
}

fn build_wave_channel_assignment(
    dual_channel: bool,
    wave_id: &WaveId,
    channel: Option<ChannelIndex>,
    supports_digital_iq_modulation: bool,
) -> String {
    if dual_channel && supports_digital_iq_modulation {
        format!("1,2,w{wave_id}_i,1,2,w{wave_id}_q")
    } else if dual_channel {
        format!("w{wave_id}_i,w{wave_id}_q")
    } else if channel.is_some() && channel.unwrap() == 1 {
        format!("1,\"\",2,w{wave_id}")
    } else {
        format!("w{wave_id}")
    }
}

#[derive(Clone, Debug)]
pub struct SeqCGenerator {
    device_traits: &'static DeviceTraits,
    dual_channel: bool,

    statements: Vec<SeqCStatement>,
    symbols: std::collections::HashSet<VariableInternal>,
}

impl std::hash::Hash for SeqCGenerator {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.statements.hash(state);
        // Hash the symbols in a deterministic order
        let mut symbols: Vec<_> = self.symbols.iter().collect();
        symbols.sort();
        for symbol in symbols {
            symbol.hash(state);
        }
    }
}

impl PartialEq for SeqCGenerator {
    fn eq(&self, other: &Self) -> bool {
        self.statements == other.statements && self.symbols == other.symbols
    }
}

impl Default for SeqCGenerator {
    fn default() -> Self {
        Self::new(DeviceKind::SHFQA.traits(), false)
    }
}

impl SeqCGenerator {
    pub fn new(device_traits: &'static DeviceTraits, dual_channel: bool) -> Self {
        Self {
            device_traits,
            dual_channel,
            statements: Vec::new(),
            symbols: std::collections::HashSet::new(),
        }
    }

    pub fn create(&self) -> Self {
        Self::new(self.device_traits, self.dual_channel)
    }

    pub fn statements(&self) -> &Vec<SeqCStatement> {
        &self.statements
    }

    pub fn num_statements(&self) -> usize {
        self.statements.len()
    }

    pub fn num_noncomment_statements(&self) -> usize {
        self.statements
            .iter()
            .filter(|s| !matches!(s, SeqCStatement::Comment { .. }))
            .count()
    }

    pub fn append_statements_from(&mut self, other: &SeqCGenerator) {
        // TODO: Consider whether it's possible to move instead of clone
        // or to use shared pointers if this turns out to be a performance issue
        self.statements.append(&mut other.statements.to_vec());
    }

    pub fn clear(&mut self) {
        self.statements.clear();
        // TODO: clear symbols? The original implementation does not
    }

    pub fn add_statement(&mut self, statement: SeqCStatement) {
        self.statements.push(statement);
    }

    pub fn add_comment<S: Into<String>>(&mut self, comment: S) {
        self.statements.push(SeqCStatement::Comment {
            text: comment.into(),
        });
    }

    pub fn add_function_call_statement<S1: Into<String>, S2: Into<String>>(
        &mut self,
        name: S1,
        args: Vec<SeqCVariant>,
        assign_to: Option<S2>,
    ) {
        self.statements.push(SeqCStatement::FunctionCall {
            name: name.into(),
            args,
            assign_to: assign_to.map(|s| s.into()),
        });
    }

    pub fn add_wave_declaration<S: Into<String>>(
        &mut self,
        wave_id: S,
        length: Samples,
        has_marker1: bool,
        has_marker2: bool,
    ) -> Result<()> {
        if length < self.device_traits.min_play_wave.into() {
            return Err(anyhow!(
                "Attempting to emit placeholder({}), which is below the minimum \
                waveform length {} of device '{}' (sample multiple is {})",
                length,
                self.device_traits.min_play_wave,
                self.device_traits.type_str,
                self.device_traits.sample_multiple,
            )
            .into());
        }
        self.statements.push(SeqCStatement::WaveDeclaration {
            wave_id: wave_id.into(),
            length,
            has_marker1,
            has_marker2,
        });
        Ok(())
    }

    pub fn add_zero_wave_declaration<S: Into<String>>(
        &mut self,
        wave_id: S,
        length: Samples,
    ) -> Result<()> {
        if length < self.device_traits.min_play_wave.into() {
            return Err(anyhow!(
                "Attempting to emit placeholder({}), which is below the minimum \
                waveform length {} of device '{}' (sample multiple is {})",
                length,
                self.device_traits.min_play_wave,
                self.device_traits.type_str,
                self.device_traits.sample_multiple,
            )
            .into());
        }
        self.statements.push(SeqCStatement::ZeroWaveDeclaration {
            wave_id: wave_id.into(),
            length,
        });
        Ok(())
    }

    pub fn add_constant_definition<S1: Into<String>, S2: Into<String>>(
        &mut self,
        name: S1,
        value: SeqCVariant,
        comment: Option<S2>,
    ) {
        self.statements.push(SeqCStatement::Constant {
            name: name.into(),
            value,
            comment: comment.map(|s| s.into()),
        });
    }

    pub fn add_repeat(&mut self, num_repeats: u64, body: SeqCGenerator) {
        let complexity = body.estimate_complexity() + 2; // penalty for loop overhead
        self.statements.push(SeqCStatement::Repeat {
            num_repeats,
            body,
            complexity,
        });
    }

    pub fn add_do_while<S: Into<String>>(&mut self, condition: S, body: SeqCGenerator) {
        let complexity = body.estimate_complexity() + 5; // penalty for loop overhead
        self.statements.push(SeqCStatement::DoWhile {
            condition: condition.into(),
            body,
            complexity,
        });
    }

    pub fn add_if<S: Into<String>>(
        &mut self,
        conditions: Vec<S>,
        mut bodies: Vec<SeqCGenerator>,
    ) -> Result<()> {
        if conditions.len() != bodies.len() && bodies.len() != conditions.len() + 1 {
            return Err(anyhow!(
                "Number of conditions {} and bodies {} do not match",
                conditions.len(),
                bodies.len()
            )
            .into());
        }
        let conditions = conditions
            .into_iter()
            .map(|s| s.into())
            .collect::<Vec<String>>();
        if conditions[..conditions.len() - 1]
            .iter()
            .any(|c| c.is_empty())
        {
            return Err(anyhow!("Condition may not be None").into());
        }

        let mut complexity = 0;
        for body in &bodies {
            complexity += body.estimate_complexity() + 1;
        }
        let has_else = bodies.len() > conditions.len();
        let else_body = if has_else { bodies.pop() } else { None };
        self.statements.push(SeqCStatement::DoIf {
            conditions,
            bodies,
            else_body,
            complexity,
        });
        Ok(())
    }

    pub fn add_function_def<S: Into<String>>(&mut self, text: S) {
        self.statements
            .push(SeqCStatement::FunctionDef { text: text.into() });
    }

    pub fn is_variable_declared(&self, variable_name: &Variable) -> bool {
        self.symbols.contains(variable_name)
    }

    // warning: this function is designed to only work if the seqc generator maps to a single scope
    //          (it asserts that each variable may only be defined once)
    pub fn add_variable_declaration<S: Into<String>>(
        &mut self,
        variable_name: S,
        initial_value: Option<SeqCVariant>,
    ) -> Result<()> {
        let variable_name: String = variable_name.into();
        if self.symbols.contains(&variable_name) {
            return Err(anyhow!(
                "Trying to declare variable {variable_name} which has already been declared in this scope"
            )
            .into());
        }
        self.symbols.insert(variable_name.clone());
        self.statements.push(SeqCStatement::VariableDeclaration {
            variable_name,
            initial_value,
        });
        Ok(())
    }

    pub fn add_variable_assignment<S: Into<String>>(
        &mut self,
        variable_name: S,
        value: SeqCVariant,
    ) {
        self.statements.push(SeqCStatement::VariableAssignment {
            variable_name: variable_name.into(),
            value,
        });
    }

    pub fn add_variable_increment<S: Into<String>>(
        &mut self,
        variable_name: S,
        value: SeqCVariant,
    ) {
        self.statements.push(SeqCStatement::VariableIncrement {
            variable_name: variable_name.into(),
            value,
        });
    }

    pub fn add_assign_wave_index_statement<S: Into<String>>(
        &mut self,
        wave_id: S,
        wave_index: WaveIndex,
        channel: Option<ChannelIndex>,
    ) {
        self.statements.push(SeqCStatement::AssignWaveIndex {
            wave_id: wave_id.into(),
            wave_index,
            channel,
        });
    }

    pub fn add_play_wave_statement<S: Into<String>>(
        &mut self,
        wave_id: S,
        channel: Option<ChannelIndex>,
    ) {
        self.statements.push(SeqCStatement::PlayWave {
            wave_id: wave_id.into(),
            channel,
        });
    }

    pub fn add_command_table_execution<S: Into<String>>(
        &mut self,
        table_index: SeqCVariant,
        latency: Option<SeqCVariant>,
        comment: Option<S>,
    ) {
        self.statements.push(SeqCStatement::CommandTableExecution {
            table_index,
            latency,
            comment: comment.map(|s| s.into()),
        });
    }

    pub fn add_play_zero_or_hold(
        &mut self,
        mut num_samples: Samples,
        hold: bool,
        deferred_calls: &mut Option<&mut SeqCGenerator>,
    ) -> Result<()> {
        let fname = if hold { "playHold" } else { "playZero" };
        let max_play_zero_hold = self.device_traits.max_play_zero_hold;
        if num_samples % self.device_traits.sample_multiple as Samples != 0 {
            return Err(anyhow!(
                "Emitting {function_name}({wave_length}), which is not divisible by \
                    {sample_multiple}, which it should be for {device_name}",
                function_name = fname,
                wave_length = num_samples,
                sample_multiple = self.device_traits.sample_multiple,
                device_name = self.device_traits.type_str,
            )
            .into());
        }
        if num_samples < self.device_traits.min_play_wave.into() {
            return Err(anyhow!(
                "Attempting to emit {}({}), which is below the minimum \
                waveform length {} of device '{}' (sample multiple is {})",
                fname,
                num_samples,
                self.device_traits.min_play_wave,
                self.device_traits.type_str,
                self.device_traits.sample_multiple,
            )
            .into());
        }
        let add_statement = |_self: &mut SeqCGenerator, n: Samples| {
            _self.statements.push(SeqCStatement::PlayZeroOrHold {
                num_samples: n,
                hold,
            });
        };
        let flush_deferred_calls =
            |_self: &mut SeqCGenerator, deferred_calls: &mut Option<&mut SeqCGenerator>| {
                if let Some(def_calls) = deferred_calls {
                    _self.append_statements_from(def_calls);
                    def_calls.clear();
                }
            };
        if num_samples <= max_play_zero_hold {
            add_statement(self, num_samples);
            flush_deferred_calls(self, deferred_calls);
        } else if num_samples <= 2 * max_play_zero_hold {
            // split in the middle
            let half_samples = (num_samples / 2 / 16) * 16;
            add_statement(self, half_samples);
            flush_deferred_calls(self, deferred_calls);
            add_statement(self, num_samples - half_samples);
        } else {
            // Non-unrolled loop.
            // There's some subtlety here: If there are any pending deferred calls,
            // we must place at least one playZero() _before_ the loop, and the pending
            // calls after that.
            // In addition, on UHFQA, there must be a non-looped playZero after the loop.
            // Otherwise, a following startQA() might be delayed by the sequencer exiting
            // the loop (HBAR-2075).

            let (mut num_segments, mut head) = (
                (num_samples / max_play_zero_hold) as u64,
                num_samples % max_play_zero_hold,
            );
            if num_segments < 2 {
                return Err(anyhow!("Shorter loops should be unrolled").into());
            }
            let mut tail = 0;
            if head > 0 && head < MIN_PLAY_ZERO_HOLD {
                let chunk = (max_play_zero_hold / 2 / 16) * 16;
                add_statement(self, chunk);
                flush_deferred_calls(self, deferred_calls);
                num_samples -= chunk;
                (num_segments, head) = (
                    (num_samples / max_play_zero_hold) as u64,
                    num_samples % max_play_zero_hold,
                );
            }
            if self.device_traits.require_play_zero_after_loop && num_segments > 1 {
                // UHFQA
                num_segments -= 1;
                tail += max_play_zero_hold;
            }
            if head > 0 {
                add_statement(self, head);
                flush_deferred_calls(self, deferred_calls);
            } else if deferred_calls.is_some()
                && deferred_calls.as_ref().unwrap().num_statements() > 0
            {
                add_statement(self, max_play_zero_hold);
                flush_deferred_calls(self, deferred_calls);
                num_segments -= 1;
            }

            match num_segments {
                1 => {
                    add_statement(self, max_play_zero_hold);
                }
                n if n > 1 => {
                    let mut loop_body = self.create();
                    loop_body.add_statement(SeqCStatement::PlayZeroOrHold {
                        num_samples: max_play_zero_hold,
                        hold,
                    });
                    self.add_repeat(num_segments, loop_body);
                }
                _ => {}
            }
            if tail > 0 {
                add_statement(self, tail);
            }
        }
        Ok(())
    }

    /// Add a playZero command
    ///
    /// If the requested number of samples exceeds the allowed number of samples for
    /// a single playZero, a tight loop of playZeros will be emitted.
    ///
    /// If deferred_calls is passed, the deferred function calls are cleared in the
    /// context of the added playZero(s). The passed list will be drained.
    ///
    pub fn add_play_zero_statement(
        &mut self,
        num_samples: Samples,
        deferred_calls: &mut Option<&mut SeqCGenerator>,
    ) -> Result<()> {
        self.add_play_zero_or_hold(num_samples, false, deferred_calls)
    }

    /// Add a playHold command
    ///
    /// If the requested number of samples exceeds the allowed number of samples for
    /// a single playHold, a tight loop of playHolds will be emitted.
    ///
    /// If deferred_calls is passed, the deferred function calls are cleared in the
    /// context of the added playHold(s). The passed list will be drained.
    ///
    pub fn add_play_hold_statement(
        &mut self,
        num_samples: Samples,
        deferred_calls: &mut Option<&mut SeqCGenerator>,
    ) -> Result<()> {
        self.add_play_zero_or_hold(num_samples, true, deferred_calls)
    }

    /// Calculate a rough estimate for the complexity (~nr of instructions)
    ///
    /// The point here is not to be accurate about every statement, but to correctly
    /// gauge the size of loops etc.
    pub fn estimate_complexity(&self) -> u64 {
        self.statements.iter().map(SeqCStatement::complexity).sum()
    }

    pub fn generate_seq_c(&self) -> String {
        self.statements
            .iter()
            .map(|statement| self.emit_statement(statement))
            .collect::<String>()
    }

    fn emit_statement(&self, statement: &SeqCStatement) -> String {
        match statement {
            SeqCStatement::Comment { text } => format!("/* {text} */\n"),
            SeqCStatement::FunctionCall {
                name,
                args,
                assign_to,
            } => {
                let assign_to = assign_to
                    .as_ref()
                    .map(|s| format!("{s} = "))
                    .unwrap_or_default();
                let args = args
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>()
                    .join(",");

                format!("{assign_to}{name}({args});\n")
            }
            SeqCStatement::WaveDeclaration {
                wave_id,
                length,
                has_marker1,
                has_marker2,
            } => match self.device_traits.supports_binary_waves {
                false => String::new(), // SHFQA
                true => gen_wave_declaration_placeholder(
                    self.dual_channel,
                    wave_id,
                    *length,
                    *has_marker1,
                    *has_marker2,
                ),
            },
            SeqCStatement::ZeroWaveDeclaration { wave_id, length } => {
                gen_zero_wave_declaration_placeholder(self.dual_channel, wave_id, *length)
            }
            SeqCStatement::FunctionDef { text } => text.clone(),
            SeqCStatement::VariableDeclaration {
                variable_name,
                initial_value,
            } => {
                let initial_value = initial_value
                    .as_ref()
                    .map(|s| format!(" = {s}"))
                    .unwrap_or_default();
                format!("var {variable_name}{initial_value};\n")
            }
            SeqCStatement::VariableAssignment {
                variable_name,
                value,
            } => format!("{variable_name} = {value};\n"),
            SeqCStatement::VariableIncrement {
                variable_name,
                value,
            } => format!("{variable_name} += {value};\n"),
            SeqCStatement::AssignWaveIndex {
                wave_id,
                wave_index,
                channel,
            } => {
                let wave_channels = build_wave_channel_assignment(
                    self.dual_channel,
                    wave_id,
                    *channel,
                    self.device_traits.supports_digital_iq_modulation,
                );
                format!("assignWaveIndex({wave_channels},{wave_index});\n")
            }
            SeqCStatement::PlayWave { wave_id, channel } => {
                let wave_channels = build_wave_channel_assignment(
                    self.dual_channel,
                    wave_id,
                    *channel,
                    self.device_traits.supports_digital_iq_modulation,
                );
                format!("playWave({wave_channels});\n")
            }
            SeqCStatement::CommandTableExecution {
                table_index,
                latency,
                comment,
            } => {
                let latency = latency
                    .as_ref()
                    .map(|s| format!(", {s}"))
                    .unwrap_or_default();
                format!(
                    "executeTableEntry({}{});{}\n",
                    table_index,
                    latency,
                    format_comment(comment)
                )
            }
            SeqCStatement::PlayZeroOrHold { num_samples, hold } => {
                let fname = if *hold { "playHold" } else { "playZero" };
                format!("{fname}({num_samples});\n")
            }
            SeqCStatement::Repeat {
                num_repeats, body, ..
            } => {
                let body = indent(&body.generate_seq_c(), "  ");
                format!("repeat ({num_repeats}) {{\n{body}}}\n")
            }
            SeqCStatement::DoWhile {
                condition, body, ..
            } => {
                let body = indent(&body.generate_seq_c(), "  ");
                format!("do {{\n{body}}}\nwhile({condition});\n")
            }
            SeqCStatement::DoIf {
                conditions,
                bodies,
                else_body,
                ..
            } => {
                let mut text = String::new();
                for (i, condition) in conditions.iter().enumerate() {
                    let body = indent(&bodies[i].generate_seq_c(), "  ");
                    if i == 0 {
                        text += &format!("if ({condition}) {{\n{body}}}\n");
                    } else {
                        text += &format!("else if ({condition}) {{\n{body}}}\n");
                    }
                }
                if let Some(else_body) = else_body {
                    let body = indent(&else_body.generate_seq_c(), "  ");
                    text += &format!("else {{\n{body}}}\n");
                }
                text
            }
            SeqCStatement::Constant {
                name,
                value,
                comment,
            } => {
                format!("const {} = {};{}\n", name, value, format_comment(comment))
            }
        }
    }
}

pub fn seqc_generator_from_device_and_signal_type<S: AsRef<str>>(
    device: S,
    signal_type: S,
) -> Result<SeqCGenerator> {
    let device_traits = match device.as_ref().to_uppercase().as_str() {
        "SHFQA" => DeviceKind::SHFQA.traits(),
        "SHFSG" => DeviceKind::SHFSG.traits(),
        "HDAWG" => DeviceKind::HDAWG.traits(),
        "UHFQA" => DeviceKind::UHFQA.traits(),
        _ => {
            return Err(anyhow!(
                "Unsupported device type: {}. Supported types are: SHFQA, SHFSG, HDAWG, UHFQA",
                device.as_ref()
            )
            .into());
        }
    };
    let signal_type = signal_type.as_ref().to_lowercase();
    let dual_channel = signal_type == "iq" || signal_type == "double";
    Ok(SeqCGenerator::new(device_traits, dual_channel))
}

pub fn seqc_generator_from_device_traits(
    device_traits: &'static DeviceTraits,
    signal_type: &AwgKind,
) -> SeqCGenerator {
    let dual_channel = !matches!(signal_type, AwgKind::SINGLE);
    SeqCGenerator::new(device_traits, dual_channel)
}
