// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::ir::compilation_job::ChannelIndex;

use crate::Samples;
use crate::seqc_generator::SeqCGenerator;
use crate::wave_index_tracker::WaveIndex;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};

type WaveIdInternal = String;
type ConditionInternal = String;
type VariableInternal = String;

// an enum to represent strings, integers, and floats
#[derive(Debug, Clone, PartialEq)]
pub enum SeqCVariant {
    Bool(bool),
    String(String),
    LiteralString(String),
    Integer(i64),
    Float(f64),
    None(),
}

impl Hash for SeqCVariant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            SeqCVariant::Bool(b) => b.hash(state),
            SeqCVariant::String(s) => s.hash(state),
            SeqCVariant::LiteralString(s) => s.hash(state),
            SeqCVariant::Integer(i) => i.hash(state),
            SeqCVariant::Float(f) => {
                let bits = f.to_bits();
                bits.hash(state);
            }
            SeqCVariant::None() => {}
        }
    }
}

impl fmt::Display for SeqCVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeqCVariant::Bool(b) => write!(f, "{b}"),
            SeqCVariant::String(s) => write!(f, "{s}"),
            SeqCVariant::LiteralString(s) => write!(f, "\"{s}\""),
            SeqCVariant::Integer(i) => write!(f, "{i}"),
            SeqCVariant::Float(flt) => write!(f, "{flt:.1}"),
            SeqCVariant::None() => write!(f, ""),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq)]
pub enum SeqCStatement {
    Comment {
        text: String,
    },
    FunctionCall {
        name: String,
        args: Vec<SeqCVariant>,
        assign_to: Option<String>,
    },
    WaveDeclaration {
        wave_id: WaveIdInternal,
        length: Samples,
        has_marker1: bool,
        has_marker2: bool,
    },
    ZeroWaveDeclaration {
        wave_id: WaveIdInternal,
        length: Samples,
    },
    Constant {
        name: String,
        value: SeqCVariant,
        comment: Option<String>,
    },
    Repeat {
        num_repeats: u64,
        body: SeqCGenerator,
        complexity: u64,
    },
    DoWhile {
        condition: ConditionInternal,
        body: SeqCGenerator,
        complexity: u64,
    },
    DoIf {
        conditions: Vec<ConditionInternal>,
        bodies: Vec<SeqCGenerator>,
        else_body: Option<SeqCGenerator>,
        complexity: u64,
    },
    FunctionDef {
        text: String,
    },
    VariableDeclaration {
        variable_name: VariableInternal,
        initial_value: Option<SeqCVariant>,
    },
    VariableAssignment {
        variable_name: VariableInternal,
        value: SeqCVariant,
    },
    VariableIncrement {
        variable_name: VariableInternal,
        value: SeqCVariant,
    },
    AssignWaveIndex {
        wave_id: WaveIdInternal,
        wave_index: WaveIndex,
        channel: Option<ChannelIndex>,
    },
    PlayWave {
        wave_id: WaveIdInternal,
        channel: Option<ChannelIndex>,
    },
    CommandTableExecution {
        table_index: SeqCVariant,
        latency: Option<SeqCVariant>,
        comment: Option<String>,
    },
    PlayZeroOrHold {
        num_samples: Samples,
        hold: bool,
    },
}

impl SeqCStatement {
    pub fn complexity(&self) -> u64 {
        match self {
            SeqCStatement::DoWhile { complexity, .. } => *complexity,
            SeqCStatement::DoIf { complexity, .. } => *complexity,
            SeqCStatement::Repeat { complexity, .. } => *complexity,
            SeqCStatement::Comment { .. } => 0,
            _ => 1,
        }
    }

    pub fn to_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}
