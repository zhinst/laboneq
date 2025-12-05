// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::ChannelIndex;

use super::seqc_generator::SeqCGenerator;
use super::wave_index_tracker::WaveIndex;
use crate::ir::Samples;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};

type WaveIdInternal = String;
type ConditionInternal = String;
type VariableInternal = String;

// an enum to represent strings, integers, and floats
#[derive(Debug, Clone, PartialEq, derive_more::From)]
pub(crate) enum SeqCVariant {
    String(String),
    Integer(i64),
    Float(f64),
}

impl Hash for SeqCVariant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            SeqCVariant::String(s) => s.hash(state),
            SeqCVariant::Integer(i) => i.hash(state),
            SeqCVariant::Float(f) => {
                let bits = f.to_bits();
                bits.hash(state);
            }
        }
    }
}

impl fmt::Display for SeqCVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeqCVariant::String(s) => write!(f, "{s}"),
            SeqCVariant::Integer(i) => write!(f, "{i}"),
            SeqCVariant::Float(flt) => write!(f, "{flt:.1}"),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq)]
pub(crate) enum SeqCStatement {
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
    pub(crate) fn complexity(&self) -> u64 {
        match self {
            SeqCStatement::DoIf { complexity, .. } => *complexity,
            SeqCStatement::Repeat { complexity, .. } => *complexity,
            SeqCStatement::Comment { .. } => 0,
            _ => 1,
        }
    }

    pub(crate) fn to_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}
