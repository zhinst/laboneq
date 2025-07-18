// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::{seqc_generator::SeqCGenerator, seqc_statements::SeqCVariant};

#[derive(Default)]
pub struct PRNGTracker {
    range: Option<u32>,
    seed: Option<u32>,
    offset: u32,
    committed: bool,
}

impl PRNGTracker {
    pub fn new() -> Self {
        PRNGTracker {
            range: None,
            seed: None,
            offset: 0,
            committed: false,
        }
    }

    pub fn set_range(&mut self, value: u32) {
        assert!(!self.committed);
        self.range = Some(value);
    }

    pub fn set_seed(&mut self, value: u32) {
        assert!(!self.committed);
        self.seed = Some(value);
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn set_offset(&mut self, value: u32) {
        assert!(!self.committed);
        self.offset = value;
    }

    pub fn is_committed(&self) -> bool {
        self.committed
    }

    pub fn commit(&mut self, seqc_gen: &mut SeqCGenerator) {
        assert!(!self.committed);
        if let Some(seed) = self.seed {
            seqc_gen.add_function_call_statement(
                "setPRNGSeed",
                vec![SeqCVariant::Integer(seed as i64)],
                None::<&str>,
            );
        }
        if let Some(range) = self.range {
            seqc_gen.add_function_call_statement(
                "setPRNGRange",
                vec![
                    SeqCVariant::Integer(self.offset as i64),
                    SeqCVariant::Integer((self.offset + range - 1) as i64),
                ],
                None::<&str>,
            );
        }
        // the tracker now has been spent, so clear it to prevent another call to `commit()`
        self.committed = true;
    }
}
