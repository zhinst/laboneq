// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{SectionUid, SignalUid};

use crate::ir::{PrngSetup, Section, Trigger};

pub(crate) struct SectionBuilder {
    inner: Section,
}

impl SectionBuilder {
    pub(crate) fn new(uid: SectionUid) -> Self {
        Self {
            inner: Section {
                uid,
                triggers: Vec::new(),
                prng_setup: None,
            },
        }
    }

    pub(crate) fn add_trigger(mut self, signal: SignalUid, state: u8) -> Self {
        self.inner.triggers.push(Trigger { signal, state });
        self
    }

    pub(crate) fn prng_setup(mut self, range: u32, seed: u32) -> Self {
        assert!(
            self.inner.prng_setup.is_none(),
            "PRNG setup already defined for this section"
        );
        self.inner.prng_setup = Some(PrngSetup { range, seed });
        self
    }

    pub(crate) fn build(self) -> Section {
        self.inner
    }
}
