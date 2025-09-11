// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::{Error, Result};
use codegenerator::ir::compilation_job::AwgKind;
use indexmap::IndexMap;

pub type WaveIndex = u32;

pub enum SignalType {
    CSV,
    COMPLEX,
    SIGNAL(AwgKind),
}
pub struct WaveIndexTracker {
    pub wave_indices: IndexMap<String, (Option<WaveIndex>, SignalType)>,
    next_wave_index: WaveIndex,
}

impl Default for WaveIndexTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl WaveIndexTracker {
    pub fn new() -> Self {
        Self {
            wave_indices: IndexMap::new(),
            next_wave_index: 0,
        }
    }
    pub fn lookup_index_by_wave_id(&self, wave_id: &str) -> Option<Option<WaveIndex>> {
        let entry = self.wave_indices.get(wave_id);
        entry.map(|entry| entry.0)
    }

    pub fn create_index_for_wave<S1: Into<String>>(
        &mut self,
        wave_id: S1,
        signal_type: SignalType,
    ) -> Result<Option<WaveIndex>> {
        let wave_id: String = wave_id.into();
        if self.wave_indices.contains_key(&wave_id) {
            return Err(Error::new("Wave ID already exists"));
        }
        let index = if matches!(signal_type, SignalType::CSV) {
            // For CSV store only the signature, do not allocate an index
            None
        } else {
            let index = self.next_wave_index;
            self.next_wave_index += 1;
            Some(index)
        };
        self.wave_indices.insert(wave_id, (index, signal_type));
        Ok(index)
    }

    pub fn add_numbered_wave<S1: Into<String>>(
        &mut self,
        wave_id: S1,
        signal_type: SignalType,
        index: WaveIndex,
    ) {
        self.wave_indices
            .insert(wave_id.into(), (Some(index), signal_type));
    }

    pub fn finish(self) -> IndexMap<String, (Option<WaveIndex>, SignalType)> {
        self.wave_indices
    }
}
