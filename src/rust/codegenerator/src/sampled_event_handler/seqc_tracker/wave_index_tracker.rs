// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::AwgKind;
use crate::sampled_event_handler::awg_events::StaticWaveformSignature;
use crate::{Error, Result};
use indexmap::IndexMap;

pub type WaveIndex = u32;

pub enum SignalType {
    COMPLEX,
    SIGNAL(AwgKind),
}
pub(crate) struct WaveIndexTracker {
    pub wave_indices: IndexMap<StaticWaveformSignature, (WaveIndex, SignalType)>,
    next_wave_index: WaveIndex,
}

impl Default for WaveIndexTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl WaveIndexTracker {
    pub(crate) fn new() -> Self {
        Self {
            wave_indices: IndexMap::new(),
            next_wave_index: 0,
        }
    }

    pub(crate) fn lookup_index_by_wave_id(
        &self,
        wave_id: &StaticWaveformSignature,
    ) -> Option<WaveIndex> {
        let entry = self.wave_indices.get(wave_id);
        entry.map(|entry| entry.0)
    }

    pub(crate) fn create_index_for_wave(
        &mut self,
        wave_id: &StaticWaveformSignature,
        signal_type: SignalType,
    ) -> Result<WaveIndex> {
        if self.wave_indices.contains_key(wave_id) {
            return Err(Error::new("Wave ID already exists"));
        }
        let index = self.next_wave_index;
        self.next_wave_index += 1;
        self.wave_indices
            .insert(wave_id.clone(), (index, signal_type));
        Ok(index)
    }

    pub(crate) fn add_numbered_wave(
        &mut self,
        wave_id: &StaticWaveformSignature,
        signal_type: SignalType,
        index: WaveIndex,
    ) {
        if !self.wave_indices.contains_key(wave_id) {
            self.wave_indices
                .insert(wave_id.clone(), (index, signal_type));
        }
    }

    pub(crate) fn finish(self) -> IndexMap<String, (WaveIndex, SignalType)> {
        self.wave_indices
            .into_iter()
            .map(|(key, (index, signal_type))| {
                (key.signature_string().to_string(), (index, signal_type))
            })
            .collect()
    }
}
