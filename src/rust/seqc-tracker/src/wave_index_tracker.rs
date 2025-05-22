// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use indexmap::IndexMap;

pub struct WaveIndexTracker {
    pub wave_indices: IndexMap<String, (i32, String)>,
    next_wave_index: i32,
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
    pub fn lookup_index_by_wave_id(&self, wave_id: &str) -> Option<i32> {
        let entry = self.wave_indices.get(wave_id);
        entry.map(|entry| entry.0)
    }

    pub fn create_index_for_wave<S1: Into<String>, S2: Into<String>>(
        &mut self,
        wave_id: S1,
        signal_type: S2,
    ) -> Result<Option<i32>, String> {
        let wave_id: String = wave_id.into();
        let signal_type: String = signal_type.into();
        if self.wave_indices.contains_key(&wave_id) {
            return Err("Wave ID already exists".to_string());
        }
        if signal_type == "csv" {
            // For CSV store only the signature, do not allocate an index
            self.wave_indices.insert(wave_id, (-1, signal_type));
            return Ok(None);
        }
        let index = self.next_wave_index;
        self.next_wave_index += 1;
        self.wave_indices.insert(wave_id, (index, signal_type));
        Ok(Some(index))
    }

    pub fn add_numbered_wave<S1: Into<String>, S2: Into<String>>(
        &mut self,
        wave_id: S1,
        signal_type: S2,
        index: i32,
    ) {
        self.wave_indices
            .insert(wave_id.into(), (index, signal_type.into()));
    }
}
