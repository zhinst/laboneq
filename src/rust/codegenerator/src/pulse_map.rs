// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use indexmap::IndexMap;
use laboneq_common::uid::{PulseParameterUid, PulseUid};
use laboneq_dsl::{operation::ExternalOrValue, types::ComplexOrFloat};

use crate::ir::compilation_job::MixerType;
use crate::result::WaveformSignatureString;

pub struct PulseWaveform {
    pub sampling_rate: f64,
    pub length_samples: usize,
    pub iq_modulation: bool,
    // UHFQA's HW modulation is not an IQ mixer. None for flux pulses etc.
    pub mixer_type: Option<MixerType>,
    pub instances: Vec<PulseInstance>,
    pub compressed: bool,
}

pub struct PulseInstance {
    pub offset_samples: usize,
    pub amplitude: Option<ComplexOrFloat>,
    pub length: Option<f64>,
    pub iq_phase: Option<f64>,
    pub modulation_frequency: Option<f64>,
    pub channel: Option<usize>,
    pub needs_conjugate: bool,
    pub parameters: HashMap<PulseParameterUid, ExternalOrValue>,
    pub pulse_parameters: HashMap<PulseParameterUid, ExternalOrValue>,
    pub has_marker1: bool,
    pub has_marker2: bool,
    pub can_compress: bool,
}

#[derive(Default)]
pub struct PulseMap(IndexMap<PulseUid, IndexMap<WaveformSignatureString, PulseWaveform>>);

impl IntoIterator for PulseMap {
    type Item = (PulseUid, IndexMap<WaveformSignatureString, PulseWaveform>);
    type IntoIter =
        indexmap::map::IntoIter<PulseUid, IndexMap<WaveformSignatureString, PulseWaveform>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl PulseMap {
    pub fn with_capacity(capacity: usize) -> Self {
        PulseMap(IndexMap::with_capacity(capacity))
    }

    pub fn insert(
        &mut self,
        pulse_uid: PulseUid,
        signature: WaveformSignatureString,
        waveform: PulseWaveform,
    ) {
        self.0
            .entry(pulse_uid)
            .or_default()
            .insert(signature, waveform);
    }

    pub fn merge(stores: impl IntoIterator<Item = PulseMap>) -> Self {
        let mut merged = PulseMap::default();
        for store in stores {
            for (key, buffer) in store.0 {
                for (signature, waveform) in buffer {
                    merged.insert(key, signature, waveform);
                }
            }
        }
        merged
    }

    pub fn into_map(self) -> IndexMap<PulseUid, IndexMap<WaveformSignatureString, PulseWaveform>> {
        self.0
    }
}
