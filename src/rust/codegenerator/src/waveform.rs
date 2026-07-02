// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use indexmap::IndexMap;
use laboneq_error::laboneq_error;
use num_complex::Complex64;

use crate::Result;

pub type WaveformSignatureString = Arc<String>;

#[derive(Debug, Clone, PartialEq, Hash, Eq, Copy, PartialOrd, Ord)]
pub enum WaveIdentifier {
    I,
    Q,
    M1,
    M2,
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct WaveKey {
    pub signature: WaveformSignatureString,
    pub identifier: Option<WaveIdentifier>,
}

impl WaveKey {
    pub fn filename(&self) -> String {
        const WAVE_SUFFIX: &str = ".wave";
        let identifer_string = match self.identifier {
            Some(WaveIdentifier::I) => "_i",
            Some(WaveIdentifier::Q) => "_q",
            Some(WaveIdentifier::M1) => "_marker1",
            Some(WaveIdentifier::M2) => "_marker2",
            None => "",
        };
        format!("{}{}{}", self.signature, identifer_string, WAVE_SUFFIX)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SampleBuffer {
    Float64(Vec<f64>),
    Complex64(Vec<Complex64>),
    U8(Vec<u8>),
}

/// Hash `SampleBuffer` contents using bitwise f64 identity — consistent with sample equality:
/// two buffers are equal if they produce the same hash.
pub(crate) fn hash_sample_buffer<H: std::hash::Hasher>(buf: &SampleBuffer, state: &mut H) {
    match buf {
        SampleBuffer::Float64(v) => {
            state.write_u8(0);
            state.write_usize(v.len());
            hash_f64_slice(state, v);
        }
        SampleBuffer::Complex64(v) => {
            state.write_u8(1);
            state.write_usize(v.len());
            // re and im are laid out sequentially — treat as flat f64 pairs
            let mut buf = [0u8; 128]; // 8 × Complex64 = 8 × 16 bytes
            for chunk in v.chunks(8) {
                let n = chunk.len() * 16;
                for (i, c) in chunk.iter().enumerate() {
                    let base = i * 16;
                    buf[base..base + 8].copy_from_slice(&c.re.to_bits().to_ne_bytes());
                    buf[base + 8..base + 16].copy_from_slice(&c.im.to_bits().to_ne_bytes());
                }
                state.write(&buf[..n]);
            }
        }
        SampleBuffer::U8(v) => {
            state.write_u8(2);
            state.write(v);
        }
    }
}

fn hash_f64_slice<H: std::hash::Hasher>(state: &mut H, v: &[f64]) {
    let mut buf = [0u8; 64]; // 8 × f64 = 8 × 8 bytes, one cache line
    for chunk in v.chunks(8) {
        let n = chunk.len() * 8;
        for (i, f) in chunk.iter().enumerate() {
            buf[i * 8..(i + 1) * 8].copy_from_slice(&f.to_bits().to_ne_bytes());
        }
        state.write(&buf[..n]);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Waveform {
    pub key: WaveKey,
    pub samples: SampleBuffer,
}

#[derive(Debug, Clone, Default)]
pub struct WaveformStore(IndexMap<WaveKey, SampleBuffer>);

impl WaveformStore {
    pub fn insert(&mut self, key: WaveKey, buffer: SampleBuffer) -> Result<()> {
        if let Some(existing) = self.0.get(&key) {
            if existing != &buffer {
                return Err(laboneq_error!(
                    "Internal error: Inconsistent waveforms with the same key '{}', {} != {}",
                    key.signature,
                    format!("{:?}", existing),
                    format!("{:?}", buffer)
                ));
            }
        } else {
            self.0.insert(key, buffer);
        }
        Ok(())
    }

    pub fn get(&self, key: &WaveKey) -> Option<&SampleBuffer> {
        self.0.get(key)
    }

    pub fn merge(stores: impl IntoIterator<Item = WaveformStore>) -> Result<Self> {
        let mut merged = WaveformStore::default();
        for store in stores {
            for (key, buffer) in store.0 {
                merged.insert(key, buffer)?;
            }
        }
        Ok(merged)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&WaveKey, &SampleBuffer)> {
        self.0.iter()
    }
}

impl IntoIterator for WaveformStore {
    type Item = (WaveKey, SampleBuffer);
    type IntoIter = indexmap::map::IntoIter<WaveKey, SampleBuffer>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompressionProperties {
    pub hold_start: usize,
    pub hold_length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodegenWaveform {
    pub key: WaveKey,
    pub compression_properties: Option<CompressionProperties>,
    pub downsampling_factor: Option<u8>,
}

impl CodegenWaveform {
    pub(crate) fn new(
        key: WaveKey,
        compression_properties: Option<CompressionProperties>,
        downsampling_factor: Option<u8>,
    ) -> Self {
        Self {
            key,
            compression_properties,
            downsampling_factor,
        }
    }

    pub fn wave_key(&self) -> &WaveKey {
        &self.key
    }

    pub fn filename(&self) -> String {
        self.key.filename()
    }
}
