// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module provides the `WaveformSignature` and `PulseSignature` types,
//! which are used to represent the signature of a waveform and its pulses.
//!
//! The signatures are used to uniquely identify waveforms and their pulses,
//! and to generate a unique waveform declaration string for them.
//!
//! The uniqueness of signatures is based on the properties of the waveform and its pulses,
//! such as the start time, length, amplitude, phase, and other properties.
//!
//! The underling promise for uniqueness is that the hash of the signatures will result
//! in same waveforms across a single AWG. Therefore using the signature hashes across different AWGs
//! is not recommended as the signatures may not match (this is undefined behavior).
use crate::Result;
use crate::ir::Samples;
use crate::ir::compilation_job::{Marker, PulseDef};
use crate::ir::experiment::PulseParametersId;
use crate::utils::{normalize_f64, normalize_phase};
use num_complex::Complex64;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use sha1::{Digest, Sha1};
use std::f64::consts::PI;
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Signature of a single pulse, part of a sampled waveform
#[derive(Debug, Clone)]
pub struct PulseSignature {
    /// The samples offset of the pulse in the waveform
    pub start: i64,
    /// The length of the pulse in samples
    pub length: i64,
    /// Pulse definition of this signature
    pub pulse: Option<Arc<PulseDef>>,
    /// The amplitude of the pulse in samples
    pub amplitude: Option<f64>,
    /// The phase of the pulse (for software oscillators)
    pub phase: f64,
    /// The oscillator frequency of the pulse (for software oscillators)
    pub oscillator_frequency: Option<f64>,
    /// The channel of the pulse (for HDAWG)
    pub channel: Option<u16>,
    /// The sub-channel of the pulse (for SHFQA)
    pub sub_channel: Option<u8>,
    /// Pulse parameters ID
    /// Pulse parameters are parameters that are associated
    /// with the given `pulse`.
    pub id_pulse_params: Option<PulseParametersId>,
    /// Markers played during this pulse
    pub markers: Vec<Marker>,
    /// Oscillator phase
    pub oscillator_phase: Option<f64>,
    /// If present, the pulse increments the HW oscillator phase.
    pub increment_oscillator_phase: Option<f64>,
    /// The preferred amplitude register for this pulse
    pub preferred_amplitude_register: Option<u16>,
    /// If this pulse increments the oscillator phase, this field indicates the name of
    /// the sweep parameters that determine the increment (if applicable)
    pub incr_phase_params: Vec<String>,
}

impl PulseSignature {
    pub fn end(&self) -> i64 {
        self.start + self.length
    }
}

impl PartialEq for PulseSignature {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start
            && match (&self.pulse, &other.pulse) {
                // NOTE: The pulse UID is currently always unique
                (Some(self_pulse), Some(other_pulse)) => self_pulse.uid == other_pulse.uid,
                (None, None) => true,
                _ => false,
            }
            && self.length == other.length
            && self.amplitude == other.amplitude
            && self.phase == other.phase
            && self.oscillator_frequency == other.oscillator_frequency
            && self.channel == other.channel
            && self.sub_channel == other.sub_channel
            && self.id_pulse_params == other.id_pulse_params
            && self.markers == other.markers
            && self.oscillator_phase == other.oscillator_phase
            && self.increment_oscillator_phase == other.increment_oscillator_phase
            && self.preferred_amplitude_register == other.preferred_amplitude_register
            && self.incr_phase_params == other.incr_phase_params
    }
}

impl Eq for PulseSignature {}

impl Hash for PulseSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.length.hash(state);
        self.pulse.as_ref().map(|p| &p.uid).hash(state);
        self.amplitude.map(normalize_f64).hash(state);
        normalize_f64(self.phase).hash(state);
        self.oscillator_frequency.map(normalize_f64).hash(state);
        self.channel.hash(state);
        self.sub_channel.hash(state);
        self.id_pulse_params.hash(state);
        self.markers.hash(state);
        self.oscillator_phase.map(normalize_f64).hash(state);
        self.increment_oscillator_phase
            .map(normalize_f64)
            .hash(state);
        self.preferred_amplitude_register.hash(state);
        self.incr_phase_params.hash(state);
    }
}

impl Serialize for PulseSignature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PulseSignature", 14)?;
        state.serialize_field("start", &self.start)?;
        state.serialize_field("length", &self.length)?;
        // Only serialize the UID of the pulse, not the full PulseDef
        state.serialize_field("pulse_uid", &self.pulse.as_ref().map(|p| &p.uid))?;
        state.serialize_field("amplitude", &self.amplitude.map(normalize_f64))?;
        state.serialize_field("phase", &normalize_f64(self.phase))?;
        state.serialize_field(
            "oscillator_frequency",
            &self.oscillator_frequency.map(normalize_f64),
        )?;
        state.serialize_field("channel", &self.channel)?;
        state.serialize_field("sub_channel", &self.sub_channel)?;
        state.serialize_field("id_pulse_params", &self.id_pulse_params.map(|p| p.0))?;
        state.serialize_field(
            "markers",
            &self
                .markers
                .iter()
                .map(|m| {
                    (
                        &m.marker_selector,
                        &m.enable,
                        m.start.map(normalize_f64),
                        m.length.map(normalize_f64),
                        &m.pulse_id,
                    )
                })
                .collect::<Vec<_>>(),
        )?;
        state.serialize_field(
            "oscillator_phase",
            &self.oscillator_phase.map(normalize_f64),
        )?;
        state.serialize_field(
            "increment_oscillator_phase",
            &self.increment_oscillator_phase.map(normalize_f64),
        )?;
        state.serialize_field(
            "preferred_amplitude_register",
            &self.preferred_amplitude_register,
        )?;
        state.serialize_field("incr_phase_params", &self.incr_phase_params)?;
        state.end()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize)]
/// A unique identifier for a set of waveform samples.
pub struct SamplesSignatureID {
    /// The unique ID of the waveform samples
    pub uid: u64,
    /// The name of the waveform samples
    pub label: String,
    /// Flag whether the samples has I-component
    pub has_i: bool,
    /// Flag whether the samples has Q-component
    pub has_q: bool,
    /// Flag whether the samples has marker 1
    pub has_marker1: bool,
    /// Flag whether the samples has marker 2
    pub has_marker2: bool,
}

pub fn sort_pulses(pulses: &mut [PulseSignature]) {
    pulses.sort_by(|a, b| (a.start, a.channel).cmp(&(b.start, b.channel)));
}

/// Signature of a waveform as stored in waveform memory.
///
/// The waveform signature is a unique identifier for a waveform, which can be
/// used to identify the waveform in the waveform memory, and the waveform representation
/// can either be a set of pulses or a samples identifier, e.g. if the waveform was compressed.
///
/// The underlying promise is that two waveforms with the same signature are
/// guaranteed to resolve to the same samples per single AWG, so we need only store one of them and
/// an use them interchangeably. Using the signature across different AWGs is undefined behavior.
///
/// Alternatively, after compression it makes no sense to associate a WaveformSignature to
/// PulseSignatures anymore, since compression can cause a single pulse to be replaced with
/// any number of PlayWave and PlayHold statements. In this case, a WaveformSignature is best
/// understood as a collection of samples, with the implied promise that equal samples are only
/// uploaded to the device once.
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize)]
pub enum WaveformSignature {
    Pulses {
        /// The length of the waveform in samples
        length: Samples,
        /// The pulses that make up the waveform
        pulses: Vec<PulseSignature>,
    },
    Samples {
        /// The length of the sampled waveform in samples
        length: Samples,
        /// The ID of the sampled waveform
        samples_id: SamplesSignatureID,
    },
}

impl WaveformSignature {
    /// UID of the waveform signature.
    ///
    /// NOTE: Currently calling `.uid()` hashes the signature, therefore it is not a cheap operation.
    pub fn uid(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn length(&self) -> Samples {
        match self {
            WaveformSignature::Samples { length, .. } => *length,
            WaveformSignature::Pulses { length, .. } => *length,
        }
    }

    pub fn pulses(&self) -> Option<&Vec<PulseSignature>> {
        match self {
            WaveformSignature::Pulses { pulses, .. } => Some(pulses),
            WaveformSignature::Samples { .. } => None,
        }
    }

    pub fn pulses_mut(&mut self) -> Option<&mut Vec<PulseSignature>> {
        match self {
            WaveformSignature::Pulses { pulses, .. } => Some(pulses),
            WaveformSignature::Samples { .. } => None,
        }
    }

    /// Returns true if the waveform signature is empty, i.e. has no pulses or samples.
    pub fn is_playzero(&self) -> bool {
        match &self {
            WaveformSignature::Pulses { pulses, .. } => {
                pulses.iter().all(|pulse| pulse.pulse.is_none())
            }
            // Current we assume that whenever the samples ID is present, the waveform is not empty. Based on legacy logic.
            _ => false,
        }
    }
}

impl WaveformSignature {
    /// The maximum length of the hash parts in the signature string.
    const MAX_LEN_HASH_PARTS: usize = 8;
    /// The maximum length of the waveform property parts in the signature string.
    const MAX_LEN_PROPERTY_PARTS: usize = 56;

    fn try_write_limited(buf: &mut String, args: std::fmt::Arguments) -> bool {
        use std::fmt::Write;
        // Estimate the length of the formatted string
        let mut temp = String::new();
        temp.write_fmt(args).unwrap();
        if buf.len() + temp.len() <= WaveformSignature::MAX_LEN_PROPERTY_PARTS {
            buf.push_str(&temp);
            true
        } else {
            false
        }
    }

    /// Create a waveform signature string.
    ///
    /// The string is a unique identifier for the waveform.
    /// Signature string represents a waveform declaration in SeqC code.
    pub fn signature_string(&self) -> String {
        // Estimate the length of the pulse UID part
        // to avoid exceeding the maximum length.
        let estimated_pulse_part_length = match self {
            WaveformSignature::Samples { samples_id, .. } => samples_id.label.len(),
            WaveformSignature::Pulses { pulses, .. } => {
                // Calculate the maximum length of the pulse UIDs
                // to ensure we do not exceed the maximum length.
                pulses
                    .iter()
                    .filter_map(|pulse| pulse.pulse.as_ref().map(|p| p.uid.len()))
                    .max()
                    .unwrap_or(0)
            }
        };

        let mut retval = String::with_capacity(
            WaveformSignature::MAX_LEN_PROPERTY_PARTS
                + WaveformSignature::MAX_LEN_HASH_PARTS
                + estimated_pulse_part_length,
        );
        write!(&mut retval, "p_{:04}", self.length()).unwrap();

        if let WaveformSignature::Pulses { pulses, .. } = self {
            'pulse_loop: for pulse in pulses {
                // NOTE: The fields below should be absorbed out of the `PulseSignature`s before a signature string can be created.
                // For legacy and compatibility reasons they still remain within PulseSignature.
                assert!(
                    pulse.oscillator_phase.is_none()
                        && pulse.incr_phase_params.is_empty()
                        && pulse.increment_oscillator_phase.is_none()
                        && pulse.preferred_amplitude_register.is_none(),
                    "PulseSignature fields 'oscillator_phase', 'incr_phase_params', 'increment_oscillator_phase' and 'preferred_amplitude_register' are not supported in signature_string"
                );
                retval.push('_');
                if let Some(pulse) = &pulse.pulse {
                    retval.push_str(&pulse.uid);
                }
                // (key, separator, scale, fill)
                let fields: [(&str, &str, f64, usize); 6] = [
                    ("start", "_", 1.0, 2),
                    ("amplitude", "_a", 1e3, 4),
                    ("length", "_l", 1.0, 3),
                    ("channel", "_c", 1.0, 0),
                    ("sub_channel", "_sc", 1.0, 0),
                    ("phase", "_ap", 1e3 / 2.0 / PI, 4),
                ];
                for (key, sep, scale, fill) in fields.iter() {
                    let value_opt = match *key {
                        "start" => Some(pulse.start as f64),
                        "amplitude" => pulse.amplitude,
                        "length" => Some(pulse.length as f64),
                        "channel" => pulse.channel.map(|v| v as f64),
                        "sub_channel" => pulse.sub_channel.map(|v| v as f64),
                        "phase" => Some(pulse.phase),
                        _ => None,
                    };
                    if let Some(value) = value_opt {
                        let sign = if value < 0.0 { "m" } else { "" };
                        let abs_val = value.abs() * scale;
                        let rounded = abs_val.round() as i64;
                        if *fill > 0 {
                            if !WaveformSignature::try_write_limited(
                                &mut retval,
                                format_args!("{}{}{:0width$}", sep, sign, rounded, width = *fill),
                            ) {
                                break 'pulse_loop;
                            }
                        } else if !WaveformSignature::try_write_limited(
                            &mut retval,
                            format_args!("{sep}{sign}{rounded}"),
                        ) {
                            break 'pulse_loop;
                        }
                    }
                }
            }
        }

        if let WaveformSignature::Samples { samples_id, .. } = self {
            let sample_to_shorthand = [
                ("has_i", "si"),
                ("has_q", "sq"),
                ("has_marker1", "m1"),
                ("has_marker2", "m2"),
            ];
            for (field, shorthand) in sample_to_shorthand.iter() {
                let value = match *field {
                    "has_i" => samples_id.has_i,
                    "has_q" => samples_id.has_q,
                    "has_marker1" => samples_id.has_marker1,
                    "has_marker2" => samples_id.has_marker2,
                    _ => false,
                };
                if !value {
                    continue;
                }
                retval.push_str(shorthand);
            }
            retval.push('_');
            retval.push_str(&samples_id.label);
            retval.push('_');
        }

        let retval = crate::utils::string_sanitize(&retval);

        // End the signature string with a hash of the waveform to ensure uniqueness
        // as the parts of the formatted fields may not be unique.
        let serialized = serde_json::to_string(self).expect("Internal error: Waveform signature serialization failed while generating signature string");
        let mut hasher = Sha1::new();
        hasher.update(serialized.as_bytes());
        let hash = hasher.finalize();
        format!(
            "{}_{}",
            retval,
            &format!("{hash:x}")[..WaveformSignature::MAX_LEN_HASH_PARTS - 1]
        )
    }
}

/// Quantize amplitude baked in pulses for the given precision.
pub fn quantize_amplitude_pulse(value: f64, amplitude_resolution_range: u64) -> f64 {
    (value * amplitude_resolution_range as f64).round() / amplitude_resolution_range as f64
}

/// Quantize the command table amplitude.
///
/// For the amplitude specified by registers on the device (e.g. command table)
/// we quantize to a fixed precision of 18 bits. This
/// serves to avoid rounding errors leading to multiple command table entries
pub fn quantize_amplitude_ct(value: f64) -> f64 {
    static AMPLITUDE_RESOLUTION_CT: u32 = 1_u32 << 18;
    (value * AMPLITUDE_RESOLUTION_CT as f64).round() / AMPLITUDE_RESOLUTION_CT as f64
}

/// Quantize phase baked in pulses for the given precision.
pub fn quantize_phase_pulse(value: f64, phase_resolution_range: u64) -> f64 {
    let phase_resolution_range = phase_resolution_range as f64 / (2.0 * PI);
    let phase = (value * phase_resolution_range).round() / phase_resolution_range;
    normalize_phase(phase)
}

/// Quantize the command table phase.
///
/// For the phase specified by registers on the device (e.g. command table)
/// we quantize to a fixed precision of 24 bits. This
/// serves to avoid rounding errors leading to multiple command table entries
pub fn quantize_phase_ct(value: f64) -> f64 {
    static PHASE_RESOLUTION_CT: f64 = (1 << 24) as f64 / (2.0 * PI);
    let phase = (value * PHASE_RESOLUTION_CT).round() / PHASE_RESOLUTION_CT;
    normalize_phase(phase)
}

/// Split complex pulse amplitude
pub fn split_complex_pulse_amplitude(amplitude: Complex64, phase: f64) -> (f64, f64) {
    let theta = amplitude.arg();
    let phase = normalize_phase(phase - theta);
    (amplitude.norm().abs(), phase)
}

#[cfg(test)]
mod tests {
    use crate::ir::compilation_job::{PulseDef, PulseDefKind};
    use std::hash::DefaultHasher;

    use super::*;

    mod test_split_complex_pulse_amplitude {
        use super::*;

        #[test]
        fn test_positive_amplitude() {
            let (amplitude, phase) = split_complex_pulse_amplitude(Complex64::new(0.5, 0.0), 0.0);
            assert_eq!(amplitude, 0.5);
            assert_eq!(phase, 0.0);
        }

        #[test]
        fn test_negative_amplitude() {
            let (amplitude, phase) = split_complex_pulse_amplitude(Complex64::new(-0.5, 0.0), 0.0);
            assert_eq!(amplitude, 0.5);
            assert_eq!(phase, PI);
        }

        #[test]
        fn test_negative_im() {
            let (amplitude, phase) = split_complex_pulse_amplitude(Complex64::new(0.5, -0.5), 0.0);
            assert_eq!(amplitude, 0.5_f64.sqrt());
            // By convention, we consider a positive frequency to correspond to e^( - jwt).
            // Hence, when multiplying the samples by a number with a negative phase, the
            // actual phase in LabOneQ lingo is positive.
            assert_eq!(phase, PI / 4.0);
        }

        #[test]
        fn test_negative_re_im() {
            let (amplitude, phase) = split_complex_pulse_amplitude(Complex64::new(-0.5, -0.5), 0.0);
            assert_eq!(amplitude, 0.5_f64.sqrt());
            assert_eq!(phase, PI - PI / 4.0);
        }
    }

    fn create_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    #[test]
    fn test_pulse_signature_hash() {
        let pulse = PulseSignature {
            start: 0,
            pulse: Arc::new(PulseDef::test("".to_string(), PulseDefKind::Pulse)).into(),
            length: 1,
            amplitude: None,
            phase: 0.0,
            oscillator_frequency: 1.1.into(),
            channel: 0.into(),
            sub_channel: 0.into(),
            id_pulse_params: PulseParametersId(1).into(),
            markers: vec![],
            oscillator_phase: None,
            increment_oscillator_phase: None,
            preferred_amplitude_register: None,
            incr_phase_params: vec![],
        };

        assert_eq!(create_hash(&pulse), create_hash(&pulse.clone()));

        let mut other_pulse = pulse.clone();
        other_pulse.start = 1;
        assert_ne!(create_hash(&pulse), create_hash(&other_pulse));

        let mut other_pulse = pulse.clone();
        other_pulse.pulse = Arc::new(PulseDef::test("1".to_string(), PulseDefKind::Pulse)).into();
        assert_ne!(create_hash(&pulse), create_hash(&other_pulse));

        let mut other_pulse = pulse.clone();
        other_pulse.length = 2;
        assert_ne!(create_hash(&pulse), create_hash(&other_pulse));

        let mut other_pulse = pulse.clone();
        other_pulse.amplitude = 1.0.into();
        assert_ne!(create_hash(&pulse), create_hash(&other_pulse));

        let mut other_pulse = pulse.clone();
        other_pulse.phase = 0.1;
        assert_ne!(create_hash(&pulse), create_hash(&other_pulse));

        let mut other_pulse = pulse.clone();
        other_pulse.oscillator_frequency = 0.2.into();
        assert_ne!(create_hash(&pulse), create_hash(&other_pulse));

        let mut other_pulse = pulse.clone();
        other_pulse.channel = 1.into();
        assert_ne!(create_hash(&pulse), create_hash(&other_pulse));

        let mut other_pulse = pulse.clone();
        other_pulse.sub_channel = 1.into();
        assert_ne!(create_hash(&pulse), create_hash(&other_pulse));

        for marker in vec![
            Marker::new("1".to_string(), true, 0.0.into(), 1.0.into(), None),
            Marker::new("0".to_string(), false, 0.0.into(), 1.0.into(), None),
            Marker::new("0".to_string(), true, 0.1.into(), 1.0.into(), None),
            Marker::new("0".to_string(), true, 0.0.into(), 1.1.into(), None),
            Marker::new(
                "0".to_string(),
                true,
                0.1.into(),
                1.0.into(),
                "p".to_string().into(),
            ),
        ] {
            let mut other_pulse = pulse.clone();
            other_pulse.markers = vec![marker.clone()];
            assert_ne!(create_hash(&pulse), create_hash(&other_pulse));
        }
    }

    fn create_pulse_signature() -> PulseSignature {
        PulseSignature {
            start: 0,
            pulse: Arc::new(PulseDef::test("0".to_string(), PulseDefKind::Pulse)).into(),
            length: 1,
            amplitude: 0.5.into(),
            phase: 0.0,
            oscillator_frequency: 0.0.into(),
            channel: 0.into(),
            sub_channel: 0.into(),
            id_pulse_params: PulseParametersId(0).into(),
            markers: vec![],
            oscillator_phase: None,
            increment_oscillator_phase: None,
            preferred_amplitude_register: None,
            incr_phase_params: vec![],
        }
    }

    /// Test signature string sensitivity for all relevant fields of `PulseSignature`.
    #[test]
    fn test_signature_string_pulse_sensitivity() {
        let p0 = create_pulse_signature();

        type PulseSignatureMutator = Box<dyn FnMut(&mut PulseSignature)>;
        let mut cases: Vec<(&str, PulseSignatureMutator)> = vec![];

        cases.push(("start", Box::new(|p: &mut PulseSignature| p.start += 1)));
        cases.push(("pulse", Box::new(|p: &mut PulseSignature| p.pulse = None)));
        cases.push(("length", Box::new(|p: &mut PulseSignature| p.length += 1)));
        cases.push((
            "amplitude",
            Box::new(|p: &mut PulseSignature| p.amplitude = p.amplitude.map(|x| x + 0.1)),
        ));
        cases.push(("phase", Box::new(|p: &mut PulseSignature| p.phase += 0.1)));
        cases.push((
            "oscillator_frequency",
            Box::new(|p: &mut PulseSignature| {
                p.oscillator_frequency = p.oscillator_frequency.map(|x| x + 0.1)
            }),
        ));
        cases.push((
            "channel",
            Box::new(|p: &mut PulseSignature| p.channel = p.channel.map(|x| x + 1)),
        ));
        cases.push((
            "sub_channel",
            Box::new(|p: &mut PulseSignature| p.sub_channel = p.sub_channel.map(|x| x + 1)),
        ));
        cases.push((
            "id_pulse_params",
            Box::new(|p: &mut PulseSignature| {
                p.id_pulse_params = p.id_pulse_params.map(|x| PulseParametersId(x.0 + 1))
            }),
        ));

        for (desc, mutator) in cases.iter_mut() {
            let mut p1 = p0.clone();
            let waveform0 = WaveformSignature::Pulses {
                length: 0,
                pulses: vec![p0.clone()],
            };
            let waveform1 = WaveformSignature::Pulses {
                length: 0,
                pulses: vec![{
                    mutator(&mut p1);
                    p1.clone()
                }],
            };
            assert_ne!(
                waveform0.signature_string(),
                waveform1.signature_string(),
                "Signature string sensitivity failed on field: {desc}"
            );
            assert_ne!(
                create_hash(&waveform0),
                create_hash(&waveform1),
                "Hash sensitivity failed on field: {desc}",
            );
        }
    }

    #[test]
    fn test_pulse_signature_f64_comparison() {
        for (value, value_other) in [
            (0.0f64, -0.0f64),
            (
                -0.0000000000000017763568394002505,
                -0.0000000000000017763568394002505,
            ),
        ] {
            let mut p = create_pulse_signature();
            p.phase = value;
            let mut p_other = p.clone();
            p_other.phase = value_other;
            let wf0 = WaveformSignature::Pulses {
                length: 100,
                pulses: vec![p],
            };
            let wf1 = WaveformSignature::Pulses {
                length: 100,
                pulses: vec![p_other],
            };
            assert_eq!(wf0, wf1);
            assert_eq!(create_hash(&wf0), create_hash(&wf1));
            assert_eq!(wf0.signature_string(), wf1.signature_string());
        }
    }

    #[test]
    fn test_waveform_signature_pulse_order() {
        let p0 = create_pulse_signature();
        let mut p1 = p0.clone();
        p1.start = p0.start + 1; // Ensure p1 is different from p0

        // Same order: hashes and signature strings must match
        let wf0 = WaveformSignature::Pulses {
            length: 100,
            pulses: vec![p0.clone(), p1.clone()],
        };
        let wf1 = WaveformSignature::Pulses {
            length: 100,
            pulses: vec![p0.clone(), p1.clone()],
        };
        assert_eq!(wf0.signature_string(), wf1.signature_string());

        // Different order: hashes and signature strings must differ
        let wf2 = WaveformSignature::Pulses {
            length: 100,
            pulses: vec![p1.clone(), p0.clone()],
        };
        assert_ne!(wf0.signature_string(), wf2.signature_string());

        // Multiple identical pulses in various orderings: signature string must match
        let (p0, p1, p2) = (p0.clone(), p0.clone(), p0.clone());
        let wf3 = WaveformSignature::Pulses {
            length: 100,
            pulses: vec![p0.clone(), p1.clone(), p2.clone()],
        };
        let wf4 = WaveformSignature::Pulses {
            length: 100,
            pulses: vec![p2.clone(), p0.clone(), p1.clone()],
        };
        assert_eq!(wf3.signature_string(), wf4.signature_string());
    }
}
