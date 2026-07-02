// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hasher};
use std::sync::Arc;

use crate::Result;
use crate::ir::compilation_job::{AwgCore, DeviceKind, Signal};
use crate::ir::{IrNode, NodeKind, PlayHold, Samples, SignalUid};
use crate::pulse_map::{PulseMap, PulseWaveform};
use crate::result::{CodegenWaveform, WaveformSignatureString, WaveformStore};
use crate::signature::{SamplesSignatureID, Uid, WaveformSignature};
use crate::waveform::{CompressionProperties, WaveIdentifier, Waveform, hash_sample_buffer};
use laboneq_dsl::types::PulseUid;
use laboneq_error::bail;

pub enum WaveCompression {
    Compressed,
    HoldWave { start_index: usize, length: usize },
}

pub struct SampledWaveformSignature {
    pub signals: Vec<SignalUid>,
    pub waveforms: Vec<Waveform>,
    pub pulse_map: HashMap<PulseUid, PulseWaveform>,
    pub compression: Option<WaveCompression>,
}

impl SampledWaveformSignature {
    pub(crate) fn has_marker1(&self) -> bool {
        self.waveforms
            .iter()
            .any(|w| w.key.identifier == Some(WaveIdentifier::M1))
    }

    pub(crate) fn has_marker2(&self) -> bool {
        self.waveforms
            .iter()
            .any(|w| w.key.identifier == Some(WaveIdentifier::M2))
    }

    fn signature(&self) -> WaveformSignatureString {
        // We can assume that all waveforms in the same sampled waveform have the same signature string, since they are generated from the same original waveform.
        self.waveforms
            .first()
            .map(|w| Arc::clone(&w.key.signature))
            .unwrap_or_else(|| Arc::new("".to_string()))
    }

    /// Generates a unique identifier for the sampled waveform signature, based on the waveforms and their properties.
    ///
    /// This is an expensive operation, as it requires hashing the waveform samples. It should be used sparingly, and the result should be cached if possible.
    pub fn samples_uid(&self) -> SamplesSignatureID {
        SamplesSignatureID {
            uid: generate_samples_uid(&self.waveforms),
            compressed: self.compression.is_some(),
            has_i: self
                .waveforms
                .iter()
                .any(|w| w.key.identifier == Some(WaveIdentifier::I)),
            has_q: self
                .waveforms
                .iter()
                .any(|w| w.key.identifier == Some(WaveIdentifier::Q)),
            has_marker1: self.has_marker1(),
            has_marker2: self.has_marker2(),
        }
    }
}

/// Generates a unique identifier for a set of waveforms, based on their samples.
fn generate_samples_uid(waveforms: &[Waveform]) -> u64 {
    let mut hasher = DefaultHasher::new();
    let mut sorted: Vec<&Waveform> = waveforms.iter().collect();
    sorted.sort_by_key(|w| w.key.identifier);
    for wf in sorted {
        hash_sample_buffer(&wf.samples, &mut hasher);
    }
    hasher.finish()
}

/// Represents a part of a compressed waveform.
///
/// This enum is used to represent different parts of a waveform that can be played,
/// either as a hold or as a set of samples.
///
/// The `offset` is the relative position in the waveform where this part starts.
pub enum CompressedWaveformPart {
    PlayHold {
        offset: Samples,
        length: Samples,
    },
    PlaySamples {
        offset: Samples,
        waveform: WaveformSignature,
        signature: SampledWaveformSignature,
    },
}

/// A trait for sampling waveforms.
pub trait SampleWaveforms {
    type PulseParameters: Sync;

    fn supports_waveform_sampling(awg: &AwgCore) -> bool;

    /// Samples and compresses a batch of waveform candidates.
    fn batch_sample_and_compress(
        &self,
        awg: &AwgCore,
        waveforms: &[WaveformSamplingCandidate],
    ) -> Result<SampledWaveformCollection>;
}

/// Represents a SeqC declaration of a waveform.
#[derive(Debug)]
pub(crate) struct WaveDeclaration {
    pub length: i64,
    pub signature_string: WaveformSignatureString,
    pub has_marker1: bool,
    pub has_marker2: bool,
}

/// This enum represents a sampled waveform that can either be a direct sample or a compressed version.
enum SampledWaveformType {
    /// Represents a sampled waveform with its signature.
    Sampled(SampledWaveformSignature),
    /// Represents a compressed waveform that consists of multiple parts.
    Compressed(Vec<CompressedWaveformPart>),
}

pub struct SampledWaveformCollection {
    // A mapping from waveform UID to sampled waveform type.
    samples: IndexMap<Uid, SampledWaveformType>,
}

impl Default for SampledWaveformCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl SampledWaveformCollection {
    pub fn new() -> Self {
        SampledWaveformCollection {
            samples: IndexMap::new(),
        }
    }

    pub fn insert_sampled_signature(
        &mut self,
        waveform: &WaveformSignature,
        signature: SampledWaveformSignature,
    ) {
        self.samples
            .insert(waveform.uid(), SampledWaveformType::Sampled(signature));
    }

    pub fn insert_compressed_parts(
        &mut self,
        waveform: &WaveformSignature,
        parts: Vec<CompressedWaveformPart>,
    ) {
        self.samples
            .insert(waveform.uid(), SampledWaveformType::Compressed(parts));
    }

    fn get_sampled_waveform_signature(
        &self,
        waveform: &WaveformSignature,
    ) -> Option<&SampledWaveformType> {
        self.samples.get(&waveform.uid())
    }
}

pub struct WaveformSamplingCandidate<'a> {
    pub waveform: &'a WaveformSignature,
    pub signals: HashSet<SignalUid>,
}

fn update_waveform_candidates<'a>(
    candidates: &mut HashMap<&'a WaveformSignature, WaveformSamplingCandidate<'a>>,
    waveform: &'a WaveformSignature,
    signals: &'a [Arc<Signal>],
) {
    if let Some(candidate) = candidates.get_mut(waveform) {
        // If the waveform is already in the candidates, we can just update the signals.
        candidate
            .signals
            .extend(signals.iter().map(|rc_signal| rc_signal.uid));
    } else {
        let candidate = WaveformSamplingCandidate {
            waveform,
            signals: signals.iter().map(|rc_signal| rc_signal.uid).collect(),
        };
        candidates.insert(waveform, candidate);
    }
}

fn find_waveforms<'a>(
    node: &'a IrNode,
    candidates: &mut HashMap<&'a WaveformSignature, WaveformSamplingCandidate<'a>>,
) {
    match node.data() {
        NodeKind::PlayWave(ob) => {
            update_waveform_candidates(candidates, &ob.waveform, &ob.signals);
        }
        NodeKind::QaEvent(ob) => {
            for play_wave in ob.play_waves() {
                update_waveform_candidates(candidates, &play_wave.waveform, &play_wave.signals);
            }
        }
        _ => {
            for child in node.iter_children() {
                find_waveforms(child, candidates);
            }
        }
    }
}

pub(crate) fn collect_waveforms_for_sampling<'a>(
    node: &'a IrNode,
) -> Result<Vec<WaveformSamplingCandidate<'a>>> {
    let mut sampling_candidates = HashMap::new();
    find_waveforms(node, &mut sampling_candidates);
    Ok(sampling_candidates.into_values().collect())
}

#[derive(Default)]
pub(crate) struct ProcessedWaveforms {
    pub waveforms: Vec<CodegenWaveform>,
    pub waveform_store: WaveformStore,
    pub wave_declarations: Vec<WaveDeclaration>,
    pub long_readout_signals: HashSet<SignalUid>,
    pub pulse_map: PulseMap,
}

/// Split waveform nodes that have been compressed.
///
/// This function traverses the IR tree, looking for waveform nodes that are compressed.
/// Compressed play wave nodes are replaced with [`NodeKind::PlayHold`] and [`NodeKind::PlayWave`] nodes.
fn split_compressed_waveforms(
    node: &mut IrNode,
    sampled_waveform_signatures: &SampledWaveformCollection,
) -> Result<Vec<IrNode>> {
    match node.data() {
        NodeKind::PlayWave(ob) => {
            if let Some(SampledWaveformType::Compressed(compressed_parts)) =
                sampled_waveform_signatures.get_sampled_waveform_signature(&ob.waveform)
            {
                let mut new_nodes = vec![];
                if compressed_parts.is_empty() {
                    return Ok(new_nodes);
                }
                for part in compressed_parts {
                    match part {
                        CompressedWaveformPart::PlayHold { offset, length } => {
                            let new_event = PlayHold { length: *length };
                            let new_node =
                                IrNode::new(NodeKind::PlayHold(new_event), node.offset() + offset);
                            new_nodes.push(new_node);
                        }
                        CompressedWaveformPart::PlaySamples {
                            offset, waveform, ..
                        } => {
                            // Compressed waveform part inherits the properties of the original play wave
                            let mut new_playwave = ob.clone();
                            new_playwave.waveform = waveform.clone();
                            let new_node = IrNode::new(
                                NodeKind::PlayWave(new_playwave),
                                node.offset() + offset,
                            );
                            new_nodes.push(new_node);
                        }
                    }
                }
                Ok(new_nodes)
            } else {
                // Waveform was not compressed
                Ok(vec![])
            }
        }
        _ => {
            let mut new_events = vec![];
            for (index, child) in node.iter_children_mut().enumerate() {
                let new_parts = split_compressed_waveforms(child, sampled_waveform_signatures)?;
                if new_parts.is_empty() {
                    continue;
                } else {
                    new_events.push((index, new_parts));
                }
            }
            // Replace orignal waveform nodes with new nodes
            // We iterate backwards to avoid index issues when removing nodes
            new_events.reverse();
            for (index, new_parts) in new_events.into_iter() {
                for (new_pos, new_node) in new_parts.into_iter().enumerate() {
                    node.insert_child_node(index + new_pos + 1, new_node);
                }
                node.remove_child(index);
            }
            Ok(vec![])
        }
    }
}

/// Collect sampled signatures from the waveform sampling result.
///
/// Waveforms that were compressed are omitted from the output.
fn collect_sampled_signatures(
    sampled_waveform_signatures: SampledWaveformCollection,
) -> HashMap<Uid, SampledWaveformSignature> {
    let mut sampled_signatures = HashMap::new();
    for (waveform_uid, sampled_waveform) in sampled_waveform_signatures.samples {
        match sampled_waveform {
            SampledWaveformType::Sampled(signature) => {
                sampled_signatures.insert(waveform_uid, signature);
            }
            SampledWaveformType::Compressed(parts) => {
                // We omit the original waveform, which was compressed
                for part in parts.into_iter() {
                    if let CompressedWaveformPart::PlaySamples {
                        waveform,
                        signature,
                        ..
                    } = part
                    {
                        sampled_signatures.insert(waveform.uid(), signature);
                    }
                }
            }
        }
    }
    sampled_signatures
}

struct PassContext {
    sampled_waveform_signatures: HashMap<Uid, SampledWaveformSignature>,
    // Collect output into vectors to keep track of the order of waveforms.
    // The output should be deterministic, so we use a vector instead of a hash map.
    sampled_waveforms: Vec<SampledWaveformSignature>,
    wave_declarations: Vec<WaveDeclaration>,
}

impl PassContext {
    fn new(sampled_waveform_signatures: HashMap<Uid, SampledWaveformSignature>) -> Self {
        PassContext {
            sampled_waveform_signatures,
            sampled_waveforms: vec![],
            wave_declarations: vec![],
        }
    }

    fn register_waveform(&mut self, waveform: &WaveformSignature) {
        let waveform_uid = waveform.uid();

        // Split waveform signature into sampled waveform and wave declaration.
        if let Some(mut sampled_waveform) = self.sampled_waveform_signatures.remove(&waveform_uid) {
            let signature_string = Arc::new(waveform.signature_string());

            let wave_declaration = WaveDeclaration {
                length: waveform.length(),
                signature_string: Arc::clone(&signature_string),
                has_marker1: sampled_waveform.has_marker1(),
                has_marker2: sampled_waveform.has_marker2(),
            };
            // Update the signature string for all waveforms in the sampled waveform.
            // This is needed for compressed waveforms, where the original waveform is replaced with multiple new waveforms that all share the same signature string.
            for waveform in sampled_waveform.waveforms.iter_mut() {
                waveform.key.signature = Arc::clone(&signature_string);
            }
            self.sampled_waveforms.push(sampled_waveform);
            self.wave_declarations.push(wave_declaration);
        }
    }
}

fn collect_waveform_info(node: &IrNode, ctx: &mut PassContext) {
    match node.data() {
        NodeKind::PlayWave(ob) => {
            ctx.register_waveform(&ob.waveform);
        }
        NodeKind::QaEvent(ob) => {
            for play_wave in ob.play_waves() {
                ctx.register_waveform(&play_wave.waveform);
            }
        }
        _ => {
            for child in node.iter_children() {
                collect_waveform_info(child, ctx);
            }
        }
    }
}

fn generate_output(
    node: &IrNode,
    sampled_waveform_signatures: HashMap<Uid, SampledWaveformSignature>,
) -> (Vec<SampledWaveformSignature>, Vec<WaveDeclaration>) {
    let mut ctx = PassContext::new(sampled_waveform_signatures);
    collect_waveform_info(node, &mut ctx);
    (ctx.sampled_waveforms, ctx.wave_declarations)
}

fn validate_waveforms(waveforms: &[&WaveformSamplingCandidate<'_>], awg: &AwgCore) -> Result<()> {
    if &DeviceKind::SHFQA == awg.device.kind() && waveforms.len() > 1 {
        let mut signal_to_pulses: HashMap<SignalUid, HashSet<&WaveformSignature>> = HashMap::new();
        for waveform in waveforms.iter() {
            for signal in waveform.signals.iter() {
                signal_to_pulses
                    .entry(*signal)
                    .or_default()
                    .insert(waveform.waveform);
            }
        }
        for (signal, waveforms) in signal_to_pulses.iter() {
            // Ensure that there is only one unique waveform per signal
            if waveforms.len() > 1 {
                bail!(
                    "Too many unique pulses on signal '{}'. Using more than one unique pulse on a SHFQA generator channel is not supported. \
                    Sweeping a SHFQA generator channel is not supported in real-time. Ensure each real-time loop uses the same pulse on a given signal.",
                    signal.0,
                );
            } else if waveforms.len() == 1 {
                // Ensure that there is only one unique pulse per signal
                let waveform = waveforms.iter().next().unwrap();
                if waveform.pulses().iter().len() > 1 {
                    bail!(
                        "Too many unique pulses on signal '{}'. Using more than one unique pulse on a SHFQA generator channel is not supported. \
                        Sweeping a SHFQA generator channel is not supported in real-time. Ensure each real-time loop uses the same pulse on a given signal.",
                        signal.0,
                    );
                }
            }
        }
    }
    Ok(())
}

/// Transformation pass to collect and finalize waveforms based on sampled waveform signatures.
///
/// This function collects waveforms from the IR node, samples them using the provided [`SampleWaveforms`],
/// splits compressed waveforms, and generates the final output containing sampled waveforms and their declarations.
///
/// # Returns
///
/// A result containing an [`ProcessedWaveforms`] struct, which includes the sampled waveforms and their declarations that
/// exists in the IR node after the transformation pass.
pub(crate) fn collect_and_finalize_waveforms<T: SampleWaveforms>(
    node: &mut IrNode,
    waveform_sampler: &T,
    awg: &AwgCore,
) -> Result<ProcessedWaveforms> {
    if !T::supports_waveform_sampling(awg) {
        return Ok(ProcessedWaveforms::default());
    }
    let waveforms = collect_waveforms_for_sampling(node)?;
    validate_waveforms(&waveforms.iter().collect::<Vec<_>>(), awg)?;
    let sampled_waveform_signatures =
        waveform_sampler.batch_sample_and_compress(awg, &waveforms)?;
    split_compressed_waveforms(node, &sampled_waveform_signatures)?;
    let sampled_waveforms_signatures = collect_sampled_signatures(sampled_waveform_signatures);
    let (sampled_waveforms, wave_declarations) =
        generate_output(node, sampled_waveforms_signatures);
    let output = create_output(sampled_waveforms, wave_declarations)?;
    Ok(output)
}

fn create_output(
    sampled_waveforms: Vec<SampledWaveformSignature>,
    wave_declarations: Vec<WaveDeclaration>,
) -> Result<ProcessedWaveforms> {
    let mut waveform_store = WaveformStore::default();
    let mut codegen_waveforms =
        Vec::with_capacity(sampled_waveforms.len() + wave_declarations.len());
    let mut pulse_map = PulseMap::with_capacity(sampled_waveforms.len());
    let mut long_readout_signals = HashSet::new();

    for sampled_waveform in sampled_waveforms.into_iter() {
        let signature = sampled_waveform.signature();
        for (pulse_uid, waveform_map) in sampled_waveform.pulse_map {
            pulse_map.insert(pulse_uid, Arc::clone(&signature), waveform_map);
        }

        for wave in sampled_waveform.waveforms {
            if let Some(WaveCompression::HoldWave { .. }) = sampled_waveform.compression {
                long_readout_signals.extend(sampled_waveform.signals.iter());
            }
            let waveform = CodegenWaveform {
                key: wave.key.clone(),
                compression_properties: match sampled_waveform.compression {
                    Some(WaveCompression::Compressed) => None,
                    Some(WaveCompression::HoldWave {
                        start_index,
                        length,
                    }) => Some(CompressionProperties {
                        hold_start: start_index,
                        hold_length: length,
                    }),
                    None => Default::default(),
                },
                downsampling_factor: None,
            };
            codegen_waveforms.push(waveform);
            waveform_store.insert(wave.key.clone(), wave.samples)?;
        }
    }
    let output = ProcessedWaveforms {
        waveforms: codegen_waveforms,
        waveform_store,
        wave_declarations,
        long_readout_signals,
        pulse_map,
    };
    Ok(output)
}
