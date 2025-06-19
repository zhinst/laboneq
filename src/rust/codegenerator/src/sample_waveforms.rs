// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Result;
use crate::ir::{IrNode, NodeKind, PlayHold, Samples};
use crate::signature::WaveformSignature;

/// A trait that defines the signature of a sampled waveform.
///
/// This trait is used to represent the properties of a sampled waveform,
/// including whether it has specific markers and the inner signature type.
pub trait SampledWaveformSignature {
    type Inner;

    fn has_marker1(&self) -> bool;
    fn has_marker2(&self) -> bool;
    fn signature(&self) -> Self::Inner;
}

/// Represents a part of a compressed waveform.
///
/// This enum is used to represent different parts of a waveform that can be played,
/// either as a hold or as a set of samples.
///
/// The `offset` is the relative position in the waveform where this part starts.
#[derive(Clone, Debug)]
pub enum CompressedWaveformPart<T: SampledWaveformSignature> {
    PlayHold {
        offset: Samples,
        length: Samples,
    },
    PlaySamples {
        offset: Samples,
        waveform: WaveformSignature,
        signature: T,
    },
}

/// A trait for sampling waveforms.
pub trait SampleWaveforms {
    type Signature: SampledWaveformSignature;
    /// Samples and compresses a batch of waveform candidates.
    fn batch_sample_and_compress(
        &self,
        waveforms: &[WaveformSamplingCandidate],
    ) -> Result<SampledWaveformCollection<Self::Signature>>;
}

/// Represents output of an AWG waveform sampling.
#[derive(Debug)]
pub struct SampledWaveform<T: SampledWaveformSignature> {
    pub signals: HashSet<String>,
    pub signature_string: Arc<String>,
    pub signature: T,
}

/// Represents a SeqC declaration of a waveform.
#[derive(Debug)]
pub struct WaveDeclaration {
    pub length: i64,
    pub signature_string: Arc<String>,
    pub has_marker1: bool,
    pub has_marker2: bool,
}

/// This enum represents a sampled waveform that can either be a direct sample or a compressed version.
enum SampledWaveformType<T: SampledWaveformSignature> {
    /// Represents a sampled waveform with its signature.
    Sampled(T),
    /// Represents a compressed waveform that consists of multiple parts.
    Compressed(Vec<CompressedWaveformPart<T>>),
}

pub struct SampledWaveformCollection<T: SampledWaveformSignature> {
    // A mapping from waveform UID to sampled waveform type.
    samples: HashMap<u64, SampledWaveformType<T>>,
}

impl<T: SampledWaveformSignature> Default for SampledWaveformCollection<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SampledWaveformSignature> SampledWaveformCollection<T> {
    pub fn new() -> Self {
        SampledWaveformCollection {
            samples: HashMap::new(),
        }
    }

    pub fn insert_sampled_signature(&mut self, waveform: &WaveformSignature, signature: T) {
        self.samples
            .insert(waveform.uid(), SampledWaveformType::Sampled(signature));
    }

    pub fn insert_compressed_parts(
        &mut self,
        waveform: &WaveformSignature,
        parts: Vec<CompressedWaveformPart<T>>,
    ) {
        self.samples
            .insert(waveform.uid(), SampledWaveformType::Compressed(parts));
    }

    fn get_sampled_waveform_signature(
        &self,
        waveform: &WaveformSignature,
    ) -> Option<&SampledWaveformType<T>> {
        self.samples.get(&waveform.uid())
    }
}

pub struct WaveformSamplingCandidate<'a> {
    pub waveform: &'a WaveformSignature,
    pub signals: HashSet<&'a str>,
}

fn find_waveforms<'a>(
    node: &'a IrNode,
    // ctx: & mut HashSet<&'a WaveformSignature>,
    candidates: &mut HashMap<&'a WaveformSignature, WaveformSamplingCandidate<'a>>,
) {
    if let NodeKind::PlayWave(ob) = node.data() {
        // TODO: Should we update the signals?
        if let Some(candidate) = candidates.get_mut(&ob.waveform) {
            // If the waveform is already in the candidates, we can just update the signals.
            candidate
                .signals
                .extend(ob.signals.iter().map(|rc_signal| rc_signal.uid.as_str()));
            return;
        }
        let candidate = WaveformSamplingCandidate {
            waveform: &ob.waveform,
            signals: ob
                .signals
                .iter()
                .map(|rc_signal| rc_signal.uid.as_str())
                .collect(),
        };
        candidates.insert(&ob.waveform, candidate);
    } else {
        for child in node.iter_children() {
            find_waveforms(child, candidates);
        }
    }
}

pub fn collect_waveforms_for_sampling<'a>(
    node: &'a IrNode,
) -> Result<Vec<WaveformSamplingCandidate<'a>>> {
    let mut sampling_candidates = HashMap::new();
    find_waveforms(node, &mut sampling_candidates);
    Ok(sampling_candidates.into_values().collect())
}

pub struct AwgWaveforms<T: SampledWaveformSignature> {
    sampled_waveforms: Vec<SampledWaveform<T>>,
    wave_declarations: Vec<WaveDeclaration>,
}

impl<T: SampledWaveformSignature> Default for AwgWaveforms<T> {
    fn default() -> Self {
        AwgWaveforms {
            sampled_waveforms: vec![],
            wave_declarations: vec![],
        }
    }
}

impl<T: SampledWaveformSignature> AwgWaveforms<T> {
    pub fn into_inner(self) -> (Vec<SampledWaveform<T>>, Vec<WaveDeclaration>) {
        (self.sampled_waveforms, self.wave_declarations)
    }
}

/// Split waveform nodes that have been compressed.
///
/// This function traverses the IR tree, looking for waveform nodes that are compressed.
/// Compressed play wave nodes are replaced with [`NodeKind::PlayHold`] and [`NodeKind::PlayWave`] nodes.
fn split_compressed_waveforms<T: SampledWaveformSignature>(
    node: &mut IrNode,
    sampled_waveform_signatures: &SampledWaveformCollection<T>,
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
fn collect_sampled_signatures<T: SampledWaveformSignature>(
    sampled_waveform_signatures: SampledWaveformCollection<T>,
) -> HashMap<u64, T> {
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

struct PassContext<T: SampledWaveformSignature> {
    sampled_waveform_signatures: HashMap<u64, T>,
    // Collect output into vectors to keep track of the order of waveforms.
    // The output should be deterministic, so we use a vector instead of a hash map.
    // Alternative way would be to insert timestamp into the output structs and sort them later.
    sampled_waveforms: Vec<SampledWaveform<T>>,
    wave_declarations: Vec<WaveDeclaration>,
    waveforms_handled_to_index: HashMap<u64, usize>,
}

fn collect_waveform_info<T: SampledWaveformSignature>(node: &IrNode, ctx: &mut PassContext<T>) {
    match node.data() {
        NodeKind::PlayWave(ob) => {
            let waveform_uid = ob.waveform.uid();
            if let Some(wave_index) = ctx.waveforms_handled_to_index.get(&waveform_uid) {
                // If the waveform has already been processed, we only update the signals.
                let sampled_waveform = &mut ctx.sampled_waveforms[*wave_index];
                sampled_waveform
                    .signals
                    .extend(ob.signals.iter().map(|s| s.uid.clone()));
                return;
            }
            let sampled_waveform = ctx.sampled_waveform_signatures.remove(&waveform_uid);
            // Split waveform signature into sampled waveform and wave declaration.
            if let Some(sampled_waveform) = sampled_waveform {
                let signature_string = Arc::new(ob.waveform.signature_string());
                let signals: HashSet<String> = ob.signals.iter().map(|s| s.uid.clone()).collect();

                let wave_declaration = WaveDeclaration {
                    length: ob.waveform.length(),
                    signature_string: signature_string.clone(),
                    has_marker1: sampled_waveform.has_marker1(),
                    has_marker2: sampled_waveform.has_marker2(),
                };
                let sampled_waveform_data = SampledWaveform {
                    signals: signals.clone(),
                    signature_string: signature_string.clone(),
                    signature: sampled_waveform,
                };
                ctx.sampled_waveforms.push(sampled_waveform_data);
                ctx.wave_declarations.push(wave_declaration);
                ctx.waveforms_handled_to_index
                    .insert(waveform_uid, ctx.sampled_waveforms.len() - 1);
            }
        }
        _ => {
            for child in node.iter_children() {
                collect_waveform_info(child, ctx);
            }
        }
    }
}

fn generate_output<T: SampledWaveformSignature>(
    node: &IrNode,
    sampled_waveform_signatures: HashMap<u64, T>,
) -> (Vec<SampledWaveform<T>>, Vec<WaveDeclaration>) {
    let mut ctx = PassContext {
        sampled_waveform_signatures,
        sampled_waveforms: vec![],
        wave_declarations: vec![],
        waveforms_handled_to_index: HashMap::new(),
    };
    collect_waveform_info(node, &mut ctx);
    (ctx.sampled_waveforms, ctx.wave_declarations)
}

/// Transformation pass to collect and finalize waveforms based on sampled waveform signatures.
///
/// This function collects waveforms from the IR node, samples them using the provided [`WaveformSampler`],
/// splits compressed waveforms, and generates the final output containing sampled waveforms and their declarations.
///
/// # Returns
///
/// A result containing an [`AwgWaveforms`] struct, which includes the sampled waveforms and their declarations that
/// exists in the IR node after the transformation pass.
pub fn collect_and_finalize_waveforms<T: SampleWaveforms>(
    node: &mut IrNode,
    waveform_sampler: T,
) -> Result<AwgWaveforms<T::Signature>> {
    let waveforms = collect_waveforms_for_sampling(node)?;
    let sampled_waveform_signatures = waveform_sampler.batch_sample_and_compress(&waveforms)?;
    split_compressed_waveforms(node, &sampled_waveform_signatures)?;
    let sampled_waveforms_signatures = collect_sampled_signatures(sampled_waveform_signatures);
    let (sampled_waveforms, wave_declarations) =
        generate_output(node, sampled_waveforms_signatures);
    let out = AwgWaveforms {
        sampled_waveforms,
        wave_declarations,
    };
    Ok(out)
}
