// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use crate::ir::compilation_job::{AwgCore, DeviceKind, Signal, SignalKind};
use crate::ir::experiment::PulseParametersId;
use crate::ir::{IrNode, NodeKind, PlayAcquire, PlayHold, Samples};
use crate::signature::{Uid, WaveformSignature};
use crate::{Error, Result};
use indexmap::{IndexMap, IndexSet};

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

/// Properties of integration weights for a pulse.
///
/// NOTE: The properties will result in unique integration weights
/// per device type.
#[derive(Clone, Debug)]
struct KernelProperties<'a> {
    pulse_id: &'a str,
    pulse_parameters_id: Option<PulseParametersId>,
    oscillator_frequency: f64,
}

impl Eq for KernelProperties<'_> {}

impl PartialEq for KernelProperties<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.pulse_id == other.pulse_id
            && self.pulse_parameters_id == other.pulse_parameters_id
            && crate::utils::normalize_f64(self.oscillator_frequency)
                == crate::utils::normalize_f64(other.oscillator_frequency)
    }
}

impl Hash for KernelProperties<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pulse_id.hash(state);
        self.pulse_parameters_id.hash(state);
        crate::utils::normalize_f64(self.oscillator_frequency).hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct IntegrationKernel<'a> {
    properties: KernelProperties<'a>,
    signals: Vec<&'a str>,
}

impl IntegrationKernel<'_> {
    pub fn pulse_id(&self) -> &str {
        self.properties.pulse_id
    }

    pub fn pulse_parameters_id(&self) -> Option<PulseParametersId> {
        self.properties.pulse_parameters_id
    }

    pub fn oscillator_frequency(&self) -> f64 {
        self.properties.oscillator_frequency
    }

    pub fn signals(&self) -> &Vec<&str> {
        &self.signals
    }
}

/// A trait for sampling waveforms.
pub trait SampleWaveforms {
    type Signature: SampledWaveformSignature + Send + Sync;
    type IntegrationWeight: Send + Sync;
    type PulseParameters: Sync;

    fn supports_waveform_sampling(awg: &AwgCore) -> bool;

    /// Calculates integration weights for a batch of integration kernels.
    fn batch_calculate_integration_weights(
        &self,
        awg: &AwgCore,
        kernels: Vec<IntegrationKernel<'_>>,
    ) -> Result<Vec<Self::IntegrationWeight>>;

    /// Samples and compresses a batch of waveform candidates.
    fn batch_sample_and_compress(
        &self,
        awg: &AwgCore,
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
    samples: HashMap<Uid, SampledWaveformType<T>>,
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

fn update_waveform_candidates<'a>(
    candidates: &mut HashMap<&'a WaveformSignature, WaveformSamplingCandidate<'a>>,
    waveform: &'a WaveformSignature,
    signals: &'a [Arc<Signal>],
) {
    if let Some(candidate) = candidates.get_mut(waveform) {
        // If the waveform is already in the candidates, we can just update the signals.
        candidate
            .signals
            .extend(signals.iter().map(|rc_signal| rc_signal.uid.as_str()));
    } else {
        let candidate = WaveformSamplingCandidate {
            waveform,
            signals: signals
                .iter()
                .map(|rc_signal| rc_signal.uid.as_str())
                .collect(),
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
) -> HashMap<Uid, T> {
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
    sampled_waveform_signatures: HashMap<Uid, T>,
    // Collect output into vectors to keep track of the order of waveforms.
    // The output should be deterministic, so we use a vector instead of a hash map.
    // Alternative way would be to insert timestamp into the output structs and sort them later.
    sampled_waveforms: Vec<SampledWaveform<T>>,
    wave_declarations: Vec<WaveDeclaration>,
    waveforms_handled_to_index: HashMap<Uid, usize>,
}

impl<T: SampledWaveformSignature> PassContext<T> {
    fn new(sampled_waveform_signatures: HashMap<Uid, T>) -> Self {
        PassContext {
            sampled_waveform_signatures,
            sampled_waveforms: vec![],
            wave_declarations: vec![],
            waveforms_handled_to_index: HashMap::new(),
        }
    }

    fn register_waveform(&mut self, waveform: &WaveformSignature, signals: &[Arc<Signal>]) {
        let waveform_uid = waveform.uid();
        if let Some(wave_index) = self.waveforms_handled_to_index.get(&waveform_uid) {
            // If the waveform has already been processed, we only update the signals.
            let sampled_waveform = &mut self.sampled_waveforms[*wave_index];
            sampled_waveform
                .signals
                .extend(signals.iter().map(|s| s.uid.clone()));
            return;
        }
        let sampled_waveform = self.sampled_waveform_signatures.remove(&waveform_uid);
        // Split waveform signature into sampled waveform and wave declaration.
        if let Some(sampled_waveform) = sampled_waveform {
            let signature_string = Arc::new(waveform.signature_string());
            let signals: HashSet<String> = signals.iter().map(|s| s.uid.clone()).collect();

            let wave_declaration = WaveDeclaration {
                length: waveform.length(),
                signature_string: Arc::clone(&signature_string),
                has_marker1: sampled_waveform.has_marker1(),
                has_marker2: sampled_waveform.has_marker2(),
            };
            let sampled_waveform_data = SampledWaveform {
                signals,
                signature_string: Arc::clone(&signature_string),
                signature: sampled_waveform,
            };
            self.sampled_waveforms.push(sampled_waveform_data);
            self.wave_declarations.push(wave_declaration);
            self.waveforms_handled_to_index
                .insert(waveform_uid, self.sampled_waveforms.len() - 1);
        }
    }
}

fn collect_waveform_info<T: SampledWaveformSignature>(node: &IrNode, ctx: &mut PassContext<T>) {
    match node.data() {
        NodeKind::PlayWave(ob) => {
            ctx.register_waveform(&ob.waveform, &ob.signals);
        }
        NodeKind::QaEvent(ob) => {
            for play_wave in ob.play_waves() {
                ctx.register_waveform(&play_wave.waveform, &play_wave.signals);
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
    sampled_waveform_signatures: HashMap<Uid, T>,
) -> (Vec<SampledWaveform<T>>, Vec<WaveDeclaration>) {
    let mut ctx = PassContext::new(sampled_waveform_signatures);
    collect_waveform_info(node, &mut ctx);
    (ctx.sampled_waveforms, ctx.wave_declarations)
}

fn validate_waveforms(waveforms: &[&WaveformSamplingCandidate<'_>], awg: &AwgCore) -> Result<()> {
    if &DeviceKind::SHFQA == awg.device.kind() && waveforms.len() > 1 {
        let mut signal_to_pulses: HashMap<&str, HashSet<&WaveformSignature>> = HashMap::new();
        for waveform in waveforms.iter() {
            for signal in waveform.signals.iter() {
                signal_to_pulses
                    .entry(signal)
                    .or_default()
                    .insert(waveform.waveform);
            }
        }
        for (signal, waveforms) in signal_to_pulses.iter() {
            // Ensure that there is only one unique waveform per signal
            if waveforms.len() > 1 {
                return Err(Error::new(format!(
                    "Too many unique pulses on signal '{signal}'. Using more than one unique pulse on a SHFQA generator channel is not supported. \
                    Sweeping a SHFQA generator channel is not supported in real-time. Ensure each real-time loop uses the same pulse on a given signal.",
                )));
            } else if waveforms.len() == 1 {
                // Ensure that there is only one unique pulse per signal
                let waveform = waveforms.iter().next().unwrap();
                if waveform.pulses().iter().len() > 1 {
                    return Err(Error::new(format!(
                        "Too many unique pulses on signal '{signal}'. Using more than one unique pulse on a SHFQA generator channel is not supported. \
                        Sweeping a SHFQA generator channel is not supported in real-time. Ensure each real-time loop uses the same pulse on a given signal.",
                    )));
                }
            }
        }
    }
    Ok(())
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
    waveform_sampler: &T,
    awg: &AwgCore,
) -> Result<AwgWaveforms<T::Signature>> {
    let waveforms = collect_waveforms_for_sampling(node)?;
    validate_waveforms(&waveforms.iter().collect::<Vec<_>>(), awg)?;
    let sampled_waveform_signatures =
        waveform_sampler.batch_sample_and_compress(awg, &waveforms)?;
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

fn collect_kernel_properties(event: &PlayAcquire) -> IndexSet<KernelProperties<'_>> {
    let mut properties = IndexSet::with_capacity(event.pulse_defs().len());
    for (pulse_def, pulse_parameters_id) in event
        .pulse_defs()
        .iter()
        .zip(event.id_pulse_params().iter())
    {
        if pulse_def.pulse_type.is_none() {
            // Skip pulses without a type, they do not contribute to integration weights
            continue;
        }
        let kernel = KernelProperties {
            pulse_id: pulse_def.uid.as_str(),
            pulse_parameters_id: *pulse_parameters_id,
            oscillator_frequency: event.oscillator_frequency(),
        };
        properties.insert(kernel);
    }
    properties
}

fn update_kernel_properties<'a>(
    properties: &mut IndexMap<&'a str, IndexSet<KernelProperties<'a>>>,
    key: &'a str,
    weights: IndexSet<KernelProperties<'a>>,
) -> Result<()> {
    if let Some(weight_properties) = properties.get_mut(key) {
        if weight_properties != &weights {
            return Err(Error::new(
                format!(
                    "Using different integration kernels on a single signal '{key}' is unsupported. They either differ on the pulse definitions or on the oscillator frequency.",
                ).as_str()));
        }
    } else {
        properties.insert(key, weights);
    }
    Ok(())
}

fn collect_integration_weights_properties<'a>(
    node: &'a IrNode,
    properties: &mut IndexMap<&'a str, IndexSet<KernelProperties<'a>>>,
) -> Result<()> {
    match node.data() {
        NodeKind::Acquire(ob) => {
            let weights: IndexSet<KernelProperties<'_>> = collect_kernel_properties(ob);
            update_kernel_properties(properties, ob.signal().uid.as_str(), weights)?;
        }
        NodeKind::QaEvent(ob) => {
            for acquire in ob.acquires() {
                let weights = collect_kernel_properties(acquire);
                update_kernel_properties(properties, acquire.signal().uid.as_str(), weights)?;
            }
        }
        _ => {
            for child in node.iter_children() {
                collect_integration_weights_properties(child, properties)?;
            }
        }
    }
    Ok(())
}

/// Collect integration kernels from the IR.
///
/// Integration kernels are deduplicated across all signals on the given AWG.
pub fn collect_integration_kernels<'a>(
    node: &'a IrNode,
    awg: &AwgCore,
) -> Result<Vec<IntegrationKernel<'a>>> {
    let has_integration_signals = awg
        .signals
        .iter()
        .any(|s| s.kind == SignalKind::INTEGRATION);
    if !has_integration_signals {
        // No integration signals, no need to collect integration weights
        return Ok(vec![]);
    }
    let mut weight_collection = IndexMap::new();
    collect_integration_weights_properties(node, &mut weight_collection)?;
    // Deduplicate the integration weight properties
    let mut unique_weights = IndexMap::new();
    for (signal_uid, weight_properties) in weight_collection {
        // Group weights by pulse definition
        let mut unique_pulse_ids = IndexSet::new();
        for weight in weight_properties {
            if unique_pulse_ids.contains(weight.pulse_id) {
                return Err(Error::new(
                    format!(
                        "Using different integration kernels on a single signal '{signal_uid}' is unsupported. They either differ on the pulse definitions or on the oscillator frequency.",
                    ).as_str()));
            }
            unique_pulse_ids.insert(weight.pulse_id);
            unique_weights
                .entry(weight)
                .or_insert_with(Vec::new)
                .push(signal_uid);
        }
    }
    let kernels = unique_weights
        .into_iter()
        .map(|(properties, signals)| IntegrationKernel {
            properties,
            signals,
        })
        .collect();
    Ok(kernels)
}
