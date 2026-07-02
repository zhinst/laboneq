// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Integration kernel pipeline: collect from IR, sample, and assemble into output-ready form.

use std::collections::HashSet;

use indexmap::{IndexMap, IndexSet};
use laboneq_error::bail;

use crate::Result;
use crate::context::CodeGenContext;
use crate::ir::compilation_job::AwgCore;
use crate::ir::experiment::PulseParametersId;
use crate::ir::{IrNode, NodeKind, PlayAcquire, SignalUid};
use crate::result::IntegrationWeight;
use crate::waveform::{CodegenWaveform, Waveform, WaveformSignatureString, WaveformStore};

pub struct SampledIntegrationKernel {
    pub(crate) signals: Vec<SignalUid>,
    pub(crate) downsampling_factor: Option<u8>,
    pub(crate) waveforms: Vec<Waveform>,
}

impl SampledIntegrationKernel {
    pub fn new(
        signals: Vec<SignalUid>,
        downsampling_factor: Option<u8>,
        waveforms: Vec<Waveform>,
    ) -> Self {
        assert!(
            !waveforms.is_empty(),
            "At least one waveform is required for a sampled integration kernel."
        );
        assert!(
            waveforms
                .iter()
                .all(|wf| wf.key.signature == waveforms[0].key.signature),
            "Internal error: Integration kernel waveforms have mismatched signatures"
        );
        Self {
            signals,
            downsampling_factor,
            waveforms,
        }
    }

    pub(crate) fn basename(&self) -> &WaveformSignatureString {
        &self.waveforms[0].key.signature
    }
}

/// A trait for sampling waveforms.
pub trait SampleIntegrationKernels: Send + Sync {
    type PulseParameters: Sync;

    /// Calculates integration weights for a batch of integration kernels.
    fn sample_integration_kernels(
        &self,
        awg: &AwgCore,
        kernels: Vec<IntegrationKernel<'_>>,
    ) -> Result<Vec<SampledIntegrationKernel>>;
}

pub(crate) struct ProcessedKernels {
    pub(crate) waveforms: Vec<CodegenWaveform>,
    pub(crate) waveform_store: WaveformStore,
    pub(crate) weights: Vec<IntegrationWeight>,
    pub(crate) long_readout_signals: HashSet<SignalUid>,
}

/// Collect integration kernels from the IR, sample them, and assemble into output-ready form.
pub(crate) fn collect_and_process_integration_kernels<T: SampleIntegrationKernels>(
    node: &IrNode,
    awg: &AwgCore,
    sampler: &T,
    ctx: &CodeGenContext,
) -> Result<ProcessedKernels> {
    let kernels = collect_integration_kernels(node, awg)?;
    let sampled = sampler.sample_integration_kernels(awg, kernels)?;
    let weights = build_integration_weights(&sampled, ctx);
    let long_readout_signals = resolve_long_readout_signals(&sampled);
    let mut all_waveforms = Vec::new();
    let stores: Vec<WaveformStore> = sampled
        .into_iter()
        .map(|kernel| -> Result<WaveformStore> {
            let (waveforms, store) = create_integration_kernel_waveforms(kernel)?;
            all_waveforms.extend(waveforms);
            Ok(store)
        })
        .collect::<Result<Vec<_>>>()?;
    let waveform_store = WaveformStore::merge(stores)?;
    Ok(ProcessedKernels {
        waveforms: all_waveforms,
        waveform_store,
        weights,
        long_readout_signals,
    })
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

impl std::hash::Hash for KernelProperties<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pulse_id.hash(state);
        self.pulse_parameters_id.hash(state);
        crate::utils::normalize_f64(self.oscillator_frequency).hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct IntegrationKernel<'a> {
    properties: KernelProperties<'a>,
    signals: Vec<SignalUid>,
}

impl<'a> IntegrationKernel<'a> {
    fn new(mut signals: Vec<SignalUid>, properties: KernelProperties<'a>) -> Self {
        signals.sort(); // Ensure deterministic order of signals for testing and code generation
        Self {
            properties,
            signals,
        }
    }

    pub fn pulse_id(&self) -> &str {
        self.properties.pulse_id
    }

    pub fn pulse_parameters_id(&self) -> Option<PulseParametersId> {
        self.properties.pulse_parameters_id
    }

    pub fn oscillator_frequency(&self) -> f64 {
        self.properties.oscillator_frequency
    }

    pub fn signals(&self) -> &[SignalUid] {
        &self.signals
    }
}

// ---------------------------------------------------------------------------
// Collect stage — walk the IR and extract deduplicated integration kernels
// ---------------------------------------------------------------------------

fn collect_kernel_properties(event: &PlayAcquire) -> IndexSet<KernelProperties<'_>> {
    let mut properties = IndexSet::with_capacity(event.pulse_defs().len());
    for (pulse_def, pulse_parameters_id) in event
        .pulse_defs()
        .iter()
        .zip(event.id_pulse_params().iter())
    {
        if pulse_def.pulse_type.is_none() {
            // Skip pulses without a type, they do not contribute to integration weights
            // This happens for acquisitions without a kernel, but with a length.
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
    properties: &mut IndexMap<SignalUid, IndexSet<KernelProperties<'a>>>,
    key: SignalUid,
    weights: IndexSet<KernelProperties<'a>>,
) -> Result<()> {
    if let Some(weight_properties) = properties.get_mut(&key) {
        if weight_properties != &weights {
            bail!(
                "Using different integration kernels on a single signal '{}' is unsupported. \
                 They either differ on the pulse definitions or on the oscillator frequency.",
                key.0
            );
        }
    } else {
        properties.insert(key, weights);
    }
    Ok(())
}

fn collect_integration_weights_properties<'a>(
    node: &'a IrNode,
    properties: &mut IndexMap<SignalUid, IndexSet<KernelProperties<'a>>>,
) -> Result<()> {
    match node.data() {
        NodeKind::Acquire(ob) => {
            let weights: IndexSet<KernelProperties<'_>> = collect_kernel_properties(ob);
            update_kernel_properties(properties, ob.signal().uid, weights)?;
        }
        NodeKind::QaEvent(ob) => {
            for acquire in ob.acquires() {
                let weights = collect_kernel_properties(acquire);
                update_kernel_properties(properties, acquire.signal().uid, weights)?;
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
fn collect_integration_kernels<'a>(
    node: &'a IrNode,
    awg: &AwgCore,
) -> Result<Vec<IntegrationKernel<'a>>> {
    let has_integration_signals = awg.signals.iter().any(|s| !s.is_output());
    if !has_integration_signals {
        return Ok(vec![]);
    }
    let mut weight_collection = IndexMap::new();
    collect_integration_weights_properties(node, &mut weight_collection)?;
    // Deduplicate the integration weight properties
    let mut unique_weights: IndexMap<KernelProperties<'_>, Vec<SignalUid>> = IndexMap::new();
    for (signal_uid, weight_properties) in weight_collection {
        // Group weights by pulse definition
        let mut unique_pulse_ids = IndexSet::new();
        for weight in weight_properties {
            if unique_pulse_ids.contains(weight.pulse_id) {
                bail!(
                    "Using different integration kernels on a single signal '{}' is unsupported. \
                     They either differ on the pulse definitions or on the oscillator frequency.",
                    signal_uid.0
                );
            }
            unique_pulse_ids.insert(weight.pulse_id);
            unique_weights.entry(weight).or_default().push(signal_uid);
        }
    }
    Ok(unique_weights
        .into_iter()
        .map(|(properties, signals)| IntegrationKernel::new(signals, properties))
        .collect())
}

// ---------------------------------------------------------------------------
// Assemble stage — derive output-ready data from sampled kernels
// ---------------------------------------------------------------------------

/// Build integration weights sorted by integration channel for output consistency.
fn build_integration_weights(
    kernels: &[SampledIntegrationKernel],
    ctx: &CodeGenContext,
) -> Vec<IntegrationWeight> {
    let mut weights = kernels
        .iter()
        .flat_map(|w| {
            w.signals.iter().map(|signal_uid| {
                ctx.integration_units_for_signal(*signal_uid).map(|units| {
                    let mut sorted_units = units.to_vec();
                    sorted_units.sort();
                    IntegrationWeight {
                        integration_units: sorted_units,
                        basename: w.basename().to_string(),
                        downsampling_factor: w.downsampling_factor.unwrap_or(1),
                    }
                })
            })
        })
        .flatten()
        .collect::<Vec<_>>();
    weights.sort_by(|a, b| a.integration_units.cmp(&b.integration_units));
    weights
}

fn resolve_long_readout_signals(kernels: &[SampledIntegrationKernel]) -> HashSet<SignalUid> {
    kernels
        .iter()
        .filter(|kernel| kernel.downsampling_factor.is_some())
        .flat_map(|kernel| kernel.signals.to_vec())
        .collect()
}

fn create_integration_kernel_waveforms(
    kernel: SampledIntegrationKernel,
) -> Result<(Vec<CodegenWaveform>, WaveformStore)> {
    let mut store = WaveformStore::default();
    let mut waveforms = Vec::with_capacity(kernel.waveforms.len());
    for waveform in kernel.waveforms {
        let cgwf = CodegenWaveform::new(waveform.key.clone(), None, kernel.downsampling_factor);
        store.insert(waveform.key, waveform.samples)?;
        waveforms.push(cgwf);
    }
    Ok((waveforms, store))
}
