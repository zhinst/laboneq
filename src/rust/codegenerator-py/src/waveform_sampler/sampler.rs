// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::ir::SignalUid;
use codegenerator::pulse_map::{PulseInstance, PulseWaveform};
use codegenerator::result::WaveformSignatureString;
use codegenerator_utils::pulse_parameters::PulseParameterDeduplicator;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{ComplexOrFloat, ExternalParameterUid, PulseDef, PulseUid};
use laboneq_error::{LabOneQError, PyErrorWithContext, WithContext, bail};
use laboneq_py_utils::py_export::pulse_def_to_py;
use laboneq_py_utils::py_object_interner::PyObjectInterner;
use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use std::collections::HashMap;
use std::sync::Arc;

use pyo3::types::{PyList, PyTuple};
use pyo3::{IntoPyObjectExt, intern, prelude::*};

use crate::common_types::{DeviceTypePy, SignalTypePy};
use crate::utils::mixer_type_to_py;
use codegenerator::ir::compilation_job::{
    AwgCore, AwgKind, DeviceKind, MixerType, OscillatorKind, Signal, SignalKind,
};
use codegenerator::ir::experiment::{AcquisitionType, PulseParametersId};
use codegenerator::signature::WaveformSignature;
use codegenerator::waveform_sampler::{
    CompressedWaveformPart, IntegrationKernel, SampleBuffer, SampleIntegrationKernels,
    SampleWaveforms, SampledIntegrationKernel, SampledWaveformCollection, SampledWaveformSignature,
    WaveCompression, WaveIdentifier, WaveKey, Waveform, WaveformSamplingCandidate,
};
use codegenerator::{Result, ir::Samples};

use super::pulse_parameters::{PulseParametersPy, pulse_parameters_to_py};
use super::signature::create_waveform_description;

// This is used as a workaround for the SHFQA requiring that for sampled pulses, abs(s) < 1.0 must hold
// to be able to play pulses with an amplitude of 1.0, we scale complex pulses by this factor
const SHFQA_COMPLEX_SAMPLE_SCALING: f64 = 1.0 - 1e-10;

#[pyclass(name = "PlayHold")]
pub struct PlayHoldPy {
    offset: i64,
    length: i64,
}

#[pymethods]
impl PlayHoldPy {
    #[new]
    pub fn new(offset: i64, length: i64) -> Self {
        PlayHoldPy { offset, length }
    }
}

#[pyclass(name = "PlaySamples")]
pub struct PlaySamplesPy {
    offset: Samples,
    length: Samples,
    signature: Py<PyAny>,
}

#[pymethods]
impl PlaySamplesPy {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(offset: Samples, length: Samples, signature: Py<PyAny>) -> Self {
        PlaySamplesPy {
            offset,
            length,
            signature,
        }
    }
}

/// WaveformSamplerPy is a wrapper around Python implementation of waveform sampling.
///
/// The sampler will be used to sample waveforms and compress them into a format that can be played on the target AWG.
///
/// This way we avoid arbitrary Python code execution (pulse definitions) in the Rust code and also avoid
/// handling of `numpy` arrays in Rust at this point.
pub(crate) struct WaveformSamplerPy<'a> {
    acquisition_type: AcquisitionType,
    pulse_parameters: HashMap<PulseParametersId, Py<PulseParametersPy>>,
    pulse_defs: HashMap<String, Py<PyAny>>,
    id_store: &'a NamedIdStore,
    py_object_store: &'a PyObjectInterner<ExternalParameterUid>,
    deduplicator: &'a PulseParameterDeduplicator,
}

struct SamplingInfo {
    sampling_rate: f64,
    rf_signal: bool,
    signal_map: HashMap<SignalUid, Arc<Signal>>,
    mixer_type: Option<MixerType>,
    signal_type: SignalKind,
    device_kind: DeviceKind,
}

impl WaveformSamplerPy<'_> {
    pub(crate) fn new<'a>(
        py: Python,
        pulse_defs: &[PulseDef],
        acquisition_type: AcquisitionType,
        dedup: &'a PulseParameterDeduplicator,
        py_object_store: &'a PyObjectInterner<ExternalParameterUid>,
        id_store: &'a NamedIdStore,
    ) -> WaveformSamplerPy<'a> {
        let pulse_defs = pulse_defs
            .iter()
            .map(|pd| {
                let pd_py = pulse_def_to_py(py, id_store, pd).unwrap();
                (
                    id_store.resolve_unchecked(pd.uid).to_string(),
                    pd_py.into_py_any(py).unwrap(),
                )
            })
            .collect();
        let pulse_parameters: HashMap<PulseParametersId, Py<PulseParametersPy>> = dedup
            .all_parameters()
            .map(|pp| {
                let pp_py = pulse_parameters_to_py(py, pp, id_store, py_object_store);
                (PulseParametersId(pp.id), Py::new(py, pp_py).unwrap())
            })
            .collect();
        WaveformSamplerPy {
            acquisition_type,
            pulse_parameters,
            pulse_defs,
            id_store,
            py_object_store,
            deduplicator: dedup,
        }
    }

    fn make_awg_info(awg: &AwgCore) -> SamplingInfo {
        let sampling_rate = awg.sampling_rate;
        let rf_signal = awg.kind == AwgKind::SINGLE || awg.kind == AwgKind::DOUBLE;
        let signal_map = awg.signals.iter().map(|s| (s.uid, Arc::clone(s))).collect();
        // Filter out integration signals, as they are not supported for waveform sampling
        // and also ensure that all signals are of the same kind and mixer type.
        // This is important for the Python sampler, which expects a single signal type.
        // If there are multiple signal types, we will use the first one as a reference.
        let supported_signals = awg
            .signals
            .iter()
            .filter(|s| s.kind != SignalKind::INTEGRATION)
            .collect::<Vec<_>>();
        let ref_signal = supported_signals
            .first()
            .expect("Internal error: No supported signals found for waveform sampling");
        assert!(
            supported_signals.iter().all(|s| s.kind == ref_signal.kind),
            "{}",
            format!(
                "Internal error: Signal type not unique across waveform playing AWG signals ({supported_signals:?})"
            )
        );
        let mixer_types = supported_signals
            .iter()
            .map(|signal| Self::evaluate_mixer(awg.device_kind(), signal))
            .collect::<Vec<_>>();
        assert!(
            mixer_types
                .iter()
                .all(|mixer_type| mixer_type == &mixer_types[0]),
            "{}",
            format!(
                "Internal error: Mixer type not unique across waveform playing AWG signals {supported_signals:?}"
            )
        );
        SamplingInfo {
            sampling_rate,
            rf_signal,
            signal_map,
            mixer_type: mixer_types.into_iter().next().unwrap(),
            signal_type: ref_signal.kind,
            device_kind: *awg.device_kind(),
        }
    }

    fn evaluate_mixer(device: &DeviceKind, signal: &Signal) -> Option<MixerType> {
        if device == &DeviceKind::UHFQA
            && signal
                .oscillator
                .as_ref()
                .is_some_and(|osc| osc.kind == OscillatorKind::HARDWARE)
        {
            Some(MixerType::UhfqaEnvelope)
        } else if signal.kind == SignalKind::SINGLE {
            None
        } else {
            Some(MixerType::IQ)
        }
    }

    pub(crate) fn pulse_parameters_to_py<'py>(
        &self,
        py: Python<'py>,
        pulse_parameters_id: PulseParametersId,
    ) -> PyResult<Bound<'py, PulseParametersPy>> {
        self.deduplicator
            .resolve(&pulse_parameters_id.0)
            .map(|pp| {
                pulse_parameters_to_py(py, pp, self.id_store, self.py_object_store)
                    .into_pyobject(py)
            })
            .expect("Internal error: Pulse parameters not found")
    }

    pub(crate) fn pulse_def_py(&self, py: Python, pulse_uid: &str) -> Py<PyAny> {
        self.pulse_defs
            .get(pulse_uid)
            .map(|pd| pd.clone_ref(py))
            .expect("Internal error: Pulse definition not found")
    }
}

impl SampleWaveforms for WaveformSamplerPy<'_> {
    type PulseParameters = PulseParametersPy;

    fn supports_waveform_sampling(awg: &AwgCore) -> bool {
        awg.signals
            .iter()
            .any(|s| s.kind != SignalKind::INTEGRATION)
    }

    fn batch_sample_and_compress(
        &self,
        awg: &AwgCore,
        waveforms: &[WaveformSamplingCandidate],
    ) -> Result<SampledWaveformCollection, LabOneQError> {
        let sampling_info = Self::make_awg_info(awg);
        let sampling_rate = sampling_info.sampling_rate;
        let rf_signal = sampling_info.rf_signal;
        let mut sampled_waveforms: SampledWaveformCollection = SampledWaveformCollection::new();

        let ctx = WaveformConversionContext {
            device_kind: sampling_info.device_kind,
            awg_type: awg.kind,
            id_store: self.id_store,
            dedup: self.deduplicator,
        };

        Python::attach(|py| -> Result<(), PyErrorWithContext> {
            let device_type: Bound<'_, DeviceTypePy> =
                DeviceTypePy::from_device_kind(awg.device_kind())
                    .into_pyobject(py)
                    .expect("Internal Error: Failed to convert DeviceType");
            let mixer_type = sampling_info
                .mixer_type
                .map(|mixer| mixer_type_to_py(py, &mixer))
                .transpose()?;
            let signal_type = SignalTypePy::from_signal_kind(&sampling_info.signal_type)
                .into_pyobject(py)
                .expect("Internal Error: Failed to convert SignalType");
            for waveform in waveforms {
                // NOTE: Signals are not relevant for the sampling process, but they are needed
                // for error messages...
                let signals = &waveform
                    .signals
                    .iter()
                    .map(|s| {
                        sampling_info
                            .signal_map
                            .get(s)
                            .expect("Internal error: Signal not found")
                            .as_ref()
                    })
                    .collect::<Vec<_>>();
                // Apply spectroscopy-only sampling path strictly to SHFQA devices.
                // We intentionally avoid a generic QA-device check to exclude UHFQA.
                let is_shfqa_device = matches!(awg.device_kind(), DeviceKind::SHFQA);
                let is_spectroscopy = matches!(
                    self.acquisition_type,
                    AcquisitionType::SPECTROSCOPY_IQ | AcquisitionType::SPECTROSCOPY_PSD
                );
                let signals = signals
                    .iter()
                    .map(|s| self.id_store.resolve_unchecked(s.uid))
                    .collect::<Vec<_>>();
                let output = if is_shfqa_device && is_spectroscopy {
                    sample_only(
                        py,
                        waveform.waveform,
                        &signals,
                        sampling_rate,
                        &device_type,
                        rf_signal,
                        mixer_type.as_ref(),
                        &signal_type,
                        self,
                    )
                } else {
                    sample_and_compress(
                        py,
                        waveform.waveform,
                        &signals,
                        sampling_rate,
                        &device_type,
                        rf_signal,
                        mixer_type.as_ref(),
                        &signal_type,
                        self,
                    )
                };
                let output = output.with_context(|| {
                    format!(
                        "Failed to sample waveform. signal(s): '{}', pulses: '{}'",
                        signals
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                        waveform
                            .waveform
                            .pulses()
                            .map(|pulses| {
                                pulses
                                    .iter()
                                    .filter_map(|p| p.pulse.as_ref().map(|p| p.uid.as_str()))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            })
                            .unwrap_or("".to_string())
                    )
                })?;
                if let Some(output) = output {
                    if output.is_none(py) {
                        continue;
                    }
                    let output = output.bind(py);

                    let signals = &waveform.signals.iter().cloned().collect::<Vec<_>>();
                    let signature_string = &waveform.waveform.signature_string().into();

                    if !output.is_instance_of::<PyList>() {
                        // If the output is not a list, it means that waveform was only sampled
                        let sampled_s = sampled_waveform_to_signature(
                            output
                                .extract()
                                .expect("Internal error: Failed to bind waveform"),
                            signals,
                            signature_string,
                            &ctx,
                        )
                        .expect("Internal error: Failed to convert sampled waveform signature");
                        sampled_waveforms.insert_sampled_signature(waveform.waveform, sampled_s);
                    }
                    if let Ok(compressed_parts) = output.cast::<PyList>() {
                        let compressed_parts = convert_compressed_waveform_parts(
                            compressed_parts,
                            signals,
                            signature_string,
                            &ctx,
                        )
                        .expect("Internal error: Failed to convert compressed parts");
                        sampled_waveforms
                            .insert_compressed_parts(waveform.waveform, compressed_parts);
                    }
                }
            }
            Ok(())
        })?;
        Ok(sampled_waveforms)
    }
}

impl SampleIntegrationKernels for WaveformSamplerPy<'_> {
    type PulseParameters = PulseParametersPy;

    /// Samples integration weights for the given kernels using the Python sampler.
    ///
    /// This function samples the given integration kernels into integration weights.
    fn sample_integration_kernels(
        &self,
        awg: &AwgCore,
        kernels: Vec<IntegrationKernel<'_>>,
    ) -> Result<Vec<SampledIntegrationKernel>> {
        let integration_signals = awg
            .signals
            .iter()
            .filter(|s| s.kind == SignalKind::INTEGRATION)
            .collect::<Vec<_>>();
        let mixer_type = if let Some(ref_signal) = integration_signals.first() {
            Self::evaluate_mixer(awg.device_kind(), ref_signal)
        } else {
            if !kernels.is_empty() {
                bail!("No integration signals found, but kernels were provided",);
            }
            return Ok(vec![]);
        };
        Python::attach(|py| {
            let mixer_type = mixer_type
                .map(|mixer| mixer_type_to_py(py, &mixer))
                .transpose()?;
            let mut bound_weights: Vec<BoundIntegrationWeight<'_>> =
                Vec::with_capacity(kernels.len());
            let sampler = sample_integration_weight_py(py)?;
            for kernel in kernels {
                let params = kernel
                    .pulse_parameters_id()
                    .and_then(|id| self.pulse_parameters.get(&id).map(|p| p.clone_ref(py)));
                let result: Bound<'_, PyAny> = sampler.call(
                    (
                        &self.pulse_defs[kernel.pulse_id()],
                        params,
                        kernel.oscillator_frequency(),
                        kernel
                            .signals()
                            .iter()
                            .map(|s| self.id_store.resolve_unchecked(*s))
                            .collect::<Vec<_>>(),
                        &awg.sampling_rate,
                        &mixer_type,
                    ),
                    None,
                )?;
                let samples_i_q = result
                    .cast::<PyTuple>()
                    .expect("Internal error: Expected a tuple from Python sampler");
                let samples_i = samples_i_q.get_item(0)?;
                let samples_q = samples_i_q.get_item(1)?;
                bound_weights.push(BoundIntegrationWeight::new(
                    samples_i,
                    samples_q,
                    kernel.signals().to_vec(),
                )?);
            }
            let out = create_integration_weights(py, bound_weights, awg.device_kind())?;
            Ok(out)
        })
    }
}

struct WaveformConversionContext<'a> {
    device_kind: DeviceKind,
    awg_type: AwgKind,

    id_store: &'a NamedIdStore,
    dedup: &'a PulseParameterDeduplicator,
}

// --- Expected outputs from the Python sampler when sampling a waveform signature ---

#[derive(FromPyObject)]
struct BoundSampledWaveformSignaturePy<'py> {
    samples: BoundSamplesSignaturePy<'py>,
    pulse_map: HashMap<String, BoundPulseWaveformPy>,

    hold_index_length: Option<(usize, usize)>,
    compressed: bool,
}

#[derive(FromPyObject)]
struct BoundSamplesSignaturePy<'py> {
    samples_i: Bound<'py, PyAny>,
    samples_q: Option<Bound<'py, PyAny>>,
    samples_marker1: Option<Bound<'py, PyAny>>,
    samples_marker2: Option<Bound<'py, PyAny>>,
}

#[derive(FromPyObject)]
struct BoundPulseWaveformPy {
    sampling_rate: f64,
    length_samples: usize,
    iq_modulation: bool,
    #[pyo3(from_py_with = mixer_type_from_py)]
    mixer_type: Option<MixerType>,
    instances: Vec<BoundPulseInstancePy>,
}

fn mixer_type_from_py(value: &Bound<PyAny>) -> PyResult<Option<MixerType>> {
    if value.is_none() {
        Ok(None)
    } else {
        let mixer_type_str: String = value.getattr("name")?.extract()?;
        match mixer_type_str.as_str() {
            "IQ" => Ok(Some(MixerType::IQ)),
            "UHFQA_ENVELOPE" => Ok(Some(MixerType::UhfqaEnvelope)),
            _ => Err(PyTypeError::new_err(format!(
                "Invalid mixer type: {mixer_type_str}"
            ))),
        }
    }
}

#[derive(FromPyObject)]
struct BoundPulseInstancePy {
    offset_samples: usize,
    #[pyo3(from_py_with = complex_or_float_from_py)]
    amplitude: Option<ComplexOrFloat>,
    length: Option<f64>,
    iq_phase: Option<f64>,
    modulation_frequency: Option<f64>,
    channel: Option<usize>,
    needs_conjugate: bool,
    parameters_id: Option<u64>,
    has_marker1: bool,
    has_marker2: bool,
    can_compress: bool,
}

fn complex_or_float_from_py(value: &Bound<PyAny>) -> PyResult<Option<ComplexOrFloat>> {
    if let Ok(f) = value.extract::<Complex64>() {
        Ok(Some(ComplexOrFloat::Complex(f)))
    } else if let Ok(c) = value.extract::<f64>() {
        Ok(Some(ComplexOrFloat::Float(c)))
    } else {
        Err(PyTypeError::new_err("Expected a complex or float value"))
    }
}

fn sampled_waveform_to_signature(
    sampled_signature: BoundSampledWaveformSignaturePy,
    signals: &[SignalUid],
    signature: &WaveformSignatureString,
    ctx: &WaveformConversionContext,
) -> Result<SampledWaveformSignature> {
    let compression = convert_compression(&sampled_signature);
    let pulse_map = convert_pulse_map(
        sampled_signature.pulse_map,
        compression.is_some(),
        ctx.id_store,
        ctx.dedup,
    );
    let waveforms = samples_to_waveform(sampled_signature.samples, signature, ctx)?;

    let wave = SampledWaveformSignature {
        waveforms,
        pulse_map,
        compression,
        signals: signals.to_vec(),
    };
    Ok(wave)
}

fn samples_to_waveform(
    samples: BoundSamplesSignaturePy,
    signature: &WaveformSignatureString,
    ctx: &WaveformConversionContext,
) -> PyResult<Vec<Waveform>> {
    if matches!(ctx.device_kind, DeviceKind::SHFQA) {
        let samples = scale_samples(
            samples
                .samples_i
                .cast::<PyArray1<f64>>()
                .map_err(|e| laboneq_error::laboneq_error!("{e}"))?
                .readonly(),
            samples
                .samples_q
                .expect("Expected Q-component for SHFQA waveforms")
                .cast::<PyArray1<f64>>()
                .map_err(|e| laboneq_error::laboneq_error!("{e}"))?
                .readonly(),
            SHFQA_COMPLEX_SAMPLE_SCALING,
        )?;
        let wave = Waveform {
            key: WaveKey {
                signature: Arc::clone(signature),
                identifier: None,
            },
            samples,
        };
        return Ok(vec![wave]);
    }

    let mut waveforms = Vec::with_capacity(2); // Expect most of the time to have I and Q components.

    let single = matches!(ctx.awg_type, AwgKind::SINGLE);

    let wave_i = Waveform {
        key: WaveKey {
            signature: Arc::clone(signature),
            identifier: if !single {
                Some(WaveIdentifier::I)
            } else {
                None
            },
        },
        samples: SampleBuffer::Float64(samples.samples_i.extract::<Vec<f64>>()?),
    };
    waveforms.push(wave_i);

    if !single && let Some(samples_q) = samples.samples_q {
        let wave_q = Waveform {
            key: WaveKey {
                signature: Arc::clone(signature),
                identifier: Some(WaveIdentifier::Q),
            },
            samples: SampleBuffer::Float64(samples_q.extract::<Vec<f64>>()?),
        };
        waveforms.push(wave_q);
    }

    if let Some(samples_marker1) = samples.samples_marker1 {
        let wave_marker1 = Waveform {
            key: WaveKey {
                signature: Arc::clone(signature),
                identifier: Some(WaveIdentifier::M1),
            },
            samples: SampleBuffer::U8(samples_marker1.extract::<Vec<u8>>()?),
        };
        waveforms.push(wave_marker1);
    }

    if let Some(samples_marker2) = samples.samples_marker2 {
        let wave_marker2 = Waveform {
            key: WaveKey {
                signature: Arc::clone(signature),
                identifier: Some(WaveIdentifier::M2),
            },
            samples: SampleBuffer::U8(samples_marker2.extract::<Vec<u8>>()?),
        };
        waveforms.push(wave_marker2);
    }
    Ok(waveforms)
}

fn convert_pulse_map(
    pulse_map: HashMap<String, BoundPulseWaveformPy>,
    compressed: bool,
    id_store: &NamedIdStore,
    dedup: &PulseParameterDeduplicator,
) -> HashMap<PulseUid, PulseWaveform> {
    pulse_map
        .into_iter()
        .map(|(name, pulse)| {
            let pulse_uid = id_store
                .get(name)
                .expect("Internal error: Failed to resolve pulse UID");

            let pulse_waveform = PulseWaveform {
                sampling_rate: pulse.sampling_rate,
                length_samples: pulse.length_samples,
                iq_modulation: pulse.iq_modulation,
                mixer_type: pulse.mixer_type,
                compressed,
                instances: pulse
                    .instances
                    .into_iter()
                    .map(|instance| {
                        let parameters = instance.parameters_id.and_then(|id| dedup.resolve(&(id)));
                        let op_parameters = parameters.map(|p| p.play_parameters.clone());
                        let pulse_parameters = parameters.map(|p| p.pulse_parameters.clone());

                        PulseInstance {
                            offset_samples: instance.offset_samples,
                            amplitude: instance.amplitude,
                            length: instance.length,
                            iq_phase: instance.iq_phase,
                            modulation_frequency: instance.modulation_frequency,
                            channel: instance.channel,
                            needs_conjugate: instance.needs_conjugate,
                            pulse_parameters: pulse_parameters.unwrap_or_default(),
                            parameters: op_parameters.unwrap_or_default(),
                            has_marker1: instance.has_marker1,
                            has_marker2: instance.has_marker2,
                            can_compress: instance.can_compress,
                        }
                    })
                    .collect(),
            };
            (pulse_uid.into(), pulse_waveform)
        })
        .collect()
}

fn convert_compression(waveform: &BoundSampledWaveformSignaturePy) -> Option<WaveCompression> {
    if let Some((hold_index, hold_length)) = waveform.hold_index_length {
        Some(WaveCompression::HoldWave {
            start_index: hold_index,
            length: hold_length,
        })
    } else if waveform.compressed {
        Some(WaveCompression::Compressed)
    } else {
        None
    }
}

/// Converts a list of compressed waveform parts from Python to Rust representation.
fn convert_compressed_waveform_parts(
    compressed_parts: &Bound<'_, PyList>,
    signals: &[SignalUid],
    signature: &WaveformSignatureString,
    ctx: &WaveformConversionContext,
) -> Result<Vec<CompressedWaveformPart>> {
    let py = compressed_parts.py();
    let mut out_parts = Vec::with_capacity(compressed_parts.len());
    for result in compressed_parts.iter() {
        if let Ok(ps) = result.cast::<PlaySamplesPy>() {
            let ps = ps.borrow();
            let signature = sampled_waveform_to_signature(
                ps.signature
                    .bind(py)
                    .extract()
                    .expect("Internal error: Failed to bind waveform"),
                signals,
                signature,
                ctx,
            )
            .expect("Internal error: Failed to convert sampled waveform signature");
            let obj = CompressedWaveformPart::PlaySamples {
                offset: ps.offset,
                waveform: WaveformSignature::Samples {
                    length: ps.length,
                    samples_id: signature.samples_uid(),
                },
                signature,
            };
            out_parts.push(obj);
        } else if let Ok(ph) = result.cast::<PlayHoldPy>() {
            let ph = ph.borrow();
            let obj = CompressedWaveformPart::PlayHold {
                offset: ph.offset,
                length: ph.length,
            };
            out_parts.push(obj);
        }
    }
    Ok(out_parts)
}

/// Whether the waveform should be sampled.
fn should_sample_waveform(waveform: &WaveformSignature) -> bool {
    if let WaveformSignature::Pulses { pulses, .. } = waveform {
        return pulses.iter().any(|pulse| pulse.pulse.is_some());
    }
    false
}

/// A helper function to sample only (no compression) using the Python sampler.
#[allow(clippy::too_many_arguments)]
fn sample_only(
    py: Python,
    waveform: &WaveformSignature,
    signals: &[&str],
    sampling_rate: f64,
    device_type: &Bound<'_, DeviceTypePy>,
    rf_signal: bool,
    mixer_type: Option<&Bound<'_, PyAny>>,
    signal_type: &Bound<'_, SignalTypePy>,
    sampler: &WaveformSamplerPy<'_>,
) -> Result<Option<Py<PyAny>>, PyErrorWithContext> {
    if !should_sample_waveform(waveform) {
        return Ok(None);
    }
    let waveform_desc = create_waveform_description(py, waveform, sampler);
    let sample_and_compress_py_func = sample_waveform_py(py)?;
    let sampled_signature = sample_and_compress_py_func.call(
        (
            signals,
            waveform_desc,
            sampling_rate,
            signal_type,
            device_type,
            mixer_type,
            rf_signal,
        ),
        None,
    )?;
    if sampled_signature.is_none() {
        return Ok(None);
    }
    Ok(Some(sampled_signature.unbind()))
}

/// A helper function to sample and compress a waveform using the Python sampler.
#[allow(clippy::too_many_arguments)]
fn sample_and_compress(
    py: Python,
    waveform: &WaveformSignature,
    signals: &[&str],
    sampling_rate: f64,
    device_type: &Bound<'_, DeviceTypePy>,
    rf_signal: bool,
    mixer_type: Option<&Bound<'_, PyAny>>,
    signal_type: &Bound<'_, SignalTypePy>,
    sampler: &WaveformSamplerPy<'_>,
) -> Result<Option<Py<PyAny>>, PyErrorWithContext> {
    if !should_sample_waveform(waveform) {
        // If the waveform does not need to be sampled, skip the sampling process
        return Ok(None);
    }
    let waveform_desc = create_waveform_description(py, waveform, sampler);
    let sample_and_compress_py_func = sample_and_compress_py(py)?;
    let sampled_signature = sample_and_compress_py_func.call(
        (
            waveform_desc,
            signals,
            sampling_rate,
            signal_type,
            device_type,
            mixer_type,
            rf_signal,
        ),
        None,
    )?;
    if sampled_signature.is_none() {
        // If the waveform does not need to be sampled, skip the sampling process
        return Ok(None);
    }
    Ok(Some(sampled_signature.unbind()))
}

/// Convenience wrapper to interact with GIL bound Numpy array.
///
/// We use this wrapper to avoid dealing with Numpy arrays directly in Rust,
/// as we are not currently interested in the actual data, but rather
/// just the length and the ability to convert it to bytes.
struct NumpyArray<'py> {
    arr: Bound<'py, PyAny>,
}

impl<'py> NumpyArray<'py> {
    fn new(arr: Bound<'py, PyAny>) -> Self {
        NumpyArray { arr }
    }

    fn len(&self) -> PyResult<usize> {
        self.arr.len()
    }

    fn tobytes(&self) -> PyResult<Bound<'_, PyAny>> {
        let c = self.arr.getattr("tobytes")?;
        c.call0()
    }
}

/// Calculates a name for the waveform based on the samples.
fn calculate_wave_name(
    samples_i: &Bound<'_, PyAny>,
    samples_q: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let py = samples_i.py();
    let hashlib_sha1 =
        PyModule::import(py, intern!(py, "hashlib"))?.getattr(intern!(py, "sha1"))?;
    let digest = hashlib_sha1.call1((samples_i.add(samples_q)?,))?;
    let hex = digest
        .getattr(intern!(py, "hexdigest"))?
        .call0()?
        .extract::<String>()?
        .chars()
        .take(32)
        .collect::<String>();
    Ok(format!("kernel_{hex}"))
}

/// GIL bound integration weight.
struct BoundIntegrationWeight<'py> {
    samples_i: NumpyArray<'py>,
    samples_q: NumpyArray<'py>,
    signals: Vec<SignalUid>,
}

impl<'py> BoundIntegrationWeight<'py> {
    fn new(
        samples_i: Bound<'py, PyAny>,
        samples_q: Bound<'py, PyAny>,
        signals: Vec<SignalUid>,
    ) -> PyResult<Self> {
        let samples_i = NumpyArray { arr: samples_i };
        let samples_q = NumpyArray { arr: samples_q };
        Ok(BoundIntegrationWeight {
            samples_i,
            samples_q,
            signals,
        })
    }
}

/// Creates GIL unbound integration weights from the bound weights.
///
/// This function evaluates the common downsampling factor for the integration weights
/// and down-samples the samples if necessary.
fn create_integration_weights(
    py: Python,
    bound_weights: Vec<BoundIntegrationWeight<'_>>,
    device: &DeviceKind,
) -> Result<Vec<SampledIntegrationKernel>> {
    let common_downsampling_factor = eval_common_downsampling_factor(&bound_weights, device)?;
    let mut integration_weights: Vec<SampledIntegrationKernel> =
        Vec::with_capacity(bound_weights.len());

    for weight in bound_weights.into_iter() {
        let basename = Arc::new(calculate_wave_name(
            &weight.samples_i.tobytes()?,
            &weight.samples_q.tobytes()?,
        )?);
        let (samples_i, samples_q) = if let Some(downsampling_factor) = common_downsampling_factor {
            downsample_samples(
                py,
                weight.samples_i,
                weight.samples_q,
                downsampling_factor as usize,
            )?
        } else {
            (weight.samples_i, weight.samples_q)
        };
        if matches!(device, DeviceKind::SHFQA) {
            let samples = scale_samples(
                samples_i
                    .arr
                    .cast::<PyArray1<f64>>()
                    .map_err(|e| laboneq_error::laboneq_error!("{e}"))?
                    .readonly(),
                samples_q
                    .arr
                    .cast::<PyArray1<f64>>()
                    .map_err(|e| laboneq_error::laboneq_error!("{e}"))?
                    .readonly(),
                SHFQA_COMPLEX_SAMPLE_SCALING,
            )?;
            let key = WaveKey {
                signature: Arc::clone(&basename),
                identifier: None,
            };
            let waveform = Waveform {
                key,
                samples: samples.clone(),
            };
            let integration_weight = SampledIntegrationKernel::new(
                weight.signals,
                common_downsampling_factor,
                vec![waveform],
            );
            integration_weights.push(integration_weight);
        } else {
            let wave_i = Waveform {
                key: WaveKey {
                    signature: Arc::clone(&basename),
                    identifier: Some(WaveIdentifier::I),
                },
                samples: SampleBuffer::Float64(samples_i.arr.extract::<Vec<f64>>()?),
            };
            let wave_q = Waveform {
                key: WaveKey {
                    signature: Arc::clone(&basename),
                    identifier: Some(WaveIdentifier::Q),
                },
                samples: SampleBuffer::Float64(samples_q.arr.extract::<Vec<f64>>()?),
            };
            let integration_weight =
                SampledIntegrationKernel::new(weight.signals.clone(), None, vec![wave_i, wave_q]);
            integration_weights.push(integration_weight);
        }
    }
    Ok(integration_weights)
}

fn scale_samples<'py>(
    samples_i: PyReadonlyArray1<'py, f64>,
    samples_q: PyReadonlyArray1<'py, f64>,
    scaling_factor: f64,
) -> PyResult<SampleBuffer> {
    let i = samples_i.as_array();
    let q = samples_q.as_array();
    let combined: Vec<Complex64> = i
        .iter()
        .zip(q.iter())
        .map(|(&i_val, &q_val)| {
            Complex64::new(i_val * scaling_factor + 0.0, -q_val * scaling_factor + 0.0)
        }) // Convert -0.0 to +0.0
        .collect();
    Ok(SampleBuffer::Complex64(combined))
}

fn downsample_samples<'py>(
    py: Python<'py>,
    samples_i: NumpyArray<'py>,
    samples_q: NumpyArray<'py>,
    downsampling_factor: usize,
) -> PyResult<(NumpyArray<'py>, NumpyArray<'py>)> {
    let scipy_resample =
        PyModule::import(py, intern!(py, "scipy.signal"))?.getattr(intern!(py, "resample"))?;
    let i_resampled = NumpyArray::new(scipy_resample.call(
        (&samples_i.arr, samples_i.len()? / downsampling_factor),
        None,
    )?);
    let q_resampled = NumpyArray::new(scipy_resample.call(
        (&samples_q.arr, samples_q.len()? / downsampling_factor),
        None,
    )?);
    Ok((i_resampled, q_resampled))
}

/// Evaluates the common downsampling factor for all the integration weights on a single AWG.
fn eval_common_downsampling_factor(
    weights: &Vec<BoundIntegrationWeight<'_>>,
    device: &DeviceKind,
) -> Result<Option<u8>> {
    // The maximum length of integration weights for SHFQA, UHFQA.
    const INTEGRATION_WEIGHT_MAX_LENGTH: usize = 4096;
    // The maximum downsampling factor for SHFQA with LRT option.
    // TODO: Get from device traits and take the option into account as well.
    const DOWNSAMPLING_FACTOR_SHFQA_MAX: usize = 16;
    let mut common_downsampling_factor: usize = 0;
    for weight in weights.iter() {
        let weight_length = weight.samples_i.len()?;
        if device == &DeviceKind::SHFQA && weight_length > INTEGRATION_WEIGHT_MAX_LENGTH {
            let downsampling_factor = weight_length.div_ceil(INTEGRATION_WEIGHT_MAX_LENGTH);
            if downsampling_factor > DOWNSAMPLING_FACTOR_SHFQA_MAX {
                bail!(
                    "Integration weight length ({}) exceeds the maximum supported by SHFQA ({})",
                    weight_length,
                    INTEGRATION_WEIGHT_MAX_LENGTH * DOWNSAMPLING_FACTOR_SHFQA_MAX
                );
            }
            common_downsampling_factor = common_downsampling_factor.max(downsampling_factor);
        }
    }
    if common_downsampling_factor < 2 {
        return Ok(None);
    }
    Ok(Some(common_downsampling_factor as u8)) // Safe to convert as we check the maximum downsampling factor for SHFQA above, which is 16 and fits into u8.
}

fn sample_integration_weight_py<'a>(py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.seqc.waveform_sampler"))?;
    m.getattr(intern!(py, "sample_integration_weight"))
}

fn sample_and_compress_py<'a>(py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.seqc.waveform_sampler"))?;
    m.getattr(intern!(py, "sample_and_compress"))
}

fn sample_waveform_py<'a>(py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.seqc.waveform_sampler"))?;
    m.getattr(intern!(py, "sample_waveform"))
}
