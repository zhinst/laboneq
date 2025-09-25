// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::types::{PyList, PyTuple};
use pyo3::{intern, prelude::*};

use crate::common_types::{DeviceTypePy, MixerTypePy, SignalTypePy};
use crate::pulse_parameters::{PulseParameters, PulseParametersPy};
use crate::signature::WaveformSignaturePy;
use codegenerator::ir::compilation_job::{
    AwgCore, AwgKind, DeviceKind, MixerType, Signal, SignalKind,
};
use codegenerator::ir::experiment::PulseParametersId;
use codegenerator::signature::{SamplesSignatureID, Uid, WaveformSignature};
use codegenerator::waveform_sampler::{
    CompressedWaveformPart, IntegrationKernel, SampleWaveforms, SampledWaveformCollection,
    SampledWaveformSignature, WaveformSamplingCandidate,
};
use codegenerator::{Error, Result, Samples};
/// Represents a sampled waveform signature that is returned by the Python sampler.
///
/// Some information is read and stored from the Python signature to avoid the
/// need to access the Python object every time we need to check for markers.
#[derive(Debug)]
pub struct SampledWaveformSignaturePy {
    /// The Python signature of the sampled waveform (i.e. SampledWaveformSignature)
    pub signature: Arc<Py<PyAny>>,
    /// Whether the sampled waveform has marker 1.
    pub has_markers1: bool,
    /// Whether the sampled waveform has marker 2.
    pub has_markers2: bool,
}

impl SampledWaveformSignature for SampledWaveformSignaturePy {
    type Inner = Arc<Py<PyAny>>;

    fn has_marker1(&self) -> bool {
        self.has_markers1
    }

    fn has_marker2(&self) -> bool {
        self.has_markers2
    }

    fn signature(&self) -> Arc<Py<PyAny>> {
        Arc::clone(&self.signature)
    }
}

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
    uid: Uid,
    label: String,
    has_i: bool,
    has_q: bool,
    has_marker1: bool,
    has_marker2: bool,
    signature: Py<PyAny>,
}

#[pymethods]
impl PlaySamplesPy {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        offset: Samples,
        length: Samples,
        uid: Uid,
        label: String,
        has_i: bool,
        has_q: bool,
        has_marker1: bool,
        has_marker2: bool,
        signature: Py<PyAny>,
    ) -> Self {
        PlaySamplesPy {
            offset,
            length,
            uid,
            label,
            has_i,
            has_q,
            has_marker1,
            has_marker2,
            signature,
        }
    }
}

#[derive(Debug)]
pub struct IntegrationWeight<'a> {
    pub signals: HashSet<&'a str>,
    pub samples_i: Py<PyAny>,
    pub samples_q: Py<PyAny>,
    pub downsampling_factor: Option<usize>,
    pub basename: String,
}
/// WaveformSamplerPy is a wrapper around Python implementation of waveform sampling.
///
/// The sampler will be used to sample waveforms and compress them into a format that can be played on the target AWG.
///
/// This way we avoid arbitrary Python code execution (pulse definitions) in the Rust code and also avoid
/// handling of `numpy` arrays in Rust at this point.
pub struct WaveformSamplerPy<'a> {
    sampler: &'a Py<PyAny>,
    sampling_rate: f64,
    device_kind: &'a DeviceKind,
    rf_signal: bool,
    signal_map: HashMap<&'a str, Arc<Signal>>,
    mixer_type: Option<&'a MixerType>,
    signal_kind: &'a SignalKind,
    pulse_parameters: &'a HashMap<PulseParametersId, PulseParameters>,
}

impl<'a> WaveformSamplerPy<'a> {
    pub fn supports_waveform_sampling(awg: &'a AwgCore) -> bool {
        awg.signals
            .iter()
            .any(|s| s.kind != SignalKind::INTEGRATION)
    }

    pub fn new(
        sampler: &'a Py<PyAny>,
        awg: &'a AwgCore,
        pulse_parameters: &'a HashMap<PulseParametersId, PulseParameters>,
    ) -> Self {
        let sampling_rate = awg.sampling_rate;
        let rf_signal = awg.kind == AwgKind::SINGLE || awg.kind == AwgKind::DOUBLE;
        let signal_map = awg
            .signals
            .iter()
            .map(|s| (s.uid.as_str(), Arc::clone(s)))
            .collect();
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
        assert!(
            supported_signals
                .iter()
                .all(|s| s.mixer_type == ref_signal.mixer_type),
            "{}",
            format!(
                "Internal error: Mixer type not unique across waveform playing AWG signals {supported_signals:?}"
            )
        );
        let mixer_type = ref_signal.mixer_type.as_ref();
        let signal_kind = &ref_signal.kind;
        WaveformSamplerPy {
            sampler,
            sampling_rate,
            device_kind: awg.device_kind(),
            rf_signal,
            signal_map,
            mixer_type,
            signal_kind,
            pulse_parameters,
        }
    }
}

impl SampleWaveforms for WaveformSamplerPy<'_> {
    type Signature = SampledWaveformSignaturePy;

    fn batch_sample_and_compress(
        &self,
        waveforms: &[WaveformSamplingCandidate],
    ) -> Result<SampledWaveformCollection<SampledWaveformSignaturePy>> {
        let sampling_rate = self.sampling_rate;
        let rf_signal = self.rf_signal;
        let mut sampled_waveforms: SampledWaveformCollection<SampledWaveformSignaturePy> =
            SampledWaveformCollection::new();
        Python::attach(|py| -> Result<()> {
            let pulse_parameters: HashMap<u64, Py<PulseParametersPy>> = self
                .pulse_parameters
                .iter()
                .map(
                    |(id, params)| -> std::result::Result<(u64, Py<PulseParametersPy>), Error> {
                        Ok((
                            id.0,
                            Py::new(
                                py,
                                PulseParametersPy {
                                    parameters: params.clone(),
                                },
                            )
                            .map_err(Error::with_error)?,
                        ))
                    },
                )
                .collect::<std::result::Result<HashMap<_, _>, Error>>()?;
            let sampler = self.sampler.bind(py);
            let device_type: Bound<'_, DeviceTypePy> =
                DeviceTypePy::from_device_kind(self.device_kind)
                    .into_pyobject(py)
                    .expect("Internal Error: Failed to convert DeviceType");
            let mixer_type = self.mixer_type.map(|mixer| {
                MixerTypePy::from_mixer_type(mixer)
                    .into_pyobject(py)
                    .expect("Internal Error: Failed to convert MixerType")
            });
            let signal_type = SignalTypePy::from_signal_kind(self.signal_kind)
                .into_pyobject(py)
                .expect("Internal Error: Failed to convert SignalType");
            for waveform in waveforms {
                // NOTE: Signals are not relevant for the sampling process, but they are needed
                // for error messages...
                let signals = &waveform
                    .signals
                    .iter()
                    .map(|s| {
                        self.signal_map
                            .get(s)
                            .expect("Internal error: Signal not found")
                            .as_ref()
                    })
                    .collect::<Vec<_>>();
                let output = sample_and_compress(
                    py,
                    sampler,
                    waveform.waveform,
                    signals,
                    sampling_rate,
                    &device_type,
                    rf_signal,
                    mixer_type.as_ref(),
                    &signal_type,
                    &pulse_parameters,
                )
                .with_context(|| {
                    format!(
                        "Failed to sample waveform. signal(s): '{}', pulses: '{}'",
                        signals
                            .iter()
                            .map(|s| s.uid.as_str())
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
                    if !output.is_instance_of::<PyList>() {
                        // If the output is not a list, it means that waveform was only sampled
                        let sampled_signature = convert_sampled_waveform_signature(output)
                            .expect("Internal error: Failed to create SampledWaveformSignature");
                        sampled_waveforms
                            .insert_sampled_signature(waveform.waveform, sampled_signature);
                    }
                    if let Ok(compressed_parts) = output.downcast::<PyList>() {
                        let compressed_parts = convert_compressed_waveform_parts(compressed_parts)
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

// Code to convert a Python sampled waveform signature to a Rust representation.
fn convert_sampled_waveform_signature(
    signature: &Bound<'_, PyAny>,
) -> PyResult<SampledWaveformSignaturePy> {
    let py = signature.py();
    let m1 = signature
        .getattr(intern!(py, "samples_marker1"))?
        .extract::<Option<Py<PyAny>>>()?
        .is_some();
    let m2 = signature
        .getattr(intern!(py, "samples_marker2"))?
        .extract::<Option<Py<PyAny>>>()?
        .is_some();
    let signature = SampledWaveformSignaturePy {
        signature: Arc::new(signature.into_pyobject(py).unwrap().into()),
        has_markers1: m1,
        has_markers2: m2,
    };
    Ok(signature)
}

/// Converts a list of compressed waveform parts from Python to Rust representation.
fn convert_compressed_waveform_parts(
    compressed_parts: &Bound<'_, PyList>,
) -> Result<Vec<CompressedWaveformPart<SampledWaveformSignaturePy>>> {
    let py = compressed_parts.py();
    let mut out_parts = Vec::with_capacity(compressed_parts.len());
    for result in compressed_parts.iter() {
        if let Ok(ps) = result.downcast::<PlaySamplesPy>() {
            let ps = ps.borrow();
            let samples_id = SamplesSignatureID {
                uid: ps.uid,
                label: ps.label.clone(),
                has_i: ps.has_i,
                has_q: ps.has_q,
                has_marker1: ps.has_marker1,
                has_marker2: ps.has_marker2,
            };
            let signature = convert_sampled_waveform_signature(ps.signature.bind(py))
                .expect("Internal error: Failed to create SampledWaveformSignature");
            let obj = CompressedWaveformPart::PlaySamples {
                offset: ps.offset,
                waveform: WaveformSignature::Samples {
                    length: ps.length,
                    samples_id,
                },
                signature,
            };
            out_parts.push(obj);
        } else if let Ok(ph) = result.downcast::<PlayHoldPy>() {
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

/// A helper function to sample and compress a waveform using the Python sampler.
#[allow(clippy::too_many_arguments)]
fn sample_and_compress(
    py: Python,
    sampler: &Bound<'_, PyAny>,
    waveform: &WaveformSignature,
    signals: &[&Signal],
    sampling_rate: f64,
    device_type: &Bound<'_, DeviceTypePy>,
    rf_signal: bool,
    mixer_type: Option<&Bound<'_, MixerTypePy>>,
    signal_type: &Bound<'_, SignalTypePy>,
    pulse_parameters: &HashMap<u64, Py<PulseParametersPy>>,
) -> Result<Option<Py<PyAny>>> {
    if !should_sample_waveform(waveform) {
        // If the waveform does not need to be sampled, skip the sampling process
        return Ok(None);
    }
    let signals_uid = signals.iter().map(|s| &s.uid).collect::<Vec<_>>();
    let waveform_py: WaveformSignaturePy = WaveformSignaturePy::new(waveform.clone());
    let sampled_signature = sampler
        .call_method(
            intern!(py, "sample_and_compress"),
            (
                waveform_py,
                &signals_uid,
                sampling_rate,
                signal_type,
                device_type,
                mixer_type,
                rf_signal,
                pulse_parameters,
            ),
            None,
        )
        .map_err(Error::with_error)?;
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
struct BoundIntegrationWeight<'a, 'py> {
    samples_i: NumpyArray<'py>,
    samples_q: NumpyArray<'py>,
    basename: String,
    signals: Vec<&'a str>,
}

impl<'a, 'py> BoundIntegrationWeight<'a, 'py> {
    fn new(
        samples_i: Bound<'py, PyAny>,
        samples_q: Bound<'py, PyAny>,
        signals: Vec<&'a str>,
    ) -> PyResult<Self> {
        let samples_i = NumpyArray { arr: samples_i };
        let samples_q = NumpyArray { arr: samples_q };
        let basename = calculate_wave_name(&samples_i.tobytes()?, &samples_q.tobytes()?)?;
        Ok(BoundIntegrationWeight {
            samples_i,
            samples_q,
            basename,
            signals: signals.to_vec(),
        })
    }
}

/// Samples integration weights for the given kernels using the Python sampler.
///
/// This function samples the given integration kernels into integration weights.
pub fn batch_calculate_integration_weights<'a>(
    awg: &'a AwgCore,
    sampler: &Py<PyAny>,
    kernels: Vec<IntegrationKernel<'_>>,
    pulse_parameters: &'a HashMap<PulseParametersId, PulseParameters>,
) -> Result<Vec<IntegrationWeight<'a>>> {
    let integration_signals = awg
        .signals
        .iter()
        .filter(|s| s.kind == SignalKind::INTEGRATION)
        .collect::<Vec<_>>();
    let mixer_type = if let Some(ref_signal) = integration_signals.first() {
        ref_signal.mixer_type.as_ref()
    } else {
        if !kernels.is_empty() {
            return Err(Error::new(
                "No integration signals found, but kernels were provided",
            ));
        }
        return Ok(vec![]);
    };
    let signal_map: HashMap<&str, &Arc<Signal>> = integration_signals
        .iter()
        .map(|s| (s.uid.as_str(), *s))
        .collect();
    Python::attach(|py| -> Result<Vec<IntegrationWeight<'a>>> {
        let mixer_type = mixer_type.map(|mixer| {
            MixerTypePy::from_mixer_type(mixer)
                .into_pyobject(py)
                .expect("Internal Error: Failed to convert MixerType")
        });
        let mut bound_weights: Vec<BoundIntegrationWeight<'a, '_>> =
            Vec::with_capacity(kernels.len());
        let sampler = sampler.bind(py);
        for kernel in kernels {
            let params = kernel
                .pulse_parameters_id()
                .and_then(|id| pulse_parameters.get(&id).map(|p| p.parameters()));
            let result: Bound<'_, PyAny> = sampler
                .call_method(
                    intern!(py, "sample_integration_weight"),
                    (
                        kernel.pulse_id(),
                        params,
                        kernel.oscillator_frequency(),
                        kernel.signals().iter().collect::<Vec<_>>(),
                        &awg.sampling_rate,
                        &mixer_type,
                    ),
                    None,
                )
                .map_err(Error::with_error)?;
            let samples_i_q = result
                .downcast::<PyTuple>()
                .expect("Internal error: Expected a tuple from Python sampler");
            let samples_i = samples_i_q.get_item(0).map_err(Error::with_error)?;
            let samples_q = samples_i_q.get_item(1).map_err(Error::with_error)?;
            bound_weights.push(
                BoundIntegrationWeight::new(
                    samples_i,
                    samples_q,
                    kernel
                        .signals()
                        .iter()
                        .map(|s| {
                            signal_map
                                .get(s)
                                .expect(
                                    "Internal error: Integration weight on non-integration signal",
                                )
                                .uid
                                .as_str()
                        })
                        .collect(),
                )
                .map_err(Error::with_error)?,
            );
        }
        let out = create_integration_weights(py, bound_weights, awg.device_kind())?;
        Ok(out)
    })
}

/// Creates GIL unbound integration weights from the bound weights.
///
/// This function evaluates the common downsampling factor for the integration weights
/// and down-samples the samples if necessary.
fn create_integration_weights<'a>(
    py: Python,
    bound_weights: Vec<BoundIntegrationWeight<'a, '_>>,
    device: &DeviceKind,
) -> Result<Vec<IntegrationWeight<'a>>> {
    let common_downsampling_factor = eval_common_downsampling_factor(&bound_weights, device)?;
    let mut integration_weights: Vec<IntegrationWeight<'a>> =
        Vec::with_capacity(bound_weights.len());
    if let Some(downsampling_factor) = common_downsampling_factor {
        for weight in bound_weights.into_iter() {
            let (samples_i, samples_q) =
                downsample_samples(py, weight.samples_i, weight.samples_q, downsampling_factor)
                    .map_err(Error::with_error)?;
            let integration_weight = IntegrationWeight {
                signals: weight.signals.iter().cloned().collect(),
                samples_i: samples_i.arr.unbind(),
                samples_q: samples_q.arr.unbind(),
                downsampling_factor: Some(downsampling_factor),
                basename: weight.basename,
            };
            integration_weights.push(integration_weight);
        }
    } else {
        for weight in bound_weights.into_iter() {
            let integration_weight = IntegrationWeight {
                signals: weight.signals.iter().cloned().collect(),
                samples_i: weight.samples_i.arr.unbind(),
                samples_q: weight.samples_q.arr.unbind(),
                downsampling_factor: None,
                basename: weight.basename,
            };
            integration_weights.push(integration_weight);
        }
    }
    Ok(integration_weights)
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
    weights: &Vec<BoundIntegrationWeight<'_, '_>>,
    device: &DeviceKind,
) -> Result<Option<usize>> {
    // The maximum length of integration weights for SHFQA, UHFQA.
    const INTEGRATION_WEIGHT_MAX_LENGTH: usize = 4096;
    // The maximum downsampling factor for SHFQA with LRT option.
    // TODO: Get from device traits and take the option into account as well.
    const DOWNSAMPLING_FACTOR_SHFQA_MAX: usize = 16;
    let mut common_downsampling_factor: usize = 0;
    for weight in weights.iter() {
        let weight_length = weight.samples_i.len().map_err(Error::with_error)?;
        if device == &DeviceKind::SHFQA && weight_length > INTEGRATION_WEIGHT_MAX_LENGTH {
            let downsampling_factor = weight_length.div_ceil(INTEGRATION_WEIGHT_MAX_LENGTH);
            if downsampling_factor > DOWNSAMPLING_FACTOR_SHFQA_MAX {
                let msg = format!(
                    "Integration weight length ({}) exceeds the maximum supported by SHFQA ({})",
                    weight_length,
                    INTEGRATION_WEIGHT_MAX_LENGTH * DOWNSAMPLING_FACTOR_SHFQA_MAX
                );
                return Err(Error::new(&msg));
            }
            common_downsampling_factor = common_downsampling_factor.max(downsampling_factor);
        }
    }
    if common_downsampling_factor < 2 {
        return Ok(None);
    }
    Ok(Some(common_downsampling_factor))
}
