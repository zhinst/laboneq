// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use pyo3::types::PyList;
use pyo3::{intern, prelude::*};

use crate::common_types::{DeviceTypePy, MixerTypePy, SignalTypePy};
use crate::signature::WaveformSignaturePy;
use codegenerator::ir::compilation_job::{
    AwgCore, AwgKind, DeviceKind, MixerType, Signal, SignalKind,
};
use codegenerator::signature::{SamplesSignatureID, WaveformSignature};
use codegenerator::waveform_sampler::{
    CompressedWaveformPart, SampleWaveforms, SampledWaveformCollection, SampledWaveformSignature,
    WaveformSamplingCandidate,
};
use codegenerator::{Error, Result};

/// Represents a sampled waveform signature that is returned by the Python sampler.
///
/// Some information is read and stored from the Python signature to avoid the
/// need to access the Python object every time we need to check for markers.
#[derive(Debug)]
pub struct SampledWaveformSignaturePy {
    /// The Python signature of the sampled waveform (i.e. SampledWaveformSignature)
    pub signature: Arc<PyObject>,
    /// Whether the sampled waveform has marker 1.
    pub has_markers1: bool,
    /// Whether the sampled waveform has marker 2.
    pub has_markers2: bool,
}

impl SampledWaveformSignature for SampledWaveformSignaturePy {
    type Inner = Arc<PyObject>;

    fn has_marker1(&self) -> bool {
        self.has_markers1
    }

    fn has_marker2(&self) -> bool {
        self.has_markers2
    }

    fn signature(&self) -> Arc<PyObject> {
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
    offset: i64,
    length: i64,
    uid: u64,
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
        offset: i64,
        length: i64,
        uid: u64,
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

/// WaveformSamplerPy is a wrapper around Python implementation of waveform sampling.
///
/// The sampler will be used to sample waveforms and compress them into a format that can be played on the target AWG.
///
/// This way we avoid arbitrary Python code execution (pulse definitions) in the Rust code and also avoid
/// handling of `numpy` arrays in Rust at this point.
pub struct WaveformSamplerPy<'a> {
    sampler: Py<PyAny>,
    sampling_rate: f64,
    device_kind: &'a DeviceKind,
    multi_iq_signal: bool,
    signal_map: HashMap<&'a str, Rc<Signal>>,
    mixer_type: Option<&'a MixerType>,
    signal_kind: &'a SignalKind,
}

impl<'a> WaveformSamplerPy<'a> {
    pub fn supports_waveform_sampling(awg: &'a AwgCore) -> bool {
        awg.signals
            .iter()
            .any(|s| s.kind != SignalKind::INTEGRATION)
    }

    pub fn new(sampler: Py<PyAny>, awg: &'a AwgCore) -> Self {
        let sampling_rate = awg.sampling_rate;
        let multi_iq_signal = awg.kind == AwgKind::MULTI;
        let signal_map = awg
            .signals
            .iter()
            .map(|s| (s.uid.as_str(), Rc::clone(s)))
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
            device_kind: &awg.device_kind,
            multi_iq_signal,
            signal_map,
            mixer_type,
            signal_kind,
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
        let multi_iq_signal = self.multi_iq_signal;
        let mut sampled_waveforms: SampledWaveformCollection<SampledWaveformSignaturePy> =
            SampledWaveformCollection::new();
        Python::with_gil(|py| -> Result<()> {
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
                    multi_iq_signal,
                    mixer_type.as_ref(),
                    &signal_type,
                )?;
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
        .extract::<Option<PyObject>>()?
        .is_some();
    let m2 = signature
        .getattr(intern!(py, "samples_marker2"))?
        .extract::<Option<PyObject>>()?
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
    multi_iq_signal: bool,
    mixer_type: Option<&Bound<'_, MixerTypePy>>,
    signal_type: &Bound<'_, SignalTypePy>,
) -> Result<Option<PyObject>> {
    if !should_sample_waveform(waveform) {
        // If the waveform does not need to be sampled, skip the sampling process
        return Ok(None);
    }
    let signals_uid = signals.iter().map(|s| &s.uid).collect::<Vec<_>>();
    let waveform_py: WaveformSignaturePy = WaveformSignaturePy::new(waveform.clone());
    let result: std::result::Result<Bound<'_, PyAny>, PyErr> = sampler.call_method(
        intern!(py, "sample_and_compress"),
        (
            waveform_py,
            &signals_uid,
            sampling_rate,
            signal_type,
            device_type,
            mixer_type,
            multi_iq_signal,
        ),
        None,
    );
    match result {
        Ok(sampled_signature) => {
            if sampled_signature.is_none() {
                // If the waveform does not need to be sampled, skip the sampling process
                return Ok(None);
            }
            Ok(Some(sampled_signature.unbind()))
        }
        Err(e) => Err(Error::External(Box::new(e))),
    }
}
