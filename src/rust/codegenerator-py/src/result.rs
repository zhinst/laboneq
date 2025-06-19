// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Result module for the code generator.
//!
//! This module defines the [`AwgCodeGenerationResultPy`] class, which is used to
//! represent the result of the code generation process for an AWG.
use std::collections::HashSet;

use crate::{awg_event::AwgEvent, waveform_sampler::SampledWaveformSignaturePy};
use codegenerator::{SampledWaveform, WaveDeclaration};
use pyo3::prelude::*;

#[pyclass(name = "WaveDeclaration")]
#[derive(Debug)]
pub struct WaveDeclarationPy {
    pub obj: WaveDeclaration,
}

#[pymethods]
impl WaveDeclarationPy {
    #[getter]
    pub fn length(&self) -> i64 {
        self.obj.length
    }

    #[getter]
    pub fn signature_string(&self) -> &str {
        &self.obj.signature_string
    }

    #[getter]
    pub fn has_marker1(&self) -> bool {
        self.obj.has_marker1
    }

    #[getter]
    pub fn has_marker2(&self) -> bool {
        self.obj.has_marker2
    }
}

#[pyclass(name = "SampledWaveform")]
#[derive(Debug)]
pub struct SampledWaveformPy {
    pub obj: SampledWaveform<SampledWaveformSignaturePy>,
}

#[pymethods]
impl SampledWaveformPy {
    #[getter]
    pub fn signals(&self) -> &HashSet<String> {
        &self.obj.signals
    }

    #[getter]
    pub fn signature_string(&self) -> &str {
        &self.obj.signature_string
    }

    #[getter]
    pub fn signature(&self, py: Python) -> PyObject {
        self.obj.signature.signature.clone_ref(py)
    }
}

/// Result structure for single AWG code generation.
#[pyclass(name = "AwgCodeGenerationResult", frozen, unsendable)]
pub struct AwgCodeGenerationResultPy {
    awg_events: Vec<Py<AwgEvent>>,
    sampled_waveforms: Vec<Py<SampledWaveformPy>>,
    wave_declarations: Vec<Py<WaveDeclarationPy>>,
}

impl AwgCodeGenerationResultPy {
    pub fn create(
        awg_events: Vec<AwgEvent>,
        sampled_waveforms: Vec<SampledWaveform<SampledWaveformSignaturePy>>,
        wave_declarations: Vec<WaveDeclaration>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            let awg_events: Vec<Py<AwgEvent>> = awg_events
                .into_iter()
                .map(|event| Py::new(py, event).expect("Failed to create AwgEvent"))
                .collect();
            let sampled_waveforms: Vec<Py<SampledWaveformPy>> = Python::with_gil(|py| {
                sampled_waveforms
                    .into_iter()
                    .map(|sampled| Py::new(py, SampledWaveformPy { obj: sampled }).unwrap())
                    .collect()
            });
            let wave_declarations: Vec<Py<WaveDeclarationPy>> = Python::with_gil(|py| {
                wave_declarations
                    .into_iter()
                    .map(|decl| Py::new(py, WaveDeclarationPy { obj: decl }).unwrap())
                    .collect()
            });
            let output = AwgCodeGenerationResultPy {
                awg_events,
                sampled_waveforms,
                wave_declarations,
            };
            Ok(output)
        })
    }

    pub fn default() -> Self {
        AwgCodeGenerationResultPy {
            awg_events: vec![],
            sampled_waveforms: vec![],
            wave_declarations: vec![],
        }
    }
}

#[pymethods]
impl AwgCodeGenerationResultPy {
    #[getter]
    fn awg_events(&self) -> &Vec<Py<AwgEvent>> {
        &self.awg_events
    }

    #[getter]
    fn sampled_waveforms(&self) -> &Vec<Py<SampledWaveformPy>> {
        &self.sampled_waveforms
    }

    #[getter]
    fn wave_declarations(&self) -> &Vec<Py<WaveDeclarationPy>> {
        &self.wave_declarations
    }
}
