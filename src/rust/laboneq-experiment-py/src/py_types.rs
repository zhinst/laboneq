// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use pyo3::types::PyString;
use pyo3::{intern, prelude::*};

#[derive(FromPyObject, Debug)]
pub(crate) struct ExperimentCapnpPy<'py> {
    /// Optional UID of the experiment.
    pub uid: Option<Bound<'py, PyString>>,
    /// Sections of the experiment.
    /// The expected Python type is `laboneq.dsl.experiment.Section`.
    pub sections: Vec<Bound<'py, PyAny>>,
    pub experiment_signals: Vec<ExperimentSignalPy<'py>>,
    /// Labels of the HQCS coprocessors declared on the experiment.
    pub coprocessors: Vec<String>,
    /// Opaque coprocessor payloads, by label. Absent when unset.
    pub coprocessor_payloads: HashMap<String, Vec<u8>>,
    pub coprocessor_mappings: Vec<(String, Bound<'py, PyAny>)>,
    pub streams: Vec<StreamPy<'py>>,
}

#[derive(Debug)]
pub(crate) struct StreamPy<'py> {
    // `obj` is retained for identity registration (a `send` operation references
    // the same object). `src`/`dst` are coprocessor labels; `None` means the
    // control system.
    pub obj: Bound<'py, PyAny>,
    pub uid: Option<String>,
    pub src: Option<String>,
    pub dst: Option<String>,
    pub link: Option<String>,
    pub fields: Vec<StructFieldPy<'py>>,
}

impl<'py> FromPyObject<'_, 'py> for StreamPy<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let py = obj.py();
        // Fields are stored in an insertion-ordered dict; serialize by value in
        // schema declaration order.
        let fields = obj
            .getattr(intern!(py, "fields"))?
            .call_method0(intern!(py, "values"))?
            .try_iter()?
            .map(|item| item.and_then(|i| i.extract()))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            obj: obj.to_owned(),
            uid: obj.getattr(intern!(py, "uid"))?.extract()?,
            src: obj.getattr(intern!(py, "src"))?.extract()?,
            dst: obj.getattr(intern!(py, "dst"))?.extract()?,
            link: obj.getattr(intern!(py, "link"))?.extract()?,
            fields,
        })
    }
}

#[derive(Debug)]
pub(crate) struct StructFieldPy<'py> {
    pub name: String,
    pub ty: Bound<'py, PyAny>,
    pub binding: FieldBindingPy<'py>,
}

impl<'py> FromPyObject<'_, 'py> for StructFieldPy<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let py = obj.py();
        Ok(Self {
            name: obj.getattr(intern!(py, "name"))?.extract()?,
            ty: obj.getattr(intern!(py, "type"))?,
            binding: obj.extract()?,
        })
    }
}
#[derive(Debug)]
pub(crate) enum FieldBindingPy<'py> {
    OutboundHandles(Vec<String>),
    InboundScalar(Option<Bound<'py, PyAny>>),
    InboundPulse(Option<Bound<'py, PyAny>>),
    Unbound,
}

impl<'py> FromPyObject<'_, 'py> for FieldBindingPy<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let py = obj.py();
        let opt_target = || -> PyResult<Option<Bound<'py, PyAny>>> {
            let t = obj.getattr(intern!(py, "target"))?;
            Ok(if t.is_none() { None } else { Some(t) })
        };
        let class_name = obj.get_type().name()?.to_string();
        Ok(match class_name.as_str() {
            "_OutboundHandlesField" => {
                Self::OutboundHandles(obj.getattr(intern!(py, "handles"))?.extract()?)
            }
            "_InboundScalarField" => Self::InboundScalar(opt_target()?),
            "_InboundPulseField" => Self::InboundPulse(opt_target()?),
            "_OutboundLiteralField" | "_FieldAccessor" => Self::Unbound,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown stream field accessor: {other}"
                )));
            }
        })
    }
}

#[derive(FromPyObject, Debug)]
pub(crate) struct OscillatorPy<'py> {
    pub uid: Bound<'py, PyString>,
    pub frequency: Bound<'py, PyAny>,
    pub modulation: Option<Bound<'py, PyString>>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct ExperimentSignalPy<'py> {
    pub uid: Bound<'py, PyString>,
    pub maps_to: Bound<'py, PyString>,

    // Calibration
    pub amplitude: Option<Bound<'py, PyAny>>,
    pub oscillator: Option<OscillatorPy<'py>>,
    pub lo_frequency: Option<Bound<'py, PyAny>>,
    pub voltage_offset: Option<Bound<'py, PyAny>>,
    pub amplifier_pump: Option<AmplifierPumpPy<'py>>,
    pub port_mode: Option<Bound<'py, PyString>>,
    pub automute: bool,
    pub delay_signal: f64,
    pub port_delay: Option<Bound<'py, PyAny>>,
    pub range: Option<QuantityPy>,
    pub precompensation: Option<PrecompensationPy>,
    pub added_outputs: Vec<OutputRoutePy<'py>>,
    pub threshold: Option<Vec<f64>>,
    pub mixer_calibration: Option<MixerCalibrationPy<'py>>,
}

#[derive(Debug)]
pub(crate) enum UnitPy {
    Volt,
    DBm,
}

impl FromPyObject<'_, '_> for UnitPy {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        match obj
            .getattr(intern!(obj.py(), "name"))?
            .extract::<&str>()?
            .to_ascii_lowercase()
            .as_str()
        {
            "volt" => Ok(Self::Volt),
            "dbm" => Ok(Self::DBm),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown Unit variant: {other}"
            ))),
        }
    }
}

#[derive(FromPyObject, Debug)]
pub(crate) struct QuantityPy {
    pub value: f64,
    pub unit: Option<UnitPy>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct MixerCalibrationPy<'py> {
    pub voltage_offsets: Option<Vec<Bound<'py, PyAny>>>,
    pub correction_matrix: Option<Vec<Vec<Bound<'py, PyAny>>>>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct OutputRoutePy<'py> {
    pub source: Bound<'py, PyString>,
    pub amplitude_scaling: Bound<'py, PyAny>,
    pub phase_shift: Bound<'py, PyAny>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct AmplifierPumpPy<'py> {
    pub alc_on: bool,
    pub pump_on: bool,
    pub pump_filter_on: bool,
    pub pump_power: Bound<'py, PyAny>,
    pub pump_frequency: Bound<'py, PyAny>,
    pub probe_on: bool,
    pub probe_power: Bound<'py, PyAny>,
    pub probe_frequency: Bound<'py, PyAny>,
    pub cancellation_on: bool,
    pub cancellation_phase: Bound<'py, PyAny>,
    pub cancellation_attenuation: Bound<'py, PyAny>,
    pub cancellation_source: CancellationSourcePy,
    pub cancellation_source_frequency: Option<f64>,
}

#[derive(Debug)]
pub(crate) enum CancellationSourcePy {
    Internal,
    External,
}

impl FromPyObject<'_, '_> for CancellationSourcePy {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let name: String = obj.getattr("name")?.extract()?;
        match name.as_str().to_ascii_lowercase().as_str() {
            "internal" => Ok(Self::Internal),
            "external" => Ok(Self::External),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown CancellationSource variant: {other}"
            ))),
        }
    }
}

#[derive(FromPyObject, Debug)]
pub(crate) struct PrecompensationPy {
    pub high_pass: Option<HighPassCompensationPy>,
    pub exponential: Option<Vec<ExponentialCompensationPy>>,
    #[pyo3(attribute("FIR"))]
    pub fir: Option<FirCompensationPy>,
    pub bounce: Option<BounceCompensationPy>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct HighPassCompensationPy {
    pub timeconstant: f64,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct ExponentialCompensationPy {
    pub timeconstant: f64,
    pub amplitude: f64,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct FirCompensationPy {
    pub coefficients: Vec<f64>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct BounceCompensationPy {
    pub delay: f64,
    pub amplitude: f64,
}
