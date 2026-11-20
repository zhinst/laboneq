// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use num_complex::Complex64;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PySlice};

use codegenerator_utils::pulse_sampling::{self, PulseSpec};
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::operation::ExternalOrValue;
use laboneq_dsl::types::{
    BuiltinPulse, ExternalParameterUid, NumericLiteral, PulseFunction, PulseParameterUid,
    ValueOrParameter,
};
use laboneq_pulse_sampling::{EnvelopeShape, PulseRequest, Samples};
use laboneq_py_utils::constant_serializer;
use laboneq_py_utils::pulse_library;
use laboneq_py_utils::py_constant::{self, PyConstant};
use laboneq_py_utils::py_object_store::{PyObjectPayload, PyObjectStore};
use numeric_array::NumericArray;

use crate::PythonSampler;

pub fn create_py_module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(py, name)?;
    module.add_function(wrap_pyfunction!(sample_pulse, &module)?)?;
    Ok(module)
}

#[pyfunction]
#[pyo3(signature = (
    *,
    signal_type,
    sampling_rate,
    length,
    amplitude,
    pulse_function,
    modulation_frequency=None,
    phase=None,
    samples=None,
    pulse_parameters=None,
    subslice=None,
))]
#[allow(clippy::too_many_arguments)]
fn sample_pulse<'py>(
    py: Python<'py>,
    signal_type: &str,
    sampling_rate: f64,
    length: f64,
    amplitude: Complex64,
    pulse_function: Option<String>,
    modulation_frequency: Option<f64>,
    phase: Option<f64>,
    samples: Option<Bound<'py, PyAny>>,
    pulse_parameters: Option<Bound<'py, PyDict>>,
    subslice: Option<(usize, usize)>,
) -> PyResult<Bound<'py, PyAny>> {
    let pulse_function = pulse_function.filter(|name| !name.is_empty());
    // `round` in Python rounds half to even.
    let length_samples = (length * sampling_rate).round_ties_even() as usize;
    let (start, end) = subslice.unwrap_or((0, length_samples));

    let mut id_store = NamedIdStore::new();
    let mut external_parameters = PyObjectStore::<ExternalParameterUid>::new();
    let parameters = pulse_parameters
        .as_ref()
        .map(|p| parameters_from_py(p, &mut id_store, &mut external_parameters))
        .transpose()?;
    let samples = samples.map(|s| sampled_array(py, &s)).transpose()?;
    let builtin_names = pulse_library::builtin_names(py)?;
    let function = pulse_function.map(|function| match BuiltinPulse::from_name(&function) {
        Some(pulse) if builtin_names.contains(pulse.name()) => PulseFunction::Builtin(pulse),
        _ => PulseFunction::Custom { function },
    });
    let shape = match (&function, &samples) {
        (Some(function), None) => PulseSpec {
            pulse_id: None,
            function,
            parameters: parameters.as_ref(),
        }
        .shape(length_samples),
        (None, Some(samples)) => EnvelopeShape::Sampled(samples),
        (Some(_), Some(_)) => {
            let msg = "Only one of samples or pulse_function may be given at the same time.";
            return Err(PyValueError::new_err(msg));
        }
        (None, None) => {
            let msg = "Must provide either samples or a function";
            return Err(PyValueError::new_err(msg));
        }
    };
    let (pulse_parameters, factor) =
        pulse_sampling::resolve_pulse_parameters(parameters.as_ref(), &id_store);
    let factor = factor
        .ok_or_else(|| PyValueError::new_err("Pulse parameter 'amplitude' must be a number"))?;
    let pulse = PulseRequest {
        shape,
        slice: start..end,
        amplitude: amplitude * factor,
        parameters: Some(&pulse_parameters),
        phase: phase.unwrap_or_default(),
        modulation_frequency: modulation_frequency.unwrap_or_default(),
        target: 0,
        offset: 0,
    };

    let mut samples = if signal_type == "iq" {
        Samples::Complex(vec![Complex64::ZERO; end - start])
    } else {
        Samples::Real(vec![0.0; end - start])
    };
    let sampler = PythonSampler::new(&id_store, &external_parameters);
    let targets = &mut [samples.as_samples_mut()];
    laboneq_pulse_sampling::sample_pulses(&[pulse], targets, sampling_rate, &sampler, None)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(match samples {
        Samples::Real(samples) => PyArray1::from_vec(py, samples).into_any(),
        Samples::Complex(samples) => PyArray1::from_vec(py, samples).into_any(),
    })
}

/// Converts pulse parameters as the compiler receives them: constants it reads, and every
/// other value handed over opaquely.
fn parameters_from_py(
    parameters: &Bound<'_, PyDict>,
    id_store: &mut NamedIdStore,
    external_parameters: &mut PyObjectStore<ExternalParameterUid>,
) -> PyResult<HashMap<PulseParameterUid, ExternalOrValue>> {
    let py = parameters.py();
    let mut result = HashMap::new();
    for (key, value) in parameters.iter() {
        let key = PulseParameterUid(id_store.get_or_insert(key.extract::<String>()?));
        let constant =
            |literal| ExternalOrValue::ValueOrParameter(ValueOrParameter::Value(literal));
        let mut external = |payload| {
            PyResult::Ok(ExternalOrValue::ExternalParameter(
                external_parameters.store(py, payload)?,
            ))
        };
        let value = match py_constant::classify_value(&value)? {
            PyConstant::None => ExternalOrValue::None,
            PyConstant::Float(value) => constant(NumericLiteral::Float(value)),
            PyConstant::Int(value) => constant(NumericLiteral::Int(value)),
            PyConstant::Complex(value) => constant(NumericLiteral::Complex(value)),
            PyConstant::Bool(value) => ExternalOrValue::Bool(value),
            PyConstant::Str(value) => external(PyObjectPayload::Str(value))?,
            PyConstant::RawBytes(value) => external(PyObjectPayload::RawBytes(value))?,
            PyConstant::Opaque => external(PyObjectPayload::Json(
                &constant_serializer::serialize_json(&value)?,
            ))?,
        };
        result.insert(key, value);
    }
    Ok(result)
}

/// Converts the samples of a sampled pulse, turning (N, 2) samples into complex ones as
/// `sample_pulse` does.
fn sampled_array(py: Python<'_>, samples: &Bound<'_, PyAny>) -> PyResult<NumericArray> {
    let numpy = py.import(intern!(py, "numpy"))?;
    let array = numpy.call_method1(intern!(py, "asarray"), (samples,))?;
    let shape: Vec<usize> = array.getattr(intern!(py, "shape"))?.extract()?;
    let array = if shape.len() == 2 && shape[1] == 2 {
        let real = array.get_item((PySlice::full(py), 0))?;
        let imag = array.get_item((PySlice::full(py), 1))?;
        real.add(PyComplex::from_doubles(py, 0.0, 1.0).mul(imag)?)?
    } else {
        array
    };
    NumericArray::from_py(&array)
}
