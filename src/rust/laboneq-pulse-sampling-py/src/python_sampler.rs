// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyKeyError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyList, PySlice};

use codegenerator_utils::pulse_sampling::PulseSpec;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::ExternalParameterUid;
use laboneq_error::{LabOneQError, bail};
use laboneq_pulse_sampling::external::{EnvelopeFn, ExternalRequest, ExternalSampler};
use laboneq_pulse_sampling::{Samples, SamplesRef};
use laboneq_py_utils::py_export;
use laboneq_py_utils::py_object_store::PyObjectStore;

/// Evaluates functional pulse envelopes using `laboneq.dsl.experiment.pulse_library.pulse_sampler`,
/// looking up their pulse library registration by name.
///
/// Holds the GIL once per batch of requests.
pub struct PythonSampler<'a> {
    id_store: &'a NamedIdStore,
    external_parameters: &'a PyObjectStore<ExternalParameterUid>,
}

impl<'a> PythonSampler<'a> {
    pub fn new(
        id_store: &'a NamedIdStore,
        external_parameters: &'a PyObjectStore<ExternalParameterUid>,
    ) -> Self {
        PythonSampler {
            id_store,
            external_parameters,
        }
    }
}

impl<'p> ExternalSampler<PulseSpec<'p>> for PythonSampler<'_> {
    fn for_each_envelope(
        &self,
        requests: &[ExternalRequest<'_, PulseSpec<'p>>],
        sampling_rate: f64,
        f: &mut EnvelopeFn<'_>,
    ) -> Result<(), LabOneQError> {
        Python::attach(|py| {
            let pulse_sampler = py
                .import(intern!(py, "laboneq.dsl.experiment.pulse_library"))?
                .getattr(intern!(py, "pulse_sampler"))?;
            let numpy = py.import(intern!(py, "numpy"))?;
            for request in requests {
                let samples = self.envelope(py, &pulse_sampler, &numpy, request, sampling_rate)?;
                let envelope = as_envelope(&samples);
                if envelope.len() < request.slice.len() {
                    let name = match request.shape.pulse.pulse_id {
                        Some(pulse_id) => format!("Pulse '{}'", pulse_id.0),
                        None => "The pulse".to_owned(),
                    };
                    bail!(
                        "{name} evaluates {} samples, but {} are expected",
                        envelope.len(),
                        request.slice.len()
                    );
                }
                f(request.index, envelope)?;
            }
            Ok(())
        })
    }
}

impl PythonSampler<'_> {
    /// Returns a request's envelope.
    fn envelope<'py>(
        &self,
        py: Python<'py>,
        pulse_sampler: &Bound<'py, PyAny>,
        numpy: &Bound<'py, PyModule>,
        request: &ExternalRequest<'_, PulseSpec<'_>>,
        sampling_rate: f64,
    ) -> Result<PythonEnvelope<'py>, LabOneQError> {
        let pulse = request.shape.pulse;
        let parameters = match pulse.parameters {
            Some(parameters) => py_export::pulse_parameters_to_py_dict(
                py,
                parameters,
                self.id_store,
                self.external_parameters,
            )?
            .into_bound(py),
            None => PyDict::new(py),
        };
        // the request amplitude includes the `amplitude` parameter.
        if parameters.contains(intern!(py, "amplitude"))? {
            parameters.del_item(intern!(py, "amplitude"))?;
        }
        let name = pulse.function.name();
        let result = call_pulse_function(pulse_sampler, name, request, sampling_rate, &parameters)?;
        functional_envelope(py, numpy, name, result, request.slice.len())
    }
}

/// Calls the pulse function registered under `name` at the evaluation points of `request`,
/// passing its length and amplitude, `sampling_rate` and `parameters` as keyword arguments.
fn call_pulse_function<'py>(
    pulse_sampler: &Bound<'py, PyAny>,
    name: &str,
    request: &ExternalRequest<'_, PulseSpec<'_>>,
    sampling_rate: f64,
    parameters: &Bound<'py, PyDict>,
) -> Result<Bound<'py, PyAny>, LabOneQError> {
    let py = pulse_sampler.py();
    let sampler = match pulse_sampler.call1((name,)) {
        Ok(sampler) => sampler,
        Err(err) if err.is_instance_of::<PyKeyError>(py) => {
            bail!("Function '{name}' is not registered in the library of pulse functions.")
        }
        Err(err) => return Err(err.into()),
    };
    let x_axis = request.evaluation_points(sampling_rate).collect();
    let (length, amplitude) = (request.shape.length, request.amplitude);
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "length"), length as f64 / sampling_rate)?;
    kwargs.set_item(
        intern!(py, "amplitude"),
        PyComplex::from_doubles(py, amplitude.re, amplitude.im),
    )?;
    kwargs.set_item(intern!(py, "sampling_rate"), sampling_rate)?;
    for (key, value) in parameters.iter() {
        if kwargs.contains(&key)? {
            bail!("Pulse functional '{name}' got multiple values for keyword argument '{key}'");
        }
        kwargs.set_item(key, value)?;
    }
    Ok(sampler.call((PyArray1::from_vec(py, x_axis),), Some(&kwargs))?)
}

type PythonEnvelope<'py> = Samples<PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, Complex64>>;

fn as_envelope<'a>(envelope: &'a PythonEnvelope<'_>) -> SamplesRef<'a> {
    match envelope {
        Samples::Real(samples) => {
            Samples::Real(samples.as_slice().expect("the array is contiguous"))
        }
        Samples::Complex(samples) => {
            Samples::Complex(samples.as_slice().expect("the array is contiguous"))
        }
    }
}

/// Returns up to the first `num_samples` samples of a functional pulse.
/// (N, 2) samples become complex, other dtypes become float64, or
/// complex128 when they are neither real nor complex numbers.
fn functional_envelope<'py>(
    py: Python<'py>,
    numpy: &Bound<'py, PyModule>,
    name: &str,
    result: Bound<'py, PyAny>,
    num_samples: usize,
) -> Result<PythonEnvelope<'py>, LabOneQError> {
    if !result.is_instance_of::<PyList>()
        && !result.is_instance(&numpy.getattr(intern!(py, "ndarray"))?)?
    {
        bail!("Pulse functional '{name}' must return a list or a numpy array");
    }
    let samples = result.get_item(PySlice::new(py, 0, num_samples as isize, 1))?;
    let mut samples = numpy.call_method1(intern!(py, "asarray"), (samples,))?;
    let shape: Vec<usize> = samples.getattr(intern!(py, "shape"))?.extract()?;
    if shape.len() == 2 && shape[1] == 2 {
        let real = samples.get_item((PySlice::full(py), 0))?;
        let imag = samples.get_item((PySlice::full(py), 1))?;
        samples = real.add(PyComplex::from_doubles(py, 0.0, 1.0).mul(imag)?)?;
    }
    let dtype = samples.getattr(intern!(py, "dtype"))?;
    let float64 = numpy.getattr(intern!(py, "float64"))?;
    let complex128 = numpy.getattr(intern!(py, "complex128"))?;
    if !dtype.eq(&float64)? && !dtype.eq(&complex128)? {
        let mut canonical = numpy.call_method1(intern!(py, "result_type"), (&dtype, &float64))?;
        let kind: String = canonical.getattr(intern!(py, "kind"))?.extract()?;
        if kind != "f" && kind != "c" {
            canonical = complex128;
        }
        samples = samples.call_method1(intern!(py, "astype"), (canonical,))?;
    }
    let samples = numpy.call_method1(intern!(py, "ascontiguousarray"), (samples,))?;
    if let Ok(samples) = samples.cast::<PyArray1<f64>>() {
        Ok(Samples::Real(samples.readonly()))
    } else if let Ok(samples) = samples.cast::<PyArray1<Complex64>>() {
        Ok(Samples::Complex(samples.readonly()))
    } else {
        bail!("Pulse functional '{name}' must return a one-dimensional array");
    }
}
