// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Deserializes `CompiledExperiment.resultShapes` into
//! `laboneq.data.scheduled_experiment.ResultShapeInfo`.

use pyo3::IntoPyObjectExt;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use laboneq_capnp::compiled_experiment::v1::result_shape_capnp;

use laboneq_error::{WithContext, laboneq_error};

use crate::Result;
use crate::context::{DeserializerContext, capnp_err, decode_numeric_array, text_to_string};

fn scheduled_experiment_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    py.import(intern!(py, "laboneq.data.scheduled_experiment"))
}

pub(crate) fn build_result_shape_info<'py>(
    py: Python<'py>,
    reader: result_shape_capnp::result_shape_info::Reader<'_>,
    ctx: &DeserializerContext,
) -> Result<Bound<'py, PyAny>> {
    let explicit_values: Vec<Bound<'py, PyAny>> = reader
        .get_explicit_values()
        .map_err(capnp_err)?
        .iter()
        .map(|r| decode_numeric_array(py, r))
        .collect::<Result<_>>()?;

    let module = scheduled_experiment_module(py)?;
    let handle_result_shape_class = module.getattr(intern!(py, "HandleResultShape"))?;

    let shapes_dict = PyDict::new(py);
    for entry in reader.get_shapes().map_err(capnp_err)?.iter() {
        let handle_name = ctx.handle_name(entry.get_handle())?;
        let shape_py = build_handle_result_shape(
            py,
            entry.get_shape().map_err(capnp_err)?,
            ctx,
            &explicit_values,
            &handle_result_shape_class,
        )
        .with_context(|| format!("Result shape of handle '{handle_name}'."))?;
        shapes_dict.set_item(handle_name, shape_py)?;
    }

    let result_shape_info_class = module.getattr(intern!(py, "ResultShapeInfo"))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("shapes", shapes_dict)?;
    Ok(result_shape_info_class.call((), Some(&kwargs))?)
}

fn build_handle_result_shape<'py>(
    py: Python<'py>,
    reader: result_shape_capnp::handle_result_shape::Reader<'_>,
    ctx: &DeserializerContext,
    explicit_values: &[Bound<'py, PyAny>],
    handle_result_shape_class: &Bound<'py, PyAny>,
) -> Result<Bound<'py, PyAny>> {
    let shape: Vec<u64> = reader.get_shape().map_err(capnp_err)?.iter().collect();
    let shape_tuple = PyTuple::new(py, shape)?;

    let mut axis_names = Vec::new();
    let mut axis_values = Vec::new();
    for axis in reader.get_axes().map_err(capnp_err)?.iter() {
        let mut names = Vec::new();
        let mut values = Vec::new();
        for param in axis.get_parameters().map_err(capnp_err)?.iter() {
            let name = text_to_string(param.get_name().map_err(capnp_err)?)?;
            let value = decode_axis_parameter(py, param, ctx, explicit_values)?;
            names.push(name);
            values.push(value);
        }
        axis_names.push(flatten_single_axis(py, names)?);
        axis_values.push(flatten_single_axis(py, values)?);
    }

    use result_shape_capnp::handle_result_shape::chunked_axis_index::Which as ChunkedWhich;
    let chunked_axis_index: Option<u32> =
        match reader.get_chunked_axis_index().which().map_err(capnp_err)? {
            ChunkedWhich::None(()) => None,
            ChunkedWhich::Value(v) => Some(v),
        };

    let mask_entries = reader.get_match_case_mask().map_err(capnp_err)?;
    let match_case_mask = if mask_entries.is_empty() {
        py.None()
    } else {
        let dict = PyDict::new(py);
        for entry in mask_entries.iter() {
            let rows: Vec<u64> = entry.get_rows().map_err(capnp_err)?.iter().collect();
            dict.set_item(entry.get_axis(), rows)?;
        }
        dict.into_any().unbind()
    };

    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", shape_tuple)?;
    kwargs.set_item("axis_names", PyList::new(py, axis_names)?)?;
    kwargs.set_item("axis_values", PyList::new(py, axis_values)?)?;
    kwargs.set_item("chunked_axis_index", chunked_axis_index)?;
    kwargs.set_item("match_case_mask", match_case_mask)?;
    Ok(handle_result_shape_class.call((), Some(&kwargs))?)
}

fn decode_axis_parameter<'py>(
    py: Python<'py>,
    reader: result_shape_capnp::axis_parameter::Reader<'_>,
    ctx: &DeserializerContext,
    explicit_values: &[Bound<'py, PyAny>],
) -> Result<Bound<'py, PyAny>> {
    use result_shape_capnp::axis_parameter::Which;
    match reader.which().map_err(capnp_err)? {
        Which::ParameterRef(idx) => Ok(ctx.parameter_array(py, idx)?.into_bound(py)),
        Which::Shots(count) => {
            Ok(numpy::PyArray1::from_iter(py, (0..count).map(|v| v as f64))
                .into_bound_py_any(py)?)
        }
        Which::ExplicitRef(idx) => explicit_values.get(idx as usize).cloned().ok_or_else(|| {
            laboneq_error!("ResultShapeInfo.explicitValues has no entry at index {idx}")
        }),
    }
}

/// Unwraps a single-element axis to its bare value, matching how the Python-side dataclass
/// represents an axis with just one name/value instead of a one-element list.
fn flatten_single_axis<'py, T>(py: Python<'py>, mut items: Vec<T>) -> PyResult<Bound<'py, PyAny>>
where
    T: IntoPyObject<'py>,
    Vec<T>: IntoPyObject<'py>,
{
    if items.len() == 1 {
        items.remove(0).into_bound_py_any(py)
    } else {
        items.into_bound_py_any(py)
    }
}
