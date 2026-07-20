// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::result_shape::{AxisValues, HandleResultShape};
use laboneq_common::named_id::NamedIdStore;

#[pyclass(name = "HandleResultShape")]
pub(crate) struct HandleResultShapePy {
    #[pyo3(get)]
    handle: String,
    #[pyo3(get)]
    shape: Vec<usize>,
    #[pyo3(get)]
    axis_names: Vec<Vec<String>>,
    #[pyo3(get)]
    axis_values: Vec<Vec<Py<PyAny>>>, // list[list[NumPyArray]]]
    #[pyo3(get)]
    chunked_axis_index: Option<usize>,
    #[pyo3(get)]
    match_case_mask: HashMap<usize, Vec<usize>>, // dict[int, list[int]]
}

pub(crate) fn create_result_shape_py(
    py: Python,
    result_shape: HandleResultShape,
    id_store: &NamedIdStore,
) -> PyResult<HandleResultShapePy> {
    use numpy::PyArray1;
    use pyo3::IntoPyObjectExt;

    let shape = result_shape.shape;
    let axis_names = result_shape
        .axis_names
        .into_iter()
        .map(|names| {
            names
                .into_iter()
                .map(|name| id_store.resolve(name).unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let axis_values = result_shape
        .axis_values
        .into_iter()
        .map(|values| {
            values
                .into_iter()
                .map(|value| match value {
                    AxisValues::Range(range) => {
                        // Create a NumPy array with values from range.start to range.end - 1 and as type float64
                        PyArray1::from_iter(py, (range.start..range.end).map(|v| v as f64))
                            .into_py_any(py)
                    }
                    AxisValues::Explicit(array) => match array.to_py(py) {
                        Ok(arr) => arr.into_py_any(py),
                        Err(e) => Err(e),
                    },
                })
                .collect::<Result<Vec<_>, PyErr>>()
        })
        .collect::<Result<Vec<Vec<_>>, PyErr>>()?;

    let chunked_axis_index = result_shape.chunked_axis_index;
    let match_case_mask = result_shape.match_case_mask;

    let result_shape_py = HandleResultShapePy {
        handle: id_store.resolve(result_shape.handle).unwrap().to_string(),
        shape,
        axis_names,
        axis_values,
        chunked_axis_index,
        match_case_mask: match_case_mask.into_iter().collect(),
    };
    Ok(result_shape_py)
}
