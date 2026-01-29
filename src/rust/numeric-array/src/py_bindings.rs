// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! `pyo3` support for `NumericArray`.
//!
//! This module provides functionality to convert Python arrays into `NumericArray`.
//!
//! This module requires optional feature `pyo3`.
use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::NumericArray;

fn extract_numeric_array(arr: &Bound<'_, PyAny>) -> Result<NumericArray, PyErr> {
    // TODO: Possibly lazy loading in the future?
    // NOTE: Somewhere down the line the input can be `list` instead of `ndarray`,
    // so we will support it for the time begin.
    let numpy = PyModule::import(arr.py(), "numpy")?;
    let array_function = numpy.getattr("array")?;

    // Convert array to `numpy.array` so that `PyArray1` will handle all the validation.
    // This supports e.g. lists, iterators with valid items and so on.
    let nd_array = numpy.getattr("ndarray")?;
    let py_arr = if !arr.is_instance(&nd_array)? {
        &array_function.call1((arr,))?
    } else {
        arr
    };
    if let Ok(arr) = py_arr.cast::<PyArray1<f64>>() {
        return Ok(NumericArray::Float64(arr.try_readonly()?.to_vec()?));
    }
    if let Ok(arr) = py_arr.cast::<PyArray1<i64>>() {
        return Ok(NumericArray::Integer64(arr.try_readonly()?.to_vec()?));
    }
    if let Ok(arr) = py_arr.cast::<PyArray1<Complex64>>() {
        return Ok(NumericArray::Complex64(arr.try_readonly()?.to_vec()?));
    }
    Err(PyValueError::new_err(
        "Expected a 1-dimensional array that can be downcast to either 'float64', 'int64' or 'complex128'",
    ))
}

#[cfg(feature = "pyo3")]
impl NumericArray {
    /// Convert a Python list or an numpy array into a `NumericArray`.
    ///
    /// # Arguments
    ///
    /// * ob - A 1-dimensional Python `list` or `numpy.ndarray` of homogenous values types
    ///
    /// # Returns
    ///
    /// * Ok(): A `NumericArray`
    /// * Err(): The value cannot be converted into `NumericArray`
    #[cfg(feature = "pyo3")]
    pub fn from_py(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        extract_numeric_array(ob)
    }

    /// Convert the `NumericArray` into a numpy array.
    #[cfg(feature = "pyo3")]
    pub fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObjectExt;

        match self {
            NumericArray::Float64(arr) => PyArray1::from_vec(py, arr.clone()).into_bound_py_any(py),
            NumericArray::Integer64(arr) => {
                PyArray1::from_vec(py, arr.clone()).into_bound_py_any(py)
            }
            NumericArray::Complex64(arr) => {
                PyArray1::from_vec(py, arr.clone()).into_bound_py_any(py)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3_ffi::c_str;
    use std::ffi::CStr;

    use crate::NumericArray;
    use num_complex::Complex;

    fn is_numpy_array(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
        let numpy = PyModule::import(arr.py(), "numpy")?;
        let ndarray_type = numpy.getattr("ndarray")?;
        arr.is_instance(&ndarray_type)
    }

    fn test_round_trip_conversion(arr: &NumericArray) {
        Python::attach(|py| {
            let c = arr.to_py(py).unwrap();
            assert!(is_numpy_array(&c).unwrap());
            let arr_back = NumericArray::from_py(&c).unwrap();
            assert_eq!(arr, &arr_back);
        })
    }

    /// Helper function to build a NumericArray from Python code defining `arr`.
    ///
    /// This function also tests round-trip conversion on successful conversion.
    fn build_arr(py_text: &CStr) -> Result<NumericArray, PyErr> {
        Python::attach(|py| {
            let activators =
                PyModule::from_code(py, py_text, c_str!("test.py"), c_str!("test")).unwrap();
            let py_obj = activators.getattr("arr").unwrap();
            let arr = NumericArray::from_py(&py_obj);
            if let Ok(arr) = arr {
                // Test round-trip conversion
                test_round_trip_conversion(&arr);
                return Ok(arr);
            }
            arr
        })
    }

    #[test]
    fn test_empty_arr() {
        let cstring = c_str!(r#"arr = []"#);
        assert_eq!(build_arr(cstring).unwrap(), NumericArray::Float64(vec![]));

        let cstring = c_str!(r#"import numpy as np; arr = np.array([])"#);
        assert_eq!(build_arr(cstring).unwrap(), NumericArray::Float64(vec![]));
    }

    #[test]
    fn test_py_list_real() {
        let cstring = c_str!(r#"arr = [1, 2]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Integer64(vec![1, 2])
        );

        let cstring = c_str!(r#"arr = [-1, 2]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Integer64(vec![-1, 2])
        );

        let cstring = c_str!(r#"arr = [-1.2, 2]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Float64(vec![-1.2, 2.0])
        );

        let cstring = c_str!(r#"arr = [2, 2.1]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Float64(vec![2.0, 2.1])
        );
    }

    #[test]
    fn test_py_list_real_complex() {
        let cstring = c_str!(r#"arr = [complex(1)]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Complex64(vec![Complex::new(1.0, 0.0)])
        );

        let cstring = c_str!(r#"arr = [complex(1, 0.2)]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Complex64(vec![Complex::new(1.0, 0.2)])
        );

        let cstring = c_str!(r#"arr = [complex(-1, 0.2)]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Complex64(vec![Complex::new(-1.0, 0.2)])
        );

        let cstring = c_str!(r#"arr = [complex(1, -0.2)]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Complex64(vec![Complex::new(1.0, -0.2)])
        );
    }

    #[test]
    fn test_numpy_arr() {
        let cstring = c_str!(r#"import numpy as np; arr = np.array([1, 2])"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Integer64(vec![1, 2])
        );

        let cstring = c_str!(r#"import numpy as np; arr = np.array([complex(1, 0.1)])"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Complex64(vec![Complex::new(1.0, 0.1)])
        );
    }

    #[test]
    fn test_numpy_type_in_list() {
        let cstring = c_str!(r#"import numpy as np; arr = [np.int64(1), np.float64(1.2)]"#);
        assert_eq!(
            build_arr(cstring).unwrap(),
            NumericArray::Float64(vec![1.0, 1.2])
        );
    }

    #[test]
    fn test_invalid_input() {
        let expected_error_message = "ValueError: Expected a 1-dimensional array that can be downcast to either 'float64', 'int64' or 'complex128'";
        let cstring = c_str!(r#"arr = [None]"#);
        assert_eq!(
            build_arr(cstring).unwrap_err().to_string(),
            expected_error_message.to_string()
        );

        let cstring = c_str!(r#"arr = ["test"]"#);
        assert_eq!(
            build_arr(cstring).unwrap_err().to_string(),
            expected_error_message.to_string()
        );

        let cstring = c_str!(r#"arr = None"#);
        assert_eq!(
            build_arr(cstring).unwrap_err().to_string(),
            expected_error_message.to_string()
        );
    }
}
