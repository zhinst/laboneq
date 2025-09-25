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
use pyo3::types::PyList;

use crate::NumericArray;

fn extract_numeric_array(ob: &Bound<'_, PyAny>) -> Result<NumericArray, PyErr> {
    // TODO: Possibly lazy loading in the future?
    // NOTE: Somewhere down the line the input can be `list` instead of `ndarray`,
    // so we will support it for the time begin.

    // Convert PyList to `numpy.array` so that `PyArray1` will handle all the validation.
    // Numpy should be the standard either way.
    let py_arr = if ob.is_instance_of::<PyList>() {
        let numpy = PyModule::import(ob.py(), "numpy")?;
        let array_function = numpy.getattr("array")?;
        &array_function.call1((ob,))?
    } else {
        ob
    };
    if let Ok(arr) = py_arr.downcast::<PyArray1<f64>>() {
        return Ok(NumericArray::Float64(arr.try_readonly()?.to_vec()?));
    }
    if let Ok(arr) = py_arr.downcast::<PyArray1<i64>>() {
        return Ok(NumericArray::Integer64(arr.try_readonly()?.to_vec()?));
    }
    if let Ok(arr) = py_arr.downcast::<PyArray1<Complex64>>() {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3_ffi::c_str;
    use std::ffi::CStr;

    use crate::NumericArray;
    use num_complex::Complex;

    fn build_arr(py_text: &CStr) -> Result<NumericArray, PyErr> {
        Python::attach(|py| {
            let activators =
                PyModule::from_code(py, py_text, c_str!("test.py"), c_str!("test")).unwrap();
            let py_obj = activators.getattr("arr").unwrap();
            NumericArray::from_py(&py_obj)
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
