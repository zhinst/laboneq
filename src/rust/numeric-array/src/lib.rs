// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex;

#[cfg(feature = "pyo3")]
mod py_bindings;

/// Numeric array supporting integer, float, and complex 1D arrays.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericArray {
    Integer64(Vec<i64>),
    Float64(Vec<f64>),
    Complex64(Vec<Complex<f64>>),
}

impl NumericArray {
    pub fn abs_at_index(&self, index: usize) -> Option<f64> {
        match self {
            NumericArray::Integer64(vec) => vec.get(index).map(|x| x.abs() as f64),
            NumericArray::Float64(vec) => vec.get(index).map(|x| x.abs()),
            NumericArray::Complex64(vec) => vec.get(index).map(|x| x.norm()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            NumericArray::Integer64(vec) => vec.len(),
            NumericArray::Float64(vec) => vec.len(),
            NumericArray::Complex64(vec) => vec.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            NumericArray::Integer64(vec) => vec.is_empty(),
            NumericArray::Float64(vec) => vec.is_empty(),
            NumericArray::Complex64(vec) => vec.is_empty(),
        }
    }
}

impl NumericArray {
    pub fn linspace(start: f64, stop: f64, count: usize) -> Self {
        if count == 0 {
            return NumericArray::Float64(vec![]);
        }
        if count == 1 {
            return NumericArray::Float64(vec![start]);
        }
        let step = (stop - start) / (count - 1) as f64;
        let mut vec = Vec::with_capacity(count);
        for i in 0..count {
            let value = start + step * i as f64;
            vec.push(value);
        }
        NumericArray::Float64(vec)
    }

    pub fn linspace_complex(start: Complex<f64>, stop: Complex<f64>, count: usize) -> Self {
        if count == 0 {
            return NumericArray::Complex64(vec![]);
        }
        if count == 1 {
            return NumericArray::Complex64(vec![start]);
        }
        let step = (stop - start) / (count - 1) as f64;
        let mut vec = Vec::with_capacity(count);
        for i in 0..count {
            let value = start + step * i as f64;
            vec.push(value);
        }
        NumericArray::Complex64(vec)
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        match self {
            NumericArray::Integer64(vec) => NumericArray::Integer64(vec[range].to_vec()),
            NumericArray::Float64(vec) => NumericArray::Float64(vec[range].to_vec()),
            NumericArray::Complex64(vec) => NumericArray::Complex64(vec[range].to_vec()),
        }
    }
}

impl From<Vec<i64>> for NumericArray {
    fn from(vec: Vec<i64>) -> Self {
        NumericArray::Integer64(vec)
    }
}

impl From<Vec<f64>> for NumericArray {
    fn from(vec: Vec<f64>) -> Self {
        NumericArray::Float64(vec)
    }
}

impl From<Vec<Complex<f64>>> for NumericArray {
    fn from(vec: Vec<Complex<f64>>) -> Self {
        NumericArray::Complex64(vec)
    }
}

impl TryFrom<NumericArray> for Vec<i64> {
    type Error = &'static str;

    fn try_from(value: NumericArray) -> Result<Self, Self::Error> {
        match value {
            NumericArray::Integer64(vec) => Ok(vec),
            _ => Err("NumericArray is not of type Integer64"),
        }
    }
}

impl TryFrom<NumericArray> for Vec<f64> {
    type Error = &'static str;

    fn try_from(value: NumericArray) -> Result<Self, Self::Error> {
        match value {
            NumericArray::Float64(vec) => Ok(vec),
            _ => Err("NumericArray is not of type Float64"),
        }
    }
}

impl TryFrom<NumericArray> for Vec<Complex<f64>> {
    type Error = &'static str;

    fn try_from(value: NumericArray) -> Result<Self, Self::Error> {
        match value {
            NumericArray::Complex64(vec) => Ok(vec),
            _ => Err("NumericArray is not of type Complex64"),
        }
    }
}

impl<T> FromIterator<T> for NumericArray
where
    NumericArray: From<Vec<T>>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<T> = iter.into_iter().collect();
        NumericArray::from(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_abs_at_index() {
        let int_array = NumericArray::Integer64(vec![-1, -2, -3]);
        assert_eq!(int_array.abs_at_index(1), Some(2.0));
        let float_array = NumericArray::Float64(vec![-1.5, -2.5, -3.5]);
        assert_eq!(float_array.abs_at_index(2), Some(3.5));
        let complex_array =
            NumericArray::Complex64(vec![Complex::new(3.0, 4.0), Complex::new(1.0, -1.0)]);
        assert_eq!(complex_array.abs_at_index(0), Some(5.0));
    }

    #[test]
    fn test_linspace_float() {
        // Increasing case
        let arr_float = NumericArray::linspace(0.0, 1.0, 5);
        assert_eq!(
            <Vec<f64>>::try_from(arr_float).unwrap(),
            vec![0.0, 0.25, 0.5, 0.75, 1.0]
        );

        // Decreasing case
        let arr_float = NumericArray::linspace(1.0, -1.0, 11);
        assert_abs_diff_eq!(
            <Vec<f64>>::try_from(arr_float).unwrap().as_slice(),
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0].as_slice(),
        );
    }

    #[test]
    fn test_linspace_complex() {
        // Increasing case
        let arr_complex =
            NumericArray::linspace_complex(Complex::new(0.0, 0.0), Complex::new(1.0, 1.0), 3);
        assert_eq!(
            <Vec<Complex<f64>>>::try_from(arr_complex).unwrap(),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.5, 0.5),
                Complex::new(1.0, 1.0)
            ]
        );

        // Decreasing case
        let arr_complex =
            NumericArray::linspace_complex(Complex::new(1.0, 1.0), Complex::new(-1.0, -1.0), 5);
        assert_eq!(
            <Vec<Complex<f64>>>::try_from(arr_complex).unwrap(),
            vec![
                Complex::new(1.0, 1.0),
                Complex::new(0.5, 0.5),
                Complex::new(0.0, 0.0),
                Complex::new(-0.5, -0.5),
                Complex::new(-1.0, -1.0)
            ]
        );
    }
}
