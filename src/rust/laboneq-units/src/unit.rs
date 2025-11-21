// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub(crate) fn round_to_significant_digits(x: f64, n: u32) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        let order = x.abs().log10().floor();
        let scale = 10f64.powf((n as f64) - 1.0 - order);
        (x * scale).round() / scale
    }
}

#[macro_export]
macro_rules! quantity {
    ($ident:ident) => {
        /// A quantity represented with unit type.
        ///
        /// # Type Parameter
        /// - `T`: The underlying value (typically a floating point number)
        /// - `U`: The unit of the value, i.e. a unit of time. Typically, it is a zero-sized type.
        ///
        /// # Examples
        /// ```rust
        /// use laboneq_units::duration::seconds;
        ///
        /// let duration = seconds(1.0); // Create a duration of 1 second
        #[derive(std::clone::Clone, std::marker::Copy, std::default::Default, core::fmt::Debug)]
        pub struct $ident<U, T = f64> {
            pub(crate) value: T,
            pub(crate) unit: U,
        }

        impl<U, T> $ident<U, T> {
            pub fn value(self) -> T {
                self.value
            }
        }

        impl<T: num_traits::Zero + std::cmp::PartialEq, U> PartialEq for $ident<U, T> {
            fn eq(&self, other: &Self) -> bool {
                let a = &self.value;
                let b = &other.value;
                if a.is_zero() && b.is_zero() {
                    true
                } else {
                    a == b
                }
            }
        }

        impl<T: num_traits::Zero + std::cmp::PartialEq, U> Eq for $ident<U, T> {}

        impl<U: num_traits::Zero + std::cmp::PartialOrd, T> PartialOrd for $ident<T, U> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl<T: std::cmp::PartialOrd + num_traits::Zero, U> Ord for $ident<U, T> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                if self.value < other.value {
                    std::cmp::Ordering::Less
                } else if self.value > other.value {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }
        }

        impl<T, U> std::ops::Add for $ident<U, T>
        where
            T: std::ops::Add<Output = T> + std::marker::Copy,
            U: std::marker::Copy,
        {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                $ident {
                    value: self.value + rhs.value,
                    unit: self.unit,
                }
            }
        }

        impl<U, T> std::ops::Sub for $ident<U, T>
        where
            T: std::ops::Sub<Output = T> + std::marker::Copy,
            U: std::marker::Copy,
        {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                $ident {
                    value: self.value - rhs.value,
                    unit: self.unit,
                }
            }
        }

        impl<U, T> std::ops::Mul<T> for $ident<U, T>
        where
            T: std::ops::Mul<T, Output = T> + std::marker::Copy,
            U: std::marker::Copy,
        {
            type Output = Self;

            fn mul(self, rhs: T) -> Self::Output {
                $ident {
                    value: self.value * rhs,
                    unit: self.unit,
                }
            }
        }

        impl<U, T> std::ops::Neg for $ident<U, T>
        where
            T: std::ops::Neg<Output = T> + std::marker::Copy,
            U: std::marker::Copy,
        {
            type Output = Self;

            fn neg(self) -> Self::Output {
                $ident {
                    value: -self.value,
                    unit: self.unit,
                }
            }
        }

        impl<U, T> std::ops::Div<T> for $ident<U, T>
        where
            T: std::ops::Div<T, Output = T> + std::marker::Copy,
            U: std::marker::Copy,
        {
            type Output = Self;

            fn div(self, rhs: T) -> Self::Output {
                $ident {
                    value: self.value / rhs,
                    unit: self.unit,
                }
            }
        }

        impl<U, T> std::fmt::Display for $ident<U, T>
        where
            T: std::fmt::Display
                + std::fmt::Debug
                + num_traits::AsPrimitive<f64>
                + num_traits::Float,
            U: std::fmt::Display,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if f.alternate() {
                    std::fmt::Display::fmt(&self.value, f)?;
                } else {
                    // For floats, the debug representation is generally preferable.
                    // It automatically chooses precision and enables scientific notation when appropriate.
                    // We make this the default.

                    // We also round to a number of significand digits slightly below that of epsilon.
                    // It avoids ugly numbers in presence of rounding errors, and no one wants to read
                    // that many digits anyway.

                    let significand_digits = (-T::epsilon().log10() - T::one()).as_() as u32;
                    let value = $crate::unit::round_to_significant_digits(
                        self.value.as_(),
                        significand_digits,
                    );

                    std::fmt::Debug::fmt(&value, f)?;
                }
                write!(f, " ")?;
                self.unit.fmt(f)
            }
        }

        impl<T, U> From<T> for $ident<U, T>
        where
            T: num_traits::Num,
            U: std::default::Default,
        {
            fn from(value: T) -> Self {
                $ident {
                    value,
                    unit: U::default(),
                }
            }
        }

        impl<U> From<$ident<U, f64>> for f64 {
            fn from(value: $ident<U, f64>) -> Self {
                value.value
            }
        }

        impl<U> From<$ident<U, f32>> for f32 {
            fn from(value: $ident<U, f32>) -> Self {
                value.value
            }
        }

        impl<U, T> num_traits::Zero for $ident<U, T>
        where
            T: num_traits::Zero + std::marker::Copy,
            U: std::marker::Copy + std::default::Default,
        {
            fn zero() -> Self {
                Self {
                    value: T::zero(),
                    unit: U::default(),
                }
            }

            fn is_zero(&self) -> bool {
                self.value.is_zero()
            }
        }
    };
}
