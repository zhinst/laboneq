// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_traits::{AsPrimitive, Float};
use std::fmt::Result as FormatterResult;
use std::fmt::{self, Debug, Display, Formatter};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A duration represented with unit type.
///
/// # Type Parameter
/// - `T`: The underlying value (typically a floating point number)
/// - `U`: The unit of the value (should be a zero-sized type)
///
/// # Examples
/// ```rust
/// use laboneq_units::duration::seconds;
///
/// let duration = seconds(1.0); // Create a duration of 1 second
/// ```
#[derive(Clone, Copy)]
pub struct Duration<U, T = f64> {
    value: T,
    unit: U,
}

impl<T: Float, U> PartialEq for Duration<U, T> {
    fn eq(&self, other: &Self) -> bool {
        let a = self.value;
        let b = other.value;
        if a.is_zero() && b.is_zero() {
            true
        } else {
            a == b
        }
    }
}

impl<T: Float, U> Eq for Duration<U, T> {}

impl<U: Float, T> PartialOrd for Duration<T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float, U> Ord for Duration<U, T> {
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

impl<T: Debug, U> Debug for Duration<U, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatterResult {
        f.debug_struct("Duration")
            .field("value", &self.value)
            .field("unit", &std::any::type_name::<T>())
            .finish()
    }
}

impl<T, U> Add for Duration<U, T>
where
    T: Add<Output = T> + Copy,
    U: Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Duration {
            value: self.value + rhs.value,
            unit: self.unit,
        }
    }
}

impl<U, T> Sub for Duration<U, T>
where
    T: Sub<Output = T> + Copy,
    U: Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Duration {
            value: self.value - rhs.value,
            unit: self.unit,
        }
    }
}

impl<U, T> Mul<T> for Duration<U, T>
where
    T: Mul<T, Output = T> + Copy,
    U: Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Duration {
            value: self.value * rhs,
            unit: self.unit,
        }
    }
}

impl<U, T> Neg for Duration<U, T>
where
    T: Neg<Output = T> + Copy,
    U: Copy,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Duration {
            value: -self.value,
            unit: self.unit,
        }
    }
}

impl<U, T> Div<T> for Duration<U, T>
where
    T: Div<T, Output = T> + Copy,
    U: Copy,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Duration {
            value: self.value / rhs,
            unit: self.unit,
        }
    }
}

fn round_to_significant_digits(x: f64, n: u32) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        let order = x.abs().log10().floor();
        let scale = 10f64.powf((n as f64) - 1.0 - order);
        (x * scale).round() / scale
    }
}

impl<U, T> Display for Duration<U, T>
where
    T: Display + Debug + AsPrimitive<f64> + Float,
    U: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            Display::fmt(&self.value, f)?;
        } else {
            // For floats, the debug representation is generally preferable.
            // It automatically chooses precision and enables scientific notation when appropriate.
            // We make this the default.

            // We also round to a number of significand digits slightly below that of epsilon.
            // It avoids ugly numbers in presence of rounding errors, and no one wants to read
            // that many digits anyway.

            let significand_digits = (-T::epsilon().log10() - T::one()).as_() as u32;
            let value = round_to_significant_digits(self.value.as_(), significand_digits);

            Debug::fmt(&value, f)?;
        }
        write!(f, " ")?;
        self.unit.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Seconds;

impl Display for Seconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatterResult {
        write!(f, "s")
    }
}

impl<T: Float, U: Default> From<T> for Duration<U, T> {
    fn from(value: T) -> Self {
        Duration {
            value,
            unit: U::default(),
        }
    }
}

impl<U> From<Duration<U, f64>> for f64 {
    fn from(duration: Duration<U, f64>) -> Self {
        duration.value
    }
}

impl<U> From<Duration<U, f32>> for f32 {
    fn from(duration: Duration<U, f32>) -> Self {
        duration.value
    }
}

pub const fn seconds<T>(value: T) -> Duration<Seconds, T> {
    Duration {
        value,
        unit: Seconds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let duration: Duration<Seconds> = 1e6.into();
        assert_eq!(duration.value, 1e6);

        let duration = Duration::<Seconds>::from(1e-6);
        assert_eq!(duration.value, 1e-6);
    }

    #[test]
    fn test_display() {
        let duration: Duration<Seconds> = 1e-6.into();
        assert_eq!(format!("{duration}"), "1e-6 s");

        let duration: Duration<Seconds> = 1.1500000000000002e-6.into();
        assert_eq!(format!("{duration}"), "1.15e-6 s");
    }

    #[test]
    fn test_eq() {
        assert_eq!(seconds(1e6), seconds(1e6));
        assert_eq!(seconds(0.0), seconds(-0.0));
        assert_ne!(seconds(1e6), seconds(2e6));
        assert_ne!(seconds(1e6), seconds(-1e6));
    }

    #[test]
    fn test_add() {
        assert_eq!(seconds(1e6) + seconds(1e6), seconds(2e6));
        assert_eq!(seconds(1e6) + seconds(-1e6), seconds(0.0));
    }

    #[test]
    fn test_cmp() {
        assert!(seconds(1e6) < seconds(2e6));
        assert!(seconds(1e6) <= seconds(1e6));
        assert!(seconds(2e6) > seconds(1e6));
        assert!(seconds(1e6) >= seconds(1e6));
    }

    #[test]
    fn test_ordering() {
        let mut c = vec![seconds(2e6), seconds(1e6)];
        c.sort();
        assert_eq!(c, vec![seconds(1e6), seconds(2e6)]);
    }
}
