// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::num::Wrapping;
use std::ops::{Add, Mul, Neg, Rem, Sub};

use num_traits::{AsPrimitive, Float, FromPrimitive, PrimInt, Signed};

/// An angle represented as a fixed-point value using a signed integer type.
///
/// The full range of the integer type maps to [-π, π) radians (or [-180°, 180°) for degrees).
/// This provides automatic wrapping on overflow and efficient arithmetic operations.
///
/// The internal representation uses a `Wrapping<T>` to ensure that angle arithmetic
/// automatically wraps around the valid range, making it impossible to have invalid
/// angle values.
///
/// # Type Parameter
/// - `T`: The underlying signed integer type (i8, i16, i32, or i64)
///
/// # Examples
/// ```rust
/// use laboneq_units::Angle64;
///
/// let angle = Angle64::from_degrees(90.0);
/// let doubled = angle * 2.0; // 180°
/// let wrapped = Angle64::from_degrees(450.0); // Same as 90° due to wrapping
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Angle<T>(Wrapping<T>)
where
    T: PrimInt + Signed;

impl<T> Angle<T>
where
    T: PrimInt + Signed + 'static,
{
    fn period<F: Float + FromPrimitive>() -> F {
        let bit_size = T::zero().count_zeros();
        F::from_u128(1 << bit_size).unwrap()
    }

    /// Normalize the value by wrapping to [-0.5, 0.5).
    fn normalize<F: Float>(value: F) -> F {
        let normalized = value.fract();
        if normalized + normalized >= F::one() {
            return normalized - F::one();
        }
        if normalized + normalized < -F::one() {
            return normalized + F::one();
        }
        normalized
    }

    /// Creates an angle from radians. Values outside [-π, π) will be wrapped.
    pub fn from_radians<F>(radians: F) -> Self
    where
        F: Float + AsPrimitive<T> + FromPrimitive,
    {
        let tau: F = F::from_f64(std::f64::consts::TAU).unwrap();
        let normalized = Self::normalize(radians / tau);
        let scaled = normalized * Self::period();
        Self(Wrapping(scaled.round().as_()))
    }

    /// Creates an angle from degrees. Values outside [-180, +180) will be wrapped.
    pub fn from_degrees<F>(degrees: F) -> Self
    where
        F: Float + AsPrimitive<T> + FromPrimitive,
    {
        let normalized = Self::normalize(degrees / F::from::<i16>(360).unwrap());
        let scaled = normalized * Self::period();
        Self(Wrapping(scaled.round().as_()))
    }

    /// Creates an angle from the raw internal representation.
    pub const fn from_raw(raw: T) -> Self {
        Angle(Wrapping(raw))
    }

    /// Returns the angle in radians within the range [-π, π).
    pub fn to_radians<F: Float + FromPrimitive>(self) -> F {
        let tau: F = F::from_f64(std::f64::consts::TAU).unwrap();
        let value = F::from(self.0.0).unwrap();
        let normalized = value / Self::period::<F>();
        normalized * tau
    }

    /// Returns the angle in degrees within the range [-180, +180)
    pub fn to_degrees<F: Float + FromPrimitive>(self) -> F {
        let value = F::from(self.0.0).unwrap();
        let normalized = value / Self::period::<F>();
        normalized * F::from(360).unwrap()
    }

    /// Returns the raw internal representation.
    pub const fn to_raw(self) -> T {
        self.0.0
    }

    /// Zero angle (0 radians, 0 degrees).
    pub fn zero() -> Self {
        Angle(Wrapping(T::zero()))
    }

    /// Sine of the angle.
    pub fn sin<F: Float + FromPrimitive>(self) -> F {
        self.to_radians::<F>().sin()
    }

    /// Cosine of the angle.
    pub fn cos<F: Float + FromPrimitive>(self) -> F {
        self.to_radians::<F>().cos()
    }

    /// Quantize the angle to the given number of bits by rounding.
    pub fn quantize_bits(self, bits: usize) -> Angle<T> {
        let mut inner = self.to_raw();
        let shift = T::zero().count_zeros() as usize - bits;
        inner = inner >> (shift - 1);
        if inner & T::one() == T::one() {
            inner = inner + T::one(); // round up
        }
        inner = inner << (shift - 1);
        Angle::from_raw(inner)
    }
}

impl<T> Add for Angle<T>
where
    T: PrimInt + Signed,
    Wrapping<T>: Add<Output = Wrapping<T>>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl<T> Sub for Angle<T>
where
    T: PrimInt + Signed,
    Wrapping<T>: Sub<Output = Wrapping<T>>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl<T> Neg for Angle<T>
where
    T: PrimInt + Signed,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self(Wrapping(-self.0.0))
    }
}

// Float multiplication implementations
impl<T> Mul<f32> for Angle<T>
where
    T: PrimInt + Signed + AsPrimitive<f32>,
    f32: AsPrimitive<T>,
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let multiplied: f32 = rhs * self.0.0.as_();
        let normalized: f32 = multiplied.rem(Self::period::<f32>());
        Self(Wrapping(normalized.as_()))
    }
}

impl<T> Mul<f64> for Angle<T>
where
    T: PrimInt + Signed + AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        let multiplied: f64 = rhs * self.0.0.as_() / Self::period::<f64>();
        let normalized: f64 = Self::normalize(multiplied) * Self::period::<f64>();

        Self(Wrapping(normalized.as_()))
    }
}

// Integer multiplication implementations
impl Mul<i64> for Angle<i64> {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self(self.0 * Wrapping(rhs))
    }
}

impl Mul<i32> for Angle<i32> {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self::Output {
        Self(self.0 * Wrapping(rhs))
    }
}

impl Mul<i16> for Angle<i16> {
    type Output = Self;

    fn mul(self, rhs: i16) -> Self::Output {
        Self(self.0 * Wrapping(rhs))
    }
}

impl Mul<i8> for Angle<i8> {
    type Output = Self;

    fn mul(self, rhs: i8) -> Self::Output {
        Self(self.0 * Wrapping(rhs))
    }
}

impl<T> fmt::Display for Angle<T>
where
    T: PrimInt + Signed + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "{:.4}", self.to_radians::<f32>())
        } else {
            write!(f, "{:.2}°", self.to_degrees::<f32>())
        }
    }
}

// Type aliases for convenience
pub type Angle64 = Angle<i64>;
pub type Angle32 = Angle<i32>;
pub type Angle16 = Angle<i16>;
pub type Angle8 = Angle<i8>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    macro_rules! assert_approx_eq {
        ($left:expr, $right:expr, $tolerance:expr) => {
            let left = $left;
            let right = $right;
            let tolerance = $tolerance;
            let diff = (left - right).abs();
            if diff >= tolerance {
                panic!(
                    "assertion failed: values are not approximately equal\n  left: {}\n  right: {}\n  difference: {}\n  tolerance: {}",
                    left, right, diff, tolerance
                );
            }
        };
        ($left:expr, $right:expr) => {
            assert_approx_eq!($left, $right, 1e-10);
        };
    }

    #[test]
    fn test_max_value() {
        // check if the internal scaling factor is correct
        assert_eq!(Angle::<i8>::period::<f32>() as u64, 1 << 8);
        assert_eq!(Angle::<i16>::period::<f32>() as u64, 1 << 16);
    }

    #[test]
    fn test_basic_conversions() {
        let angle = Angle64::from_degrees(90.0);
        assert_approx_eq!(angle.to_degrees::<f64>(), 90.0);
        assert_approx_eq!(angle.to_radians::<f64>(), PI / 2.0);
    }

    #[test]
    fn test_wrapping() {
        let angle1 = Angle64::from_degrees(370.0);
        let angle2 = Angle64::from_degrees(10.0);
        assert_approx_eq!(angle1.to_degrees::<f32>(), angle2.to_degrees::<f32>());

        let angle3 = Angle64::from_radians(-PI / 2.0);
        let angle4 = Angle64::from_radians(3.0 * PI / 2.0);
        assert_approx_eq!(angle3.to_radians::<f32>(), angle4.to_radians::<f32>());
    }

    #[test]
    fn test_arithmetic() {
        let a = Angle64::from_degrees(350.0);
        let b = Angle64::from_degrees(20.0);
        let sum = a + b;
        assert_approx_eq!(sum.to_degrees::<f64>(), 10.0);

        let diff = b - a;
        assert_approx_eq!(diff.to_degrees::<f64>(), 30.0);
    }

    #[test]
    fn test_trig() {
        let angle = Angle64::from_degrees(30.0);
        assert_approx_eq!(angle.sin::<f64>(), 0.5);
        assert_approx_eq!(angle.cos::<f64>(), 3.0f64.sqrt() / 2.0);
    }

    #[test]
    fn test_u32_variant() {
        let angle = Angle32::from_degrees(90.0);
        assert_approx_eq!(angle.to_degrees::<f64>(), 90.0, 1e-6); // Less precision for i32
        assert_approx_eq!(angle.to_radians::<f64>(), PI / 2.0, 1e-6);
    }

    #[test]
    fn test_multiplication() {
        // Test multiplying angle by scalar
        let angle = Angle64::from_degrees(30.0);
        let doubled = angle * 2.0f64;
        assert_approx_eq!(doubled.to_degrees::<f64>(), 60.0);

        let halved = angle * 0.5f64;
        assert_approx_eq!(halved.to_degrees::<f64>(), 15.0);

        // Test multiplication with f32
        let angle_f32 = Angle64::from_degrees(45.0);
        let tripled = angle_f32 * 3.0f32;
        assert_approx_eq!(tripled.to_degrees::<f32>(), 135.0, 1e-6);

        // Test wrapping with multiplication
        let angle_wrap = Angle64::from_degrees(120.0);
        let wrapped = angle_wrap * 4.0f64; // 480 degrees should wrap
        let expected = 480.0 % 360.0; // Should be 120.0
        assert_approx_eq!(wrapped.to_degrees::<f64>(), expected);
    }

    #[test]
    fn test_edge_cases() {
        // Test zero angle
        let zero = Angle64::zero();
        assert_approx_eq!(zero.to_degrees::<f64>(), 0.0);
        assert_approx_eq!(zero.to_radians::<f64>(), 0.0);

        // Test multiplication by zero
        let angle = Angle64::from_degrees(45.0);
        let zero: f64 = 0.0;
        let zeroed = angle * zero;
        assert_approx_eq!(zeroed.to_degrees::<f64>(), 0.0);

        // Test negative multiplication
        let positive = Angle64::from_degrees(30.0);
        let negative = positive * -1.0f64;
        assert_approx_eq!(negative.to_degrees::<f64>(), -30.0);

        // Test very small multiplication
        let tiny = positive * 1e-6f64;
        assert_approx_eq!(tiny.to_degrees::<f64>(), 0.0, 1e-3);
    }

    #[test]
    fn test_boundary_values() {
        // Test exact boundary values - PI maps to -180 due to [-π, π) range
        let pi_rad = Angle64::from_radians(PI);
        assert_approx_eq!(pi_rad.to_degrees::<f64>(), -180.0, 1e-6);

        let neg_pi_rad = Angle64::from_radians(-PI);
        assert_approx_eq!(neg_pi_rad.to_degrees::<f64>(), -180.0, 1e-6);

        // Test 180 and -180 degrees - both should map to -180 due to [-180, 180) range
        let deg_180 = Angle64::from_degrees(180.0);
        let deg_neg_180 = Angle64::from_degrees(-180.0);
        assert_approx_eq!(deg_180.to_degrees::<f64>(), -180.0, 1e-6);
        assert_approx_eq!(deg_neg_180.to_degrees::<f64>(), -180.0, 1e-6);

        // Test values just under the boundary
        let almost_pi = Angle64::from_radians(PI - 1e-10);
        assert!(almost_pi.to_degrees::<f64>() > 179.0);
        assert!(almost_pi.to_degrees::<f64>() < 180.0);
    }

    #[test]
    fn test_negative_angles() {
        // Test negative angle creation and operations
        let neg_angle = Angle64::from_degrees(-45.0);
        assert_approx_eq!(neg_angle.to_degrees::<f64>(), -45.0);

        // Test addition of positive and negative
        let pos_angle = Angle64::from_degrees(30.0);
        let result = pos_angle + neg_angle;
        assert_approx_eq!(result.to_degrees::<f64>(), -15.0);

        // Test negation
        let negated = -pos_angle;
        assert_approx_eq!(negated.to_degrees::<f64>(), -30.0);
    }

    #[test]
    fn test_large_values() {
        // Test large input values that require wrapping
        let large_degrees = Angle64::from_degrees(720.0 + 45.0); // Two full rotations plus 45
        assert_approx_eq!(large_degrees.to_degrees::<f64>(), 45.0, 1e-9);

        let large_negative = Angle64::from_degrees(-720.0 - 45.0);
        assert_approx_eq!(large_negative.to_degrees::<f64>(), -45.0, 1e-9);

        // Test large multiplication
        let angle = Angle64::from_degrees(1.0);
        let large_mult = angle * 1000.0f64;
        let expected = 1000.0 % 360.0 - 360.; // Should be -80°
        assert_approx_eq!(large_mult.to_degrees::<f64>(), expected, 1e-9);
    }

    #[test]
    fn test_precision_limits() {
        // Test i32 vs i64 precision differences
        let precise_angle = 1e-5; // Very small angle in degrees

        let angle64 = Angle64::from_degrees(precise_angle);
        let angle32 = Angle32::from_degrees(precise_angle);

        // i64 should handle this better than i32
        let deg64 = angle64.to_degrees::<f64>();
        let deg32 = angle32.to_degrees::<f64>();

        // Both should be small, but we don't assert exact equality due to precision limits
        assert_approx_eq!(deg64, precise_angle, 1e-10 * precise_angle);
        assert_approx_eq!(deg32, precise_angle, 1e-2 * precise_angle); // Lower precision for i32
    }

    #[test]
    fn test_raw_value_consistency() {
        // Test that from_raw(to_raw()) is identity
        let angles = [
            Angle64::from_degrees(0.0),
            Angle64::from_degrees(90.0),
            Angle64::from_degrees(-45.0),
            Angle64::from_degrees(179.999),
        ];

        for angle in angles {
            let raw = angle.to_raw();
            let reconstructed = Angle64::from_raw(raw);
            assert_eq!(angle, reconstructed, "Raw round-trip failed");
        }
    }

    #[test]
    fn test_special_float_values() {
        // Test behavior with NaN - should not panic but result is undefined
        let nan_angle = Angle64::from_degrees(f64::NAN);
        // Just ensure it doesn't panic - the result is implementation-defined
        let _ = nan_angle.to_degrees::<f64>();

        // Test behavior with infinity
        let inf_angle = Angle64::from_degrees(f64::INFINITY);
        let neg_inf_angle = Angle64::from_degrees(f64::NEG_INFINITY);

        // These should normalize to some finite value
        assert!(inf_angle.to_degrees::<f64>().is_finite());
        assert!(neg_inf_angle.to_degrees::<f64>().is_finite());
    }

    #[test]
    fn test_multiplication_edge_cases() {
        let angle = Angle64::from_degrees(45.0);

        // Test multiplication by very large numbers
        let large_mult = angle * 1e6f64;
        assert_eq!(large_mult, Angle64::from_degrees(0.));

        // Test multiplication by very small numbers and back. We loose resolution when the angle
        // is very small, so the tolerance is large.
        let small_mult = angle * 1e-15_f64;
        assert_approx_eq!(
            (small_mult * 1e15_f64 - angle).to_degrees::<f32>(),
            0.0,
            0.1
        );

        // Test multiplication that should result in negative values
        let negative_mult = angle * -3.7f64;
        let result = negative_mult.to_degrees::<f64>();
        assert_approx_eq!(
            (result - (angle.to_degrees::<f64>() * -3.7f64)).rem_euclid(360.),
            0.0
        );
    }

    #[test]
    fn test_chained_operations() {
        // Test complex chains of operations
        let start = Angle64::from_degrees(30.0);

        // Chain: multiply, add, subtract, negate
        let step1 = start * 2.0f64; // 30 * 2 = 60
        let step2 = step1 + Angle64::from_degrees(45.0); // 60 + 45 = 105
        let step3 = step2 - Angle64::from_degrees(15.0); // 105 - 15 = 90
        let result = -step3; // -90

        // Should be: -(((30*2) + 45) - 15) = -(60 + 45 - 15) = -90
        assert_approx_eq!(result.to_degrees::<f64>(), -90.0);
    }

    #[test]
    fn test_type_conversion_consistency() {
        let angle = Angle64::from_degrees(60.0);

        // Test consistency between f32 and f64 operations
        let deg_f64 = angle.to_degrees::<f64>();
        let deg_f32 = angle.to_degrees::<f32>();
        let rad_f64 = angle.to_radians::<f64>();
        let rad_f32 = angle.to_radians::<f32>();

        // Should be approximately equal (within f32 precision)
        assert_approx_eq!(deg_f64, deg_f32 as f64, 1e-6);
        assert_approx_eq!(rad_f64, rad_f32 as f64, 1e-6);

        // Test round-trip consistency
        let from_f64 = Angle64::from_degrees(deg_f64);
        let from_f32 = Angle64::from_degrees(deg_f32 as f64);

        assert_approx_eq!(
            from_f64.to_degrees::<f64>(),
            from_f32.to_degrees::<f64>(),
            1e-6
        );
    }

    #[test]
    fn test_display_formatting() {
        let angles = [
            (Angle64::from_degrees(0.0), "0.00°"),
            (Angle64::from_degrees(90.0), "90.00°"),
            (Angle64::from_degrees(-45.0), "-45.00°"),
            (Angle64::from_degrees(179.999), "180.00°"), // Should round
        ];

        for (angle, expected) in angles {
            let formatted = format!("{angle}");
            assert_eq!(formatted, expected, "Display formatting failed for angle");
        }
    }

    #[test]
    fn test_extreme_precision() {
        // Test with angles very close to zero
        let tiny = Angle64::from_degrees(1e-12);
        assert_approx_eq!(tiny.to_degrees::<f64>(), 0.0, 1e-9); // Should be very small but not necessarily exact

        // Test with angles very close to boundaries
        let almost_boundary = Angle64::from_degrees(179.9999999);
        let boundary_deg = almost_boundary.to_degrees::<f64>();
        assert!(boundary_deg > 179.0 && boundary_deg < 180.0);

        // Test normalization edge case
        let exactly_half = Angle64::from_degrees(180.0);
        assert_approx_eq!(exactly_half.to_degrees::<f64>(), -180.0); // Should be -180
    }

    #[test]
    fn test_associativity_and_commutativity() {
        let a = Angle64::from_degrees(30.0);
        let b = Angle64::from_degrees(45.0);
        let c = Angle64::from_degrees(60.0);

        // Test commutativity: a + b = b + a
        let ab = a + b;
        let ba = b + a;
        assert_eq!(ab, ba, "Addition should be commutative");

        // Test associativity: (a + b) + c = a + (b + c)
        let abc1 = (a + b) + c;
        let abc2 = a + (b + c);
        assert_eq!(abc1, abc2, "Addition should be associative");

        // Test subtraction properties: a - b = a + (-b)
        let sub_direct = a - b;
        let sub_indirect = a + (-b);
        assert_eq!(
            sub_direct, sub_indirect,
            "Subtraction should equal addition of negation"
        );
    }

    #[test]
    fn test_integer_multiplication() {
        let angle = Angle64::from_degrees(30.0);

        // Test multiplication by positive integer
        let doubled = angle * 2i64;
        assert_approx_eq!(doubled.to_degrees::<f64>(), 60.0);

        // Test multiplication by negative integer
        let negated = angle * -1i64;
        assert_approx_eq!(negated.to_degrees::<f64>(), -30.0);

        // Test multiplication by zero
        let zero: i64 = 0;
        let zeroed = angle * zero;
        assert_approx_eq!(zeroed.to_degrees::<f64>(), 0.0);

        // Test with large integer that causes wrapping
        let large = angle * 13i64; // 30 * 13 = 390 degrees -> wraps in signed integer space
        // The result depends on the internal representation and wrapping behavior
        let result_deg = large.to_degrees::<f64>();
        assert_approx_eq!(result_deg, 30.0);

        // Test i32 variant
        let angle32 = Angle32::from_degrees(45.0);
        let tripled32 = angle32 * 3i32;
        assert_approx_eq!(tripled32.to_degrees::<f64>(), 135.0, 1e-6);
    }

    #[test]
    fn test_mixed_multiplication_types() {
        let angle = Angle64::from_degrees(60.0);

        // Test that float and integer multiplication can coexist
        let float_mult = angle * 1.5f64;
        let int_mult = angle * 2i64;

        assert_approx_eq!(float_mult.to_degrees::<f64>(), 90.0);
        assert_approx_eq!(int_mult.to_degrees::<f64>(), 120.0);

        // Test f32 multiplication still works
        let f32_mult = angle * 0.5f32;
        assert_approx_eq!(f32_mult.to_degrees::<f32>(), 30.0, 1e-6);
    }
}
