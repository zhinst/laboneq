// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::quantity;
use std::fmt::{Display, Formatter};

quantity!(Duration);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Second;

impl Display for Second {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "s")
    }
}

pub const fn seconds<T>(value: T) -> Duration<Second, T> {
    Duration {
        value,
        unit: Second,
    }
}

quantity!(Frequency);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Hertz;

impl Display for Hertz {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hz")
    }
}

pub const fn hertz<T>(value: T) -> Frequency<Hertz, T> {
    Frequency { value, unit: Hertz }
}

impl<T> num_traits::Inv for Duration<Second, T>
where
    T: num_traits::Inv<Output = T>,
{
    type Output = Frequency<Hertz, T>;
    fn inv(self) -> Self::Output {
        Frequency {
            value: self.value.inv(),
            unit: Hertz,
        }
    }
}

impl<T> num_traits::Inv for Frequency<Hertz, T>
where
    T: num_traits::Inv<Output = T>,
{
    type Output = Duration<Second, T>;
    fn inv(self) -> Self::Output {
        Duration {
            value: self.value.inv(),
            unit: Second,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let duration: Duration<Second> = 1e6.into();
        assert_eq!(duration.value, 1e6);

        let duration = Duration::<Second>::from(1e-6);
        assert_eq!(duration.value, 1e-6);
    }

    #[test]
    fn test_display() {
        let duration: Duration<Second> = 1e-6.into();
        assert_eq!(format!("{duration}"), "1e-6 s");

        let duration: Duration<Second> = 1.1500000000000002e-6.into();
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
