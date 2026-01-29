// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedId;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SectionUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PulseUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct OscillatorUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ParameterUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct HandleUid(pub NamedId);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct DeviceUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SignalUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PulseParameterUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PrngSampleUid(pub NamedId);

#[macro_export]
macro_rules! impl_from_named_id {
    ($t:ty) => {
        impl From<laboneq_common::named_id::NamedId> for $t {
            fn from(value: laboneq_common::named_id::NamedId) -> Self {
                Self(value)
            }
        }

        impl From<$t> for laboneq_common::named_id::NamedId {
            fn from(value: $t) -> Self {
                value.0
            }
        }
    };
}

impl_from_named_id!(SignalUid);
impl_from_named_id!(OscillatorUid);
impl_from_named_id!(ParameterUid);
impl_from_named_id!(HandleUid);
impl_from_named_id!(DeviceUid);
impl_from_named_id!(PulseUid);
impl_from_named_id!(SectionUid);
impl_from_named_id!(PulseParameterUid);
impl_from_named_id!(PrngSampleUid);

/// UID of an external parameter.
///
/// This is used to uniquely identify external parameters in the experiment.
/// The parameter the UID refers to is not accessed by the compiler itself.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct ExternalParameterUid(pub u64);

impl From<u64> for ExternalParameterUid {
    fn from(value: u64) -> Self {
        ExternalParameterUid(value)
    }
}
