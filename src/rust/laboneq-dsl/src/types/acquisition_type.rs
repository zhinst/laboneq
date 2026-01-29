// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, PartialEq, Copy, Eq)]
pub enum AcquisitionType {
    Raw,
    Integration,
    Discrimination,
    Spectroscopy,
    SpectroscopyIq,
    SpectroscopyPsd,
}

impl AcquisitionType {
    pub fn is_spectroscopy(&self) -> bool {
        matches!(
            self,
            Self::Spectroscopy | Self::SpectroscopyIq | Self::SpectroscopyPsd
        )
    }
}
