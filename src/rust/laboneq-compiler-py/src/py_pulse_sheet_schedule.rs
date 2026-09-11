// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

use laboneq_ir::pulse_sheet_schedule::PulseSheetSchedule;

/// Opaque carrier for a [`PulseSheetSchedule`] as it passes through the Python compilation steps.
#[pyclass]
pub(crate) struct PulseSheetSchedulePy {
    pub(crate) inner: PulseSheetSchedule,
}

impl PulseSheetSchedulePy {
    pub(crate) fn take_inner(&mut self) -> PulseSheetSchedule {
        std::mem::take(&mut self.inner)
    }
}
