// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::types::DeviceKind;
use laboneq_scheduler::experiment::types::{AveragingLoop, ValueOrParameter};
use numeric_array::NumericArray;

use crate::error::{Error, Result};
use crate::scheduler::experiment::PortMode;
use crate::scheduler::experiment_validation::{
    ExperimentContext, ParamsContext, ValidationContext,
};
use crate::scheduler::signal_view::SignalView;

/// Validates signals in an [`Experiment`].
///
pub(super) fn validate_experiment_signals<'a>(
    avg_loop: &AveragingLoop,
    ctx: &ExperimentContext<'a>,
    ctx_validator: &mut ValidationContext,
    ctx_params: &mut ParamsContext<'a>,
) -> Result<()> {
    if ctx_validator.signal_check_done.len() >= ctx.signals.len() {
        return Ok(());
    }
    for (signal_uid, signal) in ctx.signals {
        if ctx_validator.signal_check_done.contains(signal_uid) {
            continue;
        }
        freq_sweep_on_acquire_line_requires_spectroscopy_mode(signal, avg_loop)?;
        check_lo_frequency(signal, ctx)?;
        digest_awgs(signal, ctx_params);
        ctx_validator.signal_check_done.push(*signal_uid);
    }
    Ok(())
}

fn check_lo_frequency(signal: &SignalView, ctx: &ExperimentContext) -> Result<()> {
    let Some(lo_freq) = (matches!(signal.device_kind(), DeviceKind::Shfqa | DeviceKind::Shfsg)
        && !matches!(signal.port_mode(), Some(PortMode::LF)))
    .then_some(signal.lo_frequency())
    .flatten() else {
        return Ok(());
    };

    const LO_FREQ_RESOLUTION_HZ: f64 = 1e-6;
    const LO_FREQ_STEP_HZ: f64 = 200e6;
    let invalid_value = match lo_freq {
        ValueOrParameter::Parameter(param_uid) => {
            ctx.parameters.get(param_uid).and_then(|x| match *x.values {
                NumericArray::Float64(ref vals) => vals
                    .iter()
                    .find(|v| (*v % LO_FREQ_STEP_HZ).abs() > LO_FREQ_RESOLUTION_HZ)
                    .cloned(),
                _ => None,
            })
        }
        ValueOrParameter::ResolvedParameter { value, .. } | ValueOrParameter::Value(value) => {
            ((value % LO_FREQ_STEP_HZ).abs() > LO_FREQ_RESOLUTION_HZ)
                .then_some(value)
                .cloned()
        }
    };

    if let Some(freq) = invalid_value {
        let err_msg = format!(
            "Cannot set local oscillator of signal {} to {:.3} GHz. \
            Only integer multiples of 200 MHz are accepted.",
            signal.uid().0,
            freq / 1e9
        );
        return Err(Error::new(&err_msg));
    }
    Ok(())
}

fn freq_sweep_on_acquire_line_requires_spectroscopy_mode(
    signal: &SignalView,
    avg_loop: &AveragingLoop,
) -> Result<()> {
    let Some(osc) = signal.oscillator() else {
        return Ok(());
    };

    let is_invalid_osc = match osc.frequency {
        ValueOrParameter::Parameter(_) | ValueOrParameter::ResolvedParameter { .. } => {
            signal.is_hardware_modulated() && !avg_loop.acquisition_type.is_spectroscopy()
        }
        _ => false,
    };

    if is_invalid_osc && signal.device_kind().is_qa_device() {
        let err_msg = format!(
            "Hardware oscillator sweep using oscillator {} on acquire line \
            {} connected to UFHQA or SHFQA device {} \
            requires acquisition type to be set to spectroscopy",
            osc.uid.0,
            signal.uid().0,
            signal.device_uid().0,
        );
        return Err(Error::new(&err_msg));
    }
    Ok(())
}

fn digest_awgs<'a>(signal: &'a SignalView, ctx_params: &mut ParamsContext<'a>) {
    if signal.automute() && matches!(signal.device_kind(), DeviceKind::Shfqa) {
        ctx_params
            .awgs_with_automute
            .insert(signal.awg_key(), signal);
    }
}
