// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::{DeviceKind, SignalKind};
use laboneq_dsl::operation::AveragingLoop;
use laboneq_dsl::signal_calibration::PortMode;
use laboneq_dsl::types::ValueOrParameter;
use numeric_array::NumericArray;

use crate::error::{Error, Result};
use crate::experiment_validation::{ExperimentContext, ParamsContext, ValidationContext};
use crate::signal_view::SignalView;

/// Validates signals in an [`Experiment`].
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
        validate_freq_sweep_on_acquire_line(signal, avg_loop)?;
        check_lo_frequency(signal, ctx)?;
        if signal.automute() {
            check_automute_requirements(signal)?;
        }
        digest_awgs(signal, ctx_params);
        ctx_validator.signal_check_done.push(*signal_uid);
    }
    Ok(())
}

/// Validate local oscillator frequencies.
/// Returning error for values inconsistent w.r.t. the associated device trait.
fn validate_lo_frequency_ranges(lo_frequency: f64, device_kind: &DeviceKind) -> Result<()> {
    let device_traits = DeviceTraits::from_device_kind(device_kind);
    if let Some(min_value) = device_traits.min_lo_frequency()
        && lo_frequency < min_value.value()
    {
        let msg = format!(
            "({:?}) Local oscillator frequency {} Hz is smaller than minimum {} Hz.",
            device_kind,
            lo_frequency,
            min_value.value(),
        );
        return Err(Error::new(&msg));
    }

    if let Some(max_value) = device_traits.max_lo_frequency()
        && lo_frequency > max_value.value()
    {
        let msg = format!(
            "({:?}) Local oscillator frequency {} Hz is larger than maximum {} Hz.",
            device_kind,
            lo_frequency,
            max_value.value(),
        );
        return Err(Error::new(&msg));
    }

    if let Some(granularity) = device_traits.lo_frequency_granularity
        && lo_frequency % granularity.value() != 0.0
    {
        let msg = format!(
            "({:?}) Local oscillator frequency {} Hz (device {:?}) is not multiple of {} Hz.",
            device_kind,
            lo_frequency,
            device_kind,
            granularity.value(),
        );
        return Err(Error::new(&msg));
    }
    Ok(())
}

fn check_automute_requirements(signal: &SignalView) -> Result<()> {
    // Device support check
    if !matches!(signal.device_kind(), DeviceKind::Shfqa | DeviceKind::Shfsg) {
        let err_msg = format!(
            "Automute is not available on device '{}'",
            signal.device_kind()
        );
        return Err(Error::new(&err_msg));
    }

    // Port mode check
    if let Some(port_mode) = signal.port_mode()
        && port_mode != &PortMode::RF
    {
        return Err(Error::new(
            "Automute can only be applied when with RF port mode.",
        ));
    }

    // Signal kind check
    if signal.signal_kind() == &SignalKind::Integration {
        return Err(Error::new(
            "Automute can only be applied to output channels.",
        ));
    }
    Ok(())
}

fn check_lo_frequency(signal: &SignalView, ctx: &ExperimentContext) -> Result<()> {
    const LO_FREQ_RESOLUTION_HZ: f64 = 1e-6;
    const LO_FREQ_STEP_HZ: f64 = 200e6;

    let Some(lo_freq) = (matches!(signal.device_kind(), DeviceKind::Shfqa | DeviceKind::Shfsg)
        && !matches!(signal.port_mode(), Some(PortMode::LF)))
    .then_some(signal.lo_frequency())
    .flatten() else {
        return Ok(());
    };

    let mut validation_result: Result<()> = Ok(());
    let invalid_value = match lo_freq {
        ValueOrParameter::Parameter(param_uid) => {
            ctx.parameters.get(param_uid).and_then(|x| match *x.values {
                NumericArray::Float64(ref vals) => {
                    let mut value = None;
                    for v in vals {
                        validation_result = validate_lo_frequency_ranges(*v, &signal.device_kind());
                        if (*v % LO_FREQ_STEP_HZ).abs() > LO_FREQ_RESOLUTION_HZ {
                            value = Some(*v);
                            break;
                        }
                        if validation_result.is_err() {
                            break;
                        }
                    }
                    value
                }
                _ => None,
            })
        }
        ValueOrParameter::ResolvedParameter { value, .. } | ValueOrParameter::Value(value) => {
            validation_result = validate_lo_frequency_ranges(*value, &signal.device_kind());
            ((value % LO_FREQ_STEP_HZ).abs() > LO_FREQ_RESOLUTION_HZ)
                .then_some(value)
                .cloned()
        }
    };

    validation_result = validation_result
        .map_err(|e| Error::new(format!("Error on signal line {}: {}", signal.uid().0, e)));

    if let Some(freq) = invalid_value {
        let err_msg = format!(
            "Cannot set local oscillator of signal {} to {:.3} GHz. \
            Only integer multiples of 200 MHz are accepted.",
            signal.uid().0,
            freq / 1e9
        );
        return match validation_result {
            Err(e) => Err(e).with_context(|| err_msg)?,
            Ok(_) => Err(Error::new(&err_msg)),
        };
    }
    validation_result
}

fn validate_freq_sweep_on_acquire_line(
    signal: &SignalView,
    avg_loop: &AveragingLoop,
) -> Result<()> {
    let Some(osc) = signal.oscillator() else {
        return Ok(());
    };

    let is_hw_osc_sweep = match osc.frequency {
        ValueOrParameter::Parameter(_) | ValueOrParameter::ResolvedParameter { .. } => {
            signal.is_hardware_modulated()
        }
        _ => false,
    };

    if !is_hw_osc_sweep || !signal.device_kind().is_qa_device() {
        return Ok(());
    }

    // SHFQA devices with LRT option support hardware oscillator sweeps
    // in non-spectroscopy acquisition modes.
    if signal.has_option("LRT") {
        return Ok(());
    }

    if !avg_loop.acquisition_type.is_spectroscopy() {
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
