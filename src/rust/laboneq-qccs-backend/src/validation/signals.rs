// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::SignalKind;

use laboneq_compiler_py::compiler_backend::{ParameterMap, ParameterValues};
use laboneq_dsl::signal_calibration::PortMode;
use laboneq_dsl::types::{AcquisitionType, ValueOrParameter};
use laboneq_error::{WithContext, bail};

use crate::Result;
use crate::device_kind::DeviceKind;
use crate::preprocessor::{QccsBackendPreprocessedData, SignalProperties};

/// Check the calibration of the signals a QCCS experiment plays on.
pub(super) fn validate(
    data: &QccsBackendPreprocessedData,
    parameters: &ParameterMap<'_>,
    acquisition_type: AcquisitionType,
) -> Result<()> {
    for signal in data.signal_properties() {
        validate_freq_sweep_on_acquire_line(&signal, acquisition_type)?;
        check_oscillator_frequency(&signal, parameters)?;
        check_lo_frequency(&signal, parameters)?;
        if signal.calibration.automute {
            check_automute_requirements(&signal)?;
        }
    }
    Ok(())
}

/// Validate local oscillator frequencies.
/// Returning error for values inconsistent w.r.t. the associated device trait.
fn validate_lo_frequency_ranges(lo_frequency: f64, device_traits: &DeviceTraits) -> Result<()> {
    if let Some(min_value) = device_traits.min_lo_frequency()
        && lo_frequency < min_value.value()
    {
        bail!(
            "({}) Local oscillator frequency {} Hz is smaller than minimum {} Hz.",
            device_traits.name,
            lo_frequency,
            min_value.value(),
        );
    }

    if let Some(max_value) = device_traits.max_lo_frequency()
        && lo_frequency > max_value.value()
    {
        bail!(
            "({}) Local oscillator frequency {} Hz is larger than maximum {} Hz.",
            device_traits.name,
            lo_frequency,
            max_value.value(),
        );
    }

    if let Some(granularity) = device_traits.lo_frequency_granularity
        && lo_frequency % granularity.value() != 0.0
    {
        bail!(
            "({}) Local oscillator frequency {} Hz is not multiple of {} Hz.",
            device_traits.name,
            lo_frequency,
            granularity.value(),
        );
    }
    Ok(())
}

fn check_automute_requirements(signal: &SignalProperties) -> Result<()> {
    // Port mode check
    if let Some(port_mode) = signal.calibration.port_mode.as_ref()
        && port_mode != &PortMode::RF
    {
        bail!("Automute can only be applied when with RF port mode.");
    }

    // Signal kind check
    if signal.kind == SignalKind::Integration {
        bail!("Automute can only be applied to output channels.");
    }
    Ok(())
}

fn check_oscillator_frequency(
    signal: &SignalProperties,
    parameters: &ParameterMap<'_>,
) -> Result<()> {
    const MAX_SHF_IF_FREQUENCY: f64 = 1e9;
    if !matches!(signal.device_kind, DeviceKind::Shfqa | DeviceKind::Shfsg) {
        return Ok(());
    }
    let Some(osc) = signal.calibration.oscillator.as_ref() else {
        return Ok(());
    };

    match &osc.frequency {
        ValueOrParameter::Value(freq) | ValueOrParameter::ResolvedParameter { value: freq, .. } => {
            if freq.abs() >= MAX_SHF_IF_FREQUENCY {
                bail!(
                    "Signal '{}': oscillator frequency {:.4} GHz \
                     exceeds the Nyquist limit.",
                    signal.uid.0,
                    freq / 1e9,
                );
            }
        }
        ValueOrParameter::Parameter(param_uid) => {
            if let Some(ParameterValues::Float64(vals)) = parameters.get(param_uid).copied() {
                for v in vals {
                    if v.abs() >= MAX_SHF_IF_FREQUENCY {
                        bail!(
                            "Signal '{}': oscillator frequency sweep contains \
                             a value ({:.4} GHz) that exceeds the Nyquist limit.",
                            signal.uid.0,
                            v / 1e9,
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

fn check_lo_frequency(signal: &SignalProperties, parameters: &ParameterMap<'_>) -> Result<()> {
    const LO_FREQ_RESOLUTION_HZ: f64 = 1e-6;
    const LO_FREQ_STEP_HZ: f64 = 200e6;

    let Some(lo_freq) = (matches!(signal.device_kind, DeviceKind::Shfqa | DeviceKind::Shfsg)
        && !matches!(signal.calibration.port_mode, Some(PortMode::LF)))
    .then_some(signal.calibration.lo_frequency.as_ref())
    .flatten() else {
        return Ok(());
    };

    let mut validation_result: Result<()> = Ok(());
    let invalid_value = match lo_freq {
        ValueOrParameter::Parameter(param_uid) => {
            parameters
                .get(param_uid)
                .copied()
                .and_then(|values| match values {
                    ParameterValues::Float64(vals) => {
                        let mut value = None;
                        for v in vals {
                            validation_result =
                                validate_lo_frequency_ranges(*v, signal.device_traits());
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
            validation_result = validate_lo_frequency_ranges(*value, signal.device_traits());
            ((value % LO_FREQ_STEP_HZ).abs() > LO_FREQ_RESOLUTION_HZ)
                .then_some(value)
                .cloned()
        }
    };

    validation_result.add_context(|| format!("Error on signal line {}", signal.uid.0));

    if let Some(freq) = invalid_value {
        let err_msg = format!(
            "Cannot set local oscillator of signal {} to {:.3} GHz. \
            Only integer multiples of 200 MHz are accepted.",
            signal.uid.0,
            freq / 1e9
        );
        return match validation_result {
            Err(e) => Err(e.with_context(|| err_msg)),
            Ok(_) => bail!("{err_msg}"),
        };
    }
    validation_result
}

fn validate_freq_sweep_on_acquire_line(
    signal: &SignalProperties,
    acquisition_type: AcquisitionType,
) -> Result<()> {
    let Some(osc) = signal.calibration.oscillator.as_ref() else {
        return Ok(());
    };

    let is_hw_osc_sweep = match osc.frequency {
        ValueOrParameter::Parameter(_) | ValueOrParameter::ResolvedParameter { .. } => {
            signal.is_hardware_modulated()
        }
        _ => false,
    };

    if !is_hw_osc_sweep || !signal.device_kind.is_qa_device() {
        return Ok(());
    }

    // SHFQA devices with LRT option support hardware oscillator sweeps
    // in non-spectroscopy acquisition modes.
    if signal.has_lrt {
        return Ok(());
    }

    if !acquisition_type.is_spectroscopy() {
        bail!(
            "Hardware oscillator sweep using oscillator {} on acquire line \
            {} connected to UFHQA or SHFQA device {} \
            requires acquisition type to be set to spectroscopy",
            osc.uid.0,
            signal.uid.0,
            signal.device_uid.0,
        );
    }
    Ok(())
}
