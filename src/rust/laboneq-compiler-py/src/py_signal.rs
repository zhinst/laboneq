// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::str::FromStr;

use laboneq_common::types::AwgKey;
use laboneq_dsl::{
    signal_calibration::{
        BounceCompensation, ExponentialCompensation, FirCompensation, HighPassCompensation,
        Precompensation,
    },
    types::{
        AmplifierPump, AmplifierPumpBuilder, Oscillator, OscillatorKind, OscillatorUid,
        ParameterUid, Quantity, SignalUid, SweepParameter, Unit, ValueOrParameter,
    },
};
use laboneq_ir::signal::{OutputRoute, PortMode, SignalKind};
use laboneq_units::duration::seconds;
use pyo3::prelude::*;

use crate::{
    error::{Error, Result},
    py_conversion::ExperimentBuilder,
    signal_properties::SignalProperties,
};
use numeric_array::NumericArray;

/// Python representation of a Signal, used for constructing the IR Signal from Python.
///
/// This class is a temporary, intermediate representation of signal properties required by
/// the compiler. It is not intended to be used directly by users, but rather as a bridge between the Python API and the Rust IR.
#[pyclass(name = "Signal", frozen)]
pub struct SignalPy {
    pub uid: String,
    pub awg_key: i64,
    pub device_uid: String,
    pub oscillator: Option<Py<OscillatorPy>>,
    pub lo_frequency: Option<Py<PyAny>>,
    pub voltage_offset: Option<Py<PyAny>>,
    pub amplifier_pump: Option<Py<AmplifierPumpPy>>,
    pub kind: SignalKind,
    pub channels: Vec<u16>,
    pub port_mode: Option<PortMode>,
    pub automute: bool,
    pub signal_delay: f64,
    /// Port delay.
    /// Either a float or a sweep parameter
    pub port_delay: Py<PyAny>,
    pub range: Option<Quantity>,
    pub precompensation: Option<Precompensation>,
    pub added_outputs: Vec<Py<OutputRoutePy>>,
}

#[pymethods]
impl SignalPy {
    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        py: Python,
        uid: String,
        awg_key: i64,
        device_uid: &str,
        oscillator: Option<Py<OscillatorPy>>,
        lo_frequency: Option<Py<PyAny>>,
        voltage_offset: Option<Py<PyAny>>,
        amplifier_pump: Option<Py<AmplifierPumpPy>>,
        kind: &str,
        channels: Vec<u16>,
        port_mode: Option<&str>,
        automute: bool,
        signal_delay: f64,
        port_delay: Py<PyAny>,
        // Tuple of (value, unit), (e.g. (1.0, 'volt')) or None
        range: Option<(f64, Option<String>)>,
        precompensation: Option<Py<PrecompensationPy>>,
        added_outputs: Vec<Py<OutputRoutePy>>,
    ) -> Result<Self> {
        let range = if let Some((value, unit)) = range {
            let unit = if let Some(unit) = unit {
                match unit.to_lowercase().as_str() {
                    "volt" => Some(Unit::Volt),
                    "dbm" => Some(Unit::Dbm),
                    _ => {
                        return Err(Error::new(format!(
                            "Unsupported unit '{}' for signal range. Supported units are 'volt' and 'dbm'.",
                            unit
                        )));
                    }
                }
            } else {
                None
            };
            Some(Quantity { value, unit })
        } else {
            None
        };

        Ok(Self {
            uid,
            awg_key,
            device_uid: device_uid.to_string(),
            oscillator,
            lo_frequency,
            voltage_offset,
            amplifier_pump,
            kind: SignalKind::from_str(kind).unwrap(),
            channels,
            port_mode: port_mode.and_then(|s| PortMode::from_str(s).ok()),
            automute,
            signal_delay,
            port_delay,
            range,
            precompensation: precompensation
                .map(|p| extract_precompensation(py, &p.bind(py).borrow())),
            added_outputs,
        })
    }
}

#[pyclass(name = "Oscillator", frozen)]
pub struct OscillatorPy {
    uid: String,
    frequency: Py<PyAny>,
    is_hardware: bool,
}

#[pymethods]
impl OscillatorPy {
    #[new]
    pub fn new(uid: String, frequency: Py<PyAny>, is_hardware: bool) -> Self {
        Self {
            uid,
            frequency,
            is_hardware,
        }
    }
}

#[pyclass(name = "SweepParameter", frozen)]
pub struct SweepParameterPy {
    pub uid: String,
    pub values: Py<PyAny>,
    pub driven_by: Vec<String>,
}

#[pymethods]
impl SweepParameterPy {
    #[new]
    pub fn new(uid: String, values: Py<PyAny>, driven_by: Vec<String>) -> Self {
        Self {
            uid,
            values,
            driven_by,
        }
    }
}

#[pyclass(name = "AmplifierPump", frozen)]
pub struct AmplifierPumpPy {
    device: String,
    channel: u16,
    pump_power: Py<PyAny>,
    pump_frequency: Py<PyAny>,
    probe_power: Py<PyAny>,
    probe_frequency: Py<PyAny>,
    cancellation_phase: Py<PyAny>,
    cancellation_attenuation: Py<PyAny>,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl AmplifierPumpPy {
    #[new]
    pub fn new(
        device: String,
        channel: u16,
        pump_power: Py<PyAny>,
        pump_frequency: Py<PyAny>,
        probe_power: Py<PyAny>,
        probe_frequency: Py<PyAny>,
        cancellation_phase: Py<PyAny>,
        cancellation_attenuation: Py<PyAny>,
    ) -> Self {
        Self {
            device,
            channel,
            pump_power,
            pump_frequency,
            probe_power,
            probe_frequency,
            cancellation_phase,
            cancellation_attenuation,
        }
    }
}

/// Convert a `SignalPy` to a `Signal`, registering any parameters in the provided `ExperimentBuilder`.
///
/// The used parameters within the Signal must be register in the `ExperimentBuilder` as they might use
/// parameters that are not yet known to the experiment (e.g. swept calibration fields, derived parameters).
pub(super) fn py_signal_to_signal(
    py: Python,
    signal_py: &SignalPy,
    builder: &mut ExperimentBuilder,
) -> Result<SignalProperties> {
    let oscillator = signal_py
        .oscillator
        .as_ref()
        .map(|py_osc| {
            let osc_py = py_osc.bind(py).borrow();
            if osc_py.frequency.is_none(py) {
                return Err(Error::new(format!(
                    "Undefined oscillator frequency on signal '{}'",
                    signal_py.uid
                )));
            }
            let frequency = extract_value_or_parameter::<f64>(osc_py.frequency.bind(py), builder)?;
            Ok(Oscillator {
                uid: OscillatorUid(builder.id_store.get_or_insert(&osc_py.uid)),
                frequency: frequency.unwrap(),
                kind: if osc_py.is_hardware {
                    OscillatorKind::Hardware
                } else {
                    OscillatorKind::Software
                },
            })
        })
        .transpose()?;
    let lo_frequency = signal_py
        .lo_frequency
        .as_ref()
        .map(|lo_freq| -> Result<_> {
            extract_value_or_parameter::<f64>(lo_freq.bind(py), builder)
        })
        .transpose()?
        .flatten();
    let voltage_offset = signal_py
        .voltage_offset
        .as_ref()
        .map(|lo_freq| -> Result<_> {
            extract_value_or_parameter::<f64>(lo_freq.bind(py), builder)
        })
        .transpose()?
        .flatten();
    let amplifier_pump = extract_amplifier_pump(py, signal_py, builder)?;
    let port_delay = if let Some(port_delay) =
        extract_value_or_parameter::<f64>(signal_py.port_delay.bind(py), builder)?
    {
        // convert ValueOrParameter<f64> to ValueOrParameter<Duration>
        match port_delay {
            ValueOrParameter::Value(v) => ValueOrParameter::Value(seconds(v)),
            ValueOrParameter::Parameter(p_uid) => ValueOrParameter::Parameter(p_uid),
            _ => unreachable!(),
        }
    } else {
        ValueOrParameter::Value(seconds(0.0))
    };
    let s = SignalProperties {
        uid: SignalUid(
            // We must insert the signal as Compiler may add dummy signals that do not exist in the experiment.
            builder.id_store.get_or_insert(&signal_py.uid),
        ),
        awg_key: AwgKey(signal_py.awg_key as u64),
        oscillator,
        device_uid: builder.id_store.get_or_insert(&signal_py.device_uid).into(),
        lo_frequency,
        voltage_offset,
        amplifier_pump,
        kind: signal_py.kind.clone(),
        channels: signal_py.channels.clone().into(),
        automute: signal_py.automute,
        port_mode: signal_py.port_mode.clone(),
        signal_delay: signal_py.signal_delay.into(),
        port_delay,
        range: signal_py.range.clone(),
        precompensation: signal_py.precompensation.clone(),
        added_outputs: signal_py
            .added_outputs
            .iter()
            .map(|o| extract_output_route(py, o.bind(py), builder).unwrap())
            .collect(),
    };
    Ok(s)
}

fn extract_value_or_parameter<'py, T: FromPyObjectOwned<'py, Error = PyErr>>(
    obj: &Bound<'py, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<Option<ValueOrParameter<T>>> {
    if obj.is_none() {
        return Ok(None);
    }
    let py: Python<'_> = obj.py();
    if let Ok(parameter) = obj.cast::<SweepParameterPy>() {
        let parameter = parameter.borrow();
        let uid = ParameterUid(builder.register_uid(&parameter.uid));
        let value: ValueOrParameter<T> = ValueOrParameter::Parameter(uid);
        if builder.parameters.contains_key(&uid) {
            return Ok(Some(value));
        }
        let values = NumericArray::from_py(parameter.values.bind(py))?;
        builder
            .parameters
            .insert(uid, SweepParameter::new(uid, values).map_err(Error::new)?);
        parameter.driven_by.iter().for_each(|p_uid| {
            let p_uid = ParameterUid(builder.register_uid(p_uid));
            builder
                .driving_parameters
                .entry(p_uid)
                .or_default()
                .insert(uid);
        });
        return Ok(Some(value));
    }
    let value = ValueOrParameter::<T>::Value(obj.extract::<T>()?);
    Ok(Some(value))
}

pub(super) fn extract_amplifier_pump(
    py: Python,
    signal_py: &SignalPy,
    builder: &mut ExperimentBuilder,
) -> Result<Option<AmplifierPump>> {
    if let Some(pump) = signal_py.amplifier_pump.as_ref() {
        let pump = pump.bind(py).borrow();
        let mut amplifier_pump_builder =
            AmplifierPumpBuilder::new(builder.register_uid(&pump.device).into(), pump.channel);
        if let Some(pump_power) =
            extract_value_or_parameter::<f64>(pump.pump_power.bind(py), builder)?
        {
            amplifier_pump_builder = amplifier_pump_builder.pump_power(pump_power);
        }
        if let Some(pump_frequency) =
            extract_value_or_parameter::<f64>(pump.pump_frequency.bind(py), builder)?
        {
            amplifier_pump_builder = amplifier_pump_builder.pump_frequency(pump_frequency);
        }
        if let Some(probe_power) =
            extract_value_or_parameter::<f64>(pump.probe_power.bind(py), builder)?
        {
            amplifier_pump_builder = amplifier_pump_builder.probe_power(probe_power);
        }
        if let Some(probe_frequency) =
            extract_value_or_parameter::<f64>(pump.probe_frequency.bind(py), builder)?
        {
            amplifier_pump_builder = amplifier_pump_builder.probe_frequency(probe_frequency);
        }
        if let Some(cancellation_phase) =
            extract_value_or_parameter::<f64>(pump.cancellation_phase.bind(py), builder)?
        {
            amplifier_pump_builder = amplifier_pump_builder.cancellation_phase(cancellation_phase);
        }
        if let Some(cancellation_attenuation) =
            extract_value_or_parameter::<f64>(pump.cancellation_attenuation.bind(py), builder)?
        {
            amplifier_pump_builder =
                amplifier_pump_builder.cancellation_attenuation(cancellation_attenuation);
        }
        let amplifier_pump = amplifier_pump_builder.build();
        return Ok(Some(amplifier_pump));
    }
    Ok(None)
}

#[pyclass(name = "Precompensation", frozen)]
pub struct PrecompensationPy {
    #[pyo3(get)]
    pub high_pass: Option<Py<HighPassCompensationPy>>,
    #[pyo3(get)]
    pub exponential: Vec<Py<ExponentialCompensationPy>>,
    #[pyo3(get)]
    pub fir: Option<Py<FirCompensationPy>>,
    #[pyo3(get)]
    pub bounce: Option<Py<BounceCompensationPy>>,
}

#[pymethods]
impl PrecompensationPy {
    #[new]
    pub fn new(
        high_pass: Option<Py<HighPassCompensationPy>>,
        exponential: Vec<Py<ExponentialCompensationPy>>,
        fir: Option<Py<FirCompensationPy>>,
        bounce: Option<Py<BounceCompensationPy>>,
    ) -> Self {
        Self {
            high_pass,
            exponential,
            fir,
            bounce,
        }
    }
}

#[pyclass(name = "HighPassCompensation", frozen)]
pub struct HighPassCompensationPy {
    #[pyo3(get)]
    pub timeconstant: f64,
}

#[pymethods]
impl HighPassCompensationPy {
    #[new]
    pub fn new(timeconstant: f64) -> Self {
        Self { timeconstant }
    }
}

#[pyclass(name = "ExponentialCompensation", frozen)]
pub struct ExponentialCompensationPy {
    #[pyo3(get)]
    pub timeconstant: f64,
    #[pyo3(get)]
    pub amplitude: f64,
}

#[pymethods]
impl ExponentialCompensationPy {
    #[new]
    pub fn new(timeconstant: f64, amplitude: f64) -> Self {
        Self {
            timeconstant,
            amplitude,
        }
    }
}

#[pyclass(name = "FirCompensation", frozen)]
pub struct FirCompensationPy {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
}

#[pymethods]
impl FirCompensationPy {
    #[new]
    pub fn new(coefficients: Vec<f64>) -> Self {
        Self { coefficients }
    }
}

#[pyclass(name = "BounceCompensation", frozen)]
pub struct BounceCompensationPy {
    #[pyo3(get)]
    pub delay: f64,
    #[pyo3(get)]
    pub amplitude: f64,
}

#[pymethods]
impl BounceCompensationPy {
    #[new]
    pub fn new(delay: f64, amplitude: f64) -> Self {
        Self { delay, amplitude }
    }
}

fn extract_precompensation(py: Python, precompensation: &PrecompensationPy) -> Precompensation {
    let bounce_py = precompensation.bounce.as_ref().map(|b| b.bind(py).borrow());
    let bounce = bounce_py.as_ref().map(|b| BounceCompensation {
        delay: b.delay,
        amplitude: b.amplitude,
    });

    let exponential = precompensation
        .exponential
        .iter()
        .map(|e| ExponentialCompensation {
            timeconstant: e.bind(py).borrow().timeconstant,
            amplitude: e.bind(py).borrow().amplitude,
        })
        .collect::<Vec<_>>();

    let fir_py = precompensation.fir.as_ref().map(|f| f.bind(py).borrow());
    let fir = fir_py.as_ref().map(|f| FirCompensation {
        coefficients: f.coefficients.clone(),
    });

    let high_pass_py = precompensation
        .high_pass
        .as_ref()
        .map(|h| h.bind(py).borrow());
    let high_pass = high_pass_py.as_ref().map(|h| HighPassCompensation {
        timeconstant: h.timeconstant,
    });
    Precompensation {
        bounce,
        exponential,
        fir,
        high_pass,
    }
}

#[pyclass(name = "OutputRoute", frozen)]
pub struct OutputRoutePy {
    pub source_channel: u16,
    pub amplitude_scaling: Py<PyAny>,
    pub phase_shift: Py<PyAny>,
}

#[pymethods]
impl OutputRoutePy {
    #[new]
    pub fn new(source_channel: u16, amplitude_scaling: Py<PyAny>, phase_shift: Py<PyAny>) -> Self {
        Self {
            source_channel,
            amplitude_scaling,
            phase_shift,
        }
    }
}

fn extract_output_route(
    py: Python,
    output_route: &Bound<'_, OutputRoutePy>,
    builder: &mut ExperimentBuilder,
) -> Result<OutputRoute> {
    let output_route = output_route.borrow();
    Ok(OutputRoute {
        source_channel: output_route.source_channel,
        amplitude_scaling: extract_value_or_parameter::<f64>(
            output_route.amplitude_scaling.bind(py),
            builder,
        )?,
        phase_shift: extract_value_or_parameter::<f64>(output_route.phase_shift.bind(py), builder)?,
    })
}
