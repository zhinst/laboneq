// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::str::FromStr;

use laboneq_common::types::{AwgKey, DeviceKind};
use laboneq_scheduler::experiment::builders::AmplifierPumpBuilder;
use laboneq_scheduler::experiment::types::{AmplifierPump, SignalUid, ValueOrParameter};
use pyo3::prelude::*;

use crate::scheduler::experiment::{Signal, SignalKind};
use crate::{
    error::{Error, Result},
    scheduler::py_conversion::ExperimentBuilder,
};
use laboneq_scheduler::experiment::sweep_parameter::SweepParameter;
use laboneq_scheduler::experiment::types::{
    Oscillator, OscillatorKind, OscillatorUid, ParameterUid,
};
use numeric_array::NumericArray;

#[pyclass(name = "Signal", frozen)]
pub struct SignalPy {
    pub uid: String,
    pub sampling_rate: f64,
    pub awg_key: i64,
    pub device_type: DeviceKind,
    pub oscillator: Option<Py<OscillatorPy>>,
    pub lo_frequency: Option<Py<PyAny>>,
    pub voltage_offset: Option<Py<PyAny>>,
    pub amplifier_pump: Option<Py<AmplifierPumpPy>>,
    pub kind: SignalKind,
}

#[pymethods]
impl SignalPy {
    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        uid: String,
        sampling_rate: f64,
        awg_key: i64,
        device: &str,
        oscillator: Option<Py<OscillatorPy>>,
        lo_frequency: Option<Py<PyAny>>,
        voltage_offset: Option<Py<PyAny>>,
        amplifier_pump: Option<Py<AmplifierPumpPy>>,
        kind: &str,
    ) -> Self {
        Self {
            uid,
            sampling_rate,
            awg_key,
            device_type: extract_device_kind(device).unwrap(),
            oscillator,
            lo_frequency,
            voltage_offset,
            amplifier_pump,
            kind: SignalKind::from_str(kind).unwrap(),
        }
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
) -> Result<Signal> {
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
    let s = Signal {
        uid: SignalUid(
            // We must insert the signal as Compiler may add dummy signals that do not exist in the experiment.
            builder.id_store.get_or_insert(&signal_py.uid),
        ),
        awg_key: AwgKey(signal_py.awg_key as u64),
        sampling_rate: signal_py.sampling_rate,
        oscillator,
        device_type: signal_py.device_type,
        lo_frequency,
        voltage_offset,
        amplifier_pump,
        kind: signal_py.kind.clone(),
    };
    Ok(s)
}

fn extract_device_kind(device: &str) -> Result<DeviceKind> {
    let kind = match device {
        "HDAWG" => DeviceKind::Hdawg,
        "SHFQA" => DeviceKind::Shfqa,
        "SHFSG" => DeviceKind::Shfsg,
        "UHFQA" => DeviceKind::Uhfqa,
        "PRETTYPRINTERDEVICE" => DeviceKind::PrettyPrinterDevice,
        _ => {
            return Err(Error::new(format!("Unknown device type: {device}")));
        }
    };
    Ok(kind)
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
            .insert(uid, SweepParameter::new(uid, values));
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
