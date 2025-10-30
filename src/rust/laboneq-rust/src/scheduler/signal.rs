// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::types::{AwgKey, DeviceKind};
use laboneq_scheduler::experiment::types::SignalUid;
use pyo3::prelude::*;

use crate::scheduler::experiment::Signal;
use crate::{
    error::{Error, Result},
    scheduler::py_conversion::{ExperimentBuilder, extract_value},
};
use laboneq_scheduler::experiment::sweep_parameter::SweepParameter;
use laboneq_scheduler::experiment::types::{
    Oscillator, OscillatorKind, OscillatorUid, ParameterUid, RealValue, Value,
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
}

#[pymethods]
impl SignalPy {
    #[new]
    pub fn new(
        uid: String,
        sampling_rate: f64,
        awg_key: i64,
        device: &str,
        oscillator: Option<Py<OscillatorPy>>,
        lo_frequency: Option<Py<PyAny>>,
        voltage_offset: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            uid,
            sampling_rate,
            awg_key,
            device_type: extract_device_kind(device).unwrap(),
            oscillator,
            lo_frequency,
            voltage_offset,
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

/// Convert a `SignalPy` to a `Signal`, registering any parameters in the provided `ExperimentBuilder`.
///
/// The used parameters within the Signal must be register in the `ExperimentBuilder` as they might use
/// parameters that are not yet known to the experiment (e.g. swept calibration fields, derived parameters).
pub fn py_signal_to_signal(
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
            let frequency =
                extract_maybe_parameter::<RealValue>(osc_py.frequency.bind(py), builder)?;
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
            extract_maybe_parameter::<RealValue>(lo_freq.bind(py), builder)
        })
        .transpose()?
        .flatten();
    let voltage_offset = signal_py
        .voltage_offset
        .as_ref()
        .map(|lo_freq| -> Result<_> {
            extract_maybe_parameter::<RealValue>(lo_freq.bind(py), builder)
        })
        .transpose()?
        .flatten();
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

fn extract_maybe_parameter<T: TryFrom<Value>>(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<Option<T>> {
    let py = obj.py();
    if let Ok(parameter) = obj.downcast::<SweepParameterPy>() {
        let parameter = parameter.borrow();
        let uid = ParameterUid(builder.register_uid(&parameter.uid));
        let value: T = Value::ParameterUid(uid)
            .try_into()
            .map_err(|_| Error::new(format!("Failed to convert parameter: {obj}")))?;
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
    let val = extract_value(obj, builder)?;
    let value_out = if let Some(val) = val {
        let value: T = val
            .try_into()
            .map_err(|_| Error::new(format!("Failed to convert parameter: {obj}")))?;
        value
    } else {
        return Ok(None);
    };
    Ok(Some(value_out))
}
