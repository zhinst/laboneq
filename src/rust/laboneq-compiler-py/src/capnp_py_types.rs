// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::types::PyString;
use pyo3::{intern, prelude::*};

#[derive(FromPyObject, Debug)]
pub(crate) struct ExperimentCapnpPy<'py> {
    /// Optional UID of the experiment.
    pub uid: Option<Bound<'py, PyString>>,
    /// Sections of the experiment.
    /// The expected Python type is `laboneq.dsl.experiment.Section`.
    pub sections: Vec<Bound<'py, PyAny>>,
    pub experiment_signals: Vec<ExperimentSignalPy<'py>>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct DeviceSetupCapnpPy<'py> {
    pub setup_description: SetupDescriptionPy<'py>,
}

#[derive(FromPyObject, Debug)]
pub(crate) enum SetupDescriptionPy<'py> {
    Qccs(SetupDescriptionQccsPy<'py>),
    Zqcs(SetupDescriptionZqcsPy<'py>),
}

#[derive(FromPyObject, Debug)]
pub(crate) struct SetupDescriptionQccsPy<'py> {
    pub instruments: Vec<InstrumentPy<'py>>,
    pub signals: Vec<DeviceSignalPy<'py>>,
    pub internal_connections: Vec<InternalConnectionPy<'py>>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct InternalConnectionPy<'py> {
    pub from_instrument: Bound<'py, PyString>,
    pub from_port: Bound<'py, PyString>,
    pub to_instrument: Bound<'py, PyString>,
    pub to_port: Bound<'py, PyString>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct OscillatorPy<'py> {
    pub uid: Bound<'py, PyString>,
    pub frequency: Bound<'py, PyAny>,
    pub modulation: Option<Bound<'py, PyString>>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct InstrumentPy<'py> {
    pub uid: Bound<'py, PyString>,
    pub device_type: Bound<'py, PyString>,
    pub options: Vec<Bound<'py, PyString>>,
    pub reference_clock_source: Option<Bound<'py, PyString>>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct DeviceSignalPy<'py> {
    pub uid: Bound<'py, PyString>,
    pub ports: Vec<Bound<'py, PyString>>,
    pub instrument_uid: Bound<'py, PyString>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct SetupDescriptionZqcsPy<'py> {
    pub uid: Bound<'py, PyString>,
    pub data: Vec<u8>,
    pub channels: Vec<ChannelConfigPy<'py>>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct ChannelConfigPy<'py> {
    pub geolocation: Bound<'py, PyString>,
    pub channel_type: ChannelTypePy,
}

#[derive(Debug)]
pub(crate) enum ChannelTypePy {
    Rf,
    Qa,
    Flux,
}

impl FromPyObject<'_, '_> for ChannelTypePy {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let name: String = obj.getattr("name")?.extract()?;
        match name.as_str().to_ascii_lowercase().as_str() {
            "rf" => Ok(Self::Rf),
            "qa" => Ok(Self::Qa),
            "flux" => Ok(Self::Flux),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown ChannelType variant: {other}"
            ))),
        }
    }
}

#[derive(FromPyObject, Debug)]
pub(crate) struct ExperimentSignalPy<'py> {
    pub uid: Bound<'py, PyString>,
    pub maps_to: Bound<'py, PyString>,

    // Calibration
    pub amplitude: Option<Bound<'py, PyAny>>,
    pub oscillator: Option<OscillatorPy<'py>>,
    pub lo_frequency: Option<Bound<'py, PyAny>>,
    pub voltage_offset: Option<Bound<'py, PyAny>>,
    pub amplifier_pump: Option<AmplifierPumpPy<'py>>,
    pub port_mode: Option<Bound<'py, PyString>>,
    pub automute: bool,
    pub delay_signal: f64,
    pub port_delay: Option<Bound<'py, PyAny>>,
    pub range: Option<QuantityPy>,
    pub precompensation: Option<PrecompensationPy>,
    pub added_outputs: Vec<OutputRoutePy<'py>>,
    pub threshold: Option<Vec<f64>>,
    pub mixer_calibration: Option<MixerCalibrationPy<'py>>,
}

#[derive(Debug)]
pub(crate) enum UnitPy {
    Volt,
    DBm,
}

impl FromPyObject<'_, '_> for UnitPy {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        match obj
            .getattr(intern!(obj.py(), "name"))?
            .extract::<&str>()?
            .to_ascii_lowercase()
            .as_str()
        {
            "volt" => Ok(Self::Volt),
            "dbm" => Ok(Self::DBm),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown Unit variant: {other}"
            ))),
        }
    }
}

#[derive(FromPyObject, Debug)]
pub(crate) struct QuantityPy {
    pub value: f64,
    pub unit: Option<UnitPy>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct MixerCalibrationPy<'py> {
    pub voltage_offsets: Option<Vec<Bound<'py, PyAny>>>,
    pub correction_matrix: Option<Vec<Vec<Bound<'py, PyAny>>>>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct OutputRoutePy<'py> {
    pub source: Bound<'py, PyString>,
    pub amplitude_scaling: Bound<'py, PyAny>,
    pub phase_shift: Bound<'py, PyAny>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct AmplifierPumpPy<'py> {
    pub alc_on: bool,
    pub pump_on: bool,
    pub pump_filter_on: bool,
    pub pump_power: Bound<'py, PyAny>,
    pub pump_frequency: Bound<'py, PyAny>,
    pub probe_on: bool,
    pub probe_power: Bound<'py, PyAny>,
    pub probe_frequency: Bound<'py, PyAny>,
    pub cancellation_on: bool,
    pub cancellation_phase: Bound<'py, PyAny>,
    pub cancellation_attenuation: Bound<'py, PyAny>,
    pub cancellation_source: CancellationSourcePy,
    pub cancellation_source_frequency: Option<f64>,
}

#[derive(Debug)]
pub(crate) enum CancellationSourcePy {
    Internal,
    External,
}

impl FromPyObject<'_, '_> for CancellationSourcePy {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let name: String = obj.getattr("name")?.extract()?;
        match name.as_str().to_ascii_lowercase().as_str() {
            "internal" => Ok(Self::Internal),
            "external" => Ok(Self::External),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown CancellationSource variant: {other}"
            ))),
        }
    }
}

#[derive(FromPyObject, Debug)]
pub(crate) struct PrecompensationPy {
    pub high_pass: Option<HighPassCompensationPy>,
    pub exponential: Option<Vec<ExponentialCompensationPy>>,
    #[pyo3(attribute("FIR"))]
    pub fir: Option<FirCompensationPy>,
    pub bounce: Option<BounceCompensationPy>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct HighPassCompensationPy {
    pub timeconstant: f64,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct ExponentialCompensationPy {
    pub timeconstant: f64,
    pub amplitude: f64,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct FirCompensationPy {
    pub coefficients: Vec<f64>,
}

#[derive(FromPyObject, Debug)]
pub(crate) struct BounceCompensationPy {
    pub delay: f64,
    pub amplitude: f64,
}
