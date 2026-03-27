// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::py_signal::{AmplifierPumpPy, PrecompensationPy};

#[pyclass(name = "DeviceSetupBuilder")]
pub(crate) struct DeviceSetupCapnpBuilderPy {
    pub instruments: Vec<InstrumentPayload>,
    pub signals: Vec<SignalPayload>,
    pub oscillators: Vec<OscillatorPayload>,
}

#[pymethods]
impl DeviceSetupCapnpBuilderPy {
    #[new]
    fn new() -> Self {
        Self {
            instruments: Vec::new(),
            signals: Vec::new(),
            oscillators: Vec::new(),
        }
    }

    #[pyo3(signature = (uid, device_type, physical_device_uid, options=None, reference_clock_source=None, is_shfqc=false))]
    fn add_instrument(
        &mut self,
        uid: Bound<'_, PyString>,
        device_type: Bound<'_, PyString>,
        physical_device_uid: u16,
        options: Option<Vec<Bound<'_, PyString>>>,
        reference_clock_source: Option<Bound<'_, PyString>>,
        is_shfqc: bool,
    ) -> PyResult<()> {
        let payload = InstrumentPayload {
            uid: uid.into(),
            device_type: device_type.into(),
            options: options
                .unwrap_or_default()
                .into_iter()
                .map(|o| o.into())
                .collect(),
            reference_clock_source: reference_clock_source.map(|s| s.into()),
            physical_device_uid,
            is_shfqc,
        };
        self.instruments.push(payload);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        uid,
        ports,
        instrument_uid,
        channel_type,
        awg_core,
        amplitude=None,
        oscillator=None,
        lo_frequency=None,
        voltage_offset=None,
        amplifier_pump=None,
        port_mode=None,
        automute=false,
        signal_delay=0.0,
        port_delay=None,
        range=None,
        precompensation=None,
        added_outputs=vec![],
        threshold=None,
    ))]
    fn add_signal_with_calibration(
        &mut self,
        uid: Py<PyString>,
        ports: Vec<Py<PyString>>,
        instrument_uid: Py<PyString>,
        channel_type: Py<PyString>,
        awg_core: i64,

        // Calibration
        amplitude: Option<Py<PyAny>>,
        oscillator: Option<Bound<'_, OscillatorRef>>,
        lo_frequency: Option<Py<PyAny>>,
        voltage_offset: Option<Py<PyAny>>,
        amplifier_pump: Option<Py<AmplifierPumpPy>>,
        port_mode: Option<Py<PyString>>,
        automute: bool,
        signal_delay: f64,
        port_delay: Option<Py<PyAny>>,
        range: Option<(f64, Option<String>)>,
        precompensation: Option<Py<PrecompensationPy>>,
        added_outputs: Vec<Py<OutputRoutePy>>,
        threshold: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let payload = SignalPayload {
            uid,
            ports,
            instrument_uid,
            channel_type,
            awg_core: awg_core as u64,

            // Calibration
            oscillator_index: oscillator.map(|o| o.borrow().index),
            amplitude,
            lo_frequency,
            voltage_offset,
            amplifier_pump,
            port_mode,
            automute,
            signal_delay,
            port_delay,
            range,
            precompensation,
            added_outputs,
            threshold,
        };
        self.signals.push(payload);
        Ok(())
    }

    fn create_oscillator(
        &mut self,
        uid: Py<PyString>,
        frequency: Py<PyAny>,
        modulation: Option<Py<PyString>>,
    ) -> PyResult<OscillatorRef> {
        let index = self.oscillators.len();
        let payload = OscillatorPayload {
            uid,
            frequency,
            modulation,
        };
        self.oscillators.push(payload);
        Ok(OscillatorRef { index })
    }

    fn create_output_route(
        &self,
        source_signal: Py<PyString>,
        amplitude_scaling: Py<PyAny>,
        phase_shift: Py<PyAny>,
    ) -> PyResult<OutputRoutePy> {
        Ok(OutputRoutePy {
            source_signal,
            amplitude_scaling,
            phase_shift,
        })
    }
}

pub(crate) struct InstrumentPayload {
    pub uid: Py<PyString>,
    pub device_type: Py<PyString>,
    pub options: Vec<Py<PyString>>,
    pub reference_clock_source: Option<Py<PyString>>,
    pub physical_device_uid: u16,
    pub is_shfqc: bool,
}

pub(crate) struct OscillatorPayload {
    pub uid: Py<PyString>,
    pub frequency: Py<PyAny>,
    pub modulation: Option<Py<PyString>>,
}

#[pyclass(name = "OscillatorRef", frozen)]
pub(crate) struct OscillatorRef {
    pub index: usize,
}

pub(crate) struct SignalPayload {
    pub uid: Py<PyString>,
    pub ports: Vec<Py<PyString>>,
    pub instrument_uid: Py<PyString>,
    pub channel_type: Py<PyString>,
    pub awg_core: u64,

    // Calibration
    pub amplitude: Option<Py<PyAny>>,
    pub oscillator_index: Option<usize>,
    pub lo_frequency: Option<Py<PyAny>>,
    pub voltage_offset: Option<Py<PyAny>>,
    pub amplifier_pump: Option<Py<AmplifierPumpPy>>,
    pub port_mode: Option<Py<PyString>>,
    pub automute: bool,
    pub signal_delay: f64,
    pub port_delay: Option<Py<PyAny>>,
    pub range: Option<(f64, Option<String>)>,
    pub precompensation: Option<Py<PrecompensationPy>>,
    pub added_outputs: Vec<Py<OutputRoutePy>>,
    pub threshold: Option<Vec<f64>>,
}

#[pyclass(name = "OutputRoute", frozen)]
pub(crate) struct OutputRoutePy {
    pub source_signal: Py<PyString>,
    pub amplitude_scaling: Py<PyAny>,
    pub phase_shift: Py<PyAny>,
}
