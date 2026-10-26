// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Encodes the QCCS setup description from the Python objects assembled by the payload builder.

use capnp::any_pointer;
use capnp::message::TypedBuilder;
use pyo3::PyErr;
use pyo3::prelude::*;
use pyo3::types::PyString;

use laboneq_qccs_capnp::setup::v1::setup_description_capnp as schema;

/// Encodes the Python setup description into a message, ready to be stored in
/// `DeviceSetup.setupDescription`.
pub fn message_from_py(
    setup_description: &Bound<'_, PyAny>,
) -> PyResult<TypedBuilder<any_pointer::Owned>> {
    let description: SetupDescriptionQccsPy<'_> = setup_description.extract()?;

    let mut message = capnp::message::Builder::new_default();
    let mut builder = message.init_root::<schema::setup_description_qccs::Builder<'_>>();
    write_instruments(&description.instruments, &mut builder)?;
    write_device_signals(&description.signals, &mut builder)?;
    write_internal_connections(&description.internal_connections, &mut builder)?;

    Ok(message.into_typed())
}

#[derive(FromPyObject, Debug)]
struct SetupDescriptionQccsPy<'py> {
    instruments: Vec<InstrumentPy<'py>>,
    signals: Vec<DeviceSignalPy<'py>>,
    internal_connections: Vec<InternalConnectionPy<'py>>,
}

#[derive(FromPyObject, Debug)]
struct InstrumentPy<'py> {
    uid: Bound<'py, PyString>,
    device_type: Bound<'py, PyString>,
    options: Vec<Bound<'py, PyString>>,
    reference_clock_source: Option<Bound<'py, PyString>>,
}

#[derive(FromPyObject, Debug)]
struct DeviceSignalPy<'py> {
    uid: Bound<'py, PyString>,
    ports: Vec<Bound<'py, PyString>>,
    instrument_uid: Bound<'py, PyString>,
}

#[derive(FromPyObject, Debug)]
struct InternalConnectionPy<'py> {
    from_instrument: Bound<'py, PyString>,
    from_port: Bound<'py, PyString>,
    to_instrument: Bound<'py, PyString>,
    to_port: Bound<'py, PyString>,
}

fn write_instruments(
    instruments: &[InstrumentPy<'_>],
    builder: &mut schema::setup_description_qccs::Builder<'_>,
) -> PyResult<()> {
    let mut instruments_builder = builder
        .reborrow()
        .init_instruments(instruments.len() as u32);
    for (i, instrument) in instruments.iter().enumerate() {
        let mut instrument_builder = instruments_builder.reborrow().get(i as u32);
        instrument_builder.set_uid(instrument.uid.to_str()?);
        instrument_builder.set_device_type(instrument.device_type.to_str()?);

        let mut options_builder = instrument_builder
            .reborrow()
            .init_options(instrument.options.len() as u32);
        for (j, option) in instrument.options.iter().enumerate() {
            options_builder.set(j as u32, option.to_str()?);
        }

        if let Some(clock_source) = &instrument.reference_clock_source {
            let clock_source = match clock_source.to_str()? {
                "INTERNAL" => schema::ReferenceClock::Internal,
                "EXTERNAL" => schema::ReferenceClock::External,
                other => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid reference clock source: {other}. Expected 'INTERNAL' or 'EXTERNAL'."
                    )));
                }
            };
            instrument_builder.set_reference_clock_source(clock_source);
        }
    }
    Ok(())
}

fn write_device_signals(
    signals: &[DeviceSignalPy<'_>],
    builder: &mut schema::setup_description_qccs::Builder<'_>,
) -> PyResult<()> {
    let mut signals_builder = builder.reborrow().init_signals(signals.len() as u32);
    for (i, signal) in signals.iter().enumerate() {
        let mut signal_builder = signals_builder.reborrow().get(i as u32);
        signal_builder.set_uid(signal.uid.to_str()?);
        signal_builder.set_instrument_uid(signal.instrument_uid.to_str()?);

        let mut ports_builder = signal_builder
            .reborrow()
            .init_ports(signal.ports.len() as u32);
        for (j, port) in signal.ports.iter().enumerate() {
            ports_builder.set(j as u32, port.to_str()?);
        }
    }
    Ok(())
}

fn write_internal_connections(
    connections: &[InternalConnectionPy<'_>],
    builder: &mut schema::setup_description_qccs::Builder<'_>,
) -> PyResult<()> {
    let mut connections_builder = builder
        .reborrow()
        .init_internal_connections(connections.len() as u32);
    for (i, connection) in connections.iter().enumerate() {
        let mut connection_builder = connections_builder.reborrow().get(i as u32);
        connection_builder.set_from_instrument(connection.from_instrument.to_str()?);
        connection_builder.set_from_port(connection.from_port.to_str()?);
        connection_builder.set_to_instrument(connection.to_instrument.to_str()?);
        connection_builder.set_to_port(connection.to_port.to_str()?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use laboneq_common::named_id::NamedIdStore;
    use laboneq_compiler_py::capnp_setup_description::SetupDescriptionPayload;
    use pyo3::ffi::c_str;
    use pyo3::types::PyDict;

    use super::*;
    use crate::{BACKEND_NAME, setup_description_capnp};

    /// The Python setup description the compiler is handed today: attribute access only, so a
    /// `SimpleNamespace` stands in for the real object.
    const SETUP_DESCRIPTION_PY: &CStr = c_str!(
        r#"
from types import SimpleNamespace

setup_description = SimpleNamespace(
    instruments=[
        SimpleNamespace(
            uid="shfqc",
            device_type="SHFQC",
            options=["SHFQC6"],
            reference_clock_source="EXTERNAL",
        ),
        SimpleNamespace(
            uid="ppc", device_type="SHFPPC", options=[], reference_clock_source=None
        ),
    ],
    signals=[
        SimpleNamespace(
            uid="shfqc/measure", ports=["QACHANNELS/0/OUTPUT"], instrument_uid="shfqc"
        ),
    ],
    internal_connections=[
        SimpleNamespace(
            from_instrument="ppc",
            from_port="PPCHANNELS/0",
            to_instrument="shfqc",
            to_port="QACHANNELS/0/INPUT",
        ),
    ],
)
"#
    );

    /// What Python describes is what the compiler ends up compiling against.
    #[test]
    fn setup_description_round_trips_from_python() {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(SETUP_DESCRIPTION_PY, Some(&globals), None).unwrap();
            let setup_description = globals.get_item("setup_description").unwrap().unwrap();

            let message = message_from_py(&setup_description).unwrap();
            let payload = SetupDescriptionPayload::from_pointer(
                BACKEND_NAME,
                message.get_root_as_reader().unwrap(),
            )
            .unwrap();

            let mut id_store = NamedIdStore::new();
            let setup = setup_description_capnp::read(&payload, &mut id_store).unwrap();

            // `SHFPPC` is an auxiliary device, so it is partitioned off the instrument list.
            assert_eq!(setup.instruments().len(), 1);
            assert_eq!(setup.auxiliary_devices().len(), 1);
            assert_eq!(id_store.resolve(setup.instruments()[0].uid), Some("shfqc"));
            assert_eq!(*setup.instruments()[0].options, ["SHFQC6"]);
            assert_eq!(
                id_store.resolve(setup.device_signals[0].uid),
                Some("shfqc/measure")
            );
            assert_eq!(
                setup.internal_connections()[0].to_port,
                "QACHANNELS/0/INPUT"
            );
        });
    }
}
