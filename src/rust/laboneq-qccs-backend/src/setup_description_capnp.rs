// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Decodes the QCCS setup description.
//!
//! It reaches the compiler as an opaque `DeviceSetup.setupDescription` pointer
//! ([`SetupDescriptionPayload`]); this module is the only place that knows the schema behind
//! it. [`crate::setup_description_py`] encodes it on the way in.

use capnp::Error as CapnpError;

use crate::setup_description::ReferenceClock;
use laboneq_common::device_options::DeviceOptions;
use laboneq_common::named_id::NamedIdStore;
use laboneq_compiler_py::capnp_setup_description::SetupDescriptionPayload;
use laboneq_error::laboneq_error;
use laboneq_qccs_capnp::setup::v1::setup_description_capnp as schema;

use crate::setup_description::{
    Instrument, InternalConnection, PhysicalChannel, SetupDescriptionQccs,
};
use crate::{BACKEND_NAME, Result};

/// Decodes the payload into the QCCS setup description, interning its UIDs into `id_store`.
pub(crate) fn read(
    payload: &SetupDescriptionPayload,
    id_store: &mut NamedIdStore,
) -> Result<SetupDescriptionQccs> {
    let reader = payload.read_as::<schema::setup_description_qccs::Owned>(BACKEND_NAME)?;
    read_setup_description(reader, id_store)
        .map_err(|e| laboneq_error!("Invalid QCCS setup description: {e}"))
}

fn read_setup_description(
    reader: schema::setup_description_qccs::Reader<'_>,
    id_store: &mut NamedIdStore,
) -> capnp::Result<SetupDescriptionQccs> {
    let instruments = reader
        .get_instruments()?
        .iter()
        .map(|instrument| read_instrument(instrument, id_store))
        .collect::<capnp::Result<Vec<_>>>()?;
    let device_signals = read_signals(&reader, id_store)?;
    let internal_connections = read_internal_connections(&reader, id_store)?;

    Ok(SetupDescriptionQccs::new(
        instruments,
        device_signals,
        internal_connections,
    ))
}

fn read_instrument(
    reader: schema::instrument::Reader<'_>,
    id_store: &mut NamedIdStore,
) -> capnp::Result<Instrument> {
    let uid = reader.get_uid()?.to_str()?;
    let device_type = reader.get_device_type()?.to_str()?;
    let options = reader
        .get_options()?
        .iter()
        .map(|option| Ok(option?.to_str()?.to_string()))
        .collect::<capnp::Result<Vec<_>>>()?;

    Ok(Instrument {
        uid: id_store.get_or_insert(uid).into(),
        kind: device_type
            .parse()
            .map_err(|_| CapnpError::failed(format!("Unknown device type: {device_type}")))?,
        options: DeviceOptions::new(options),
        reference_clock: match reader.get_reference_clock_source()? {
            schema::ReferenceClock::External => Some(ReferenceClock::External),
            schema::ReferenceClock::Internal => Some(ReferenceClock::Internal),
            schema::ReferenceClock::Unspecified => None,
        },
    })
}

fn read_signals(
    reader: &schema::setup_description_qccs::Reader<'_>,
    id_store: &mut NamedIdStore,
) -> capnp::Result<Vec<PhysicalChannel>> {
    reader
        .get_signals()?
        .iter()
        .map(|signal| {
            let uid = signal.get_uid()?.to_str()?;
            let device_uid = signal.get_instrument_uid()?.to_str()?;
            let ports = signal
                .get_ports()?
                .iter()
                .map(|port| Ok(port?.to_str()?.to_string()))
                .collect::<capnp::Result<Vec<_>>>()?;

            Ok(PhysicalChannel {
                uid: id_store.get_or_insert(uid).into(),
                device_uid: id_store.get_or_insert(device_uid).into(),
                ports,
            })
        })
        .collect()
}

fn read_internal_connections(
    reader: &schema::setup_description_qccs::Reader<'_>,
    id_store: &mut NamedIdStore,
) -> capnp::Result<Vec<InternalConnection>> {
    reader
        .get_internal_connections()?
        .iter()
        .map(|connection| {
            let from_instrument = connection.get_from_instrument()?.to_str()?;
            let from_port = connection.get_from_port()?.to_str()?.to_string();
            let to_instrument = connection.get_to_instrument()?.to_str()?;
            let to_port = connection.get_to_port()?.to_str()?.to_string();

            Ok(InternalConnection {
                from_instrument: id_store.get_or_insert(from_instrument).into(),
                from_port,
                to_instrument: id_store.get_or_insert(to_instrument).into(),
                to_port,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::setup_description::InstrumentKind;
    use capnp::any_pointer;
    use capnp::message::TypedBuilder;

    use super::*;

    /// A setup description as the serializer leaves it, wrapped the way the deserializer
    /// hands it back to the backend.
    fn payload(device_type: &str) -> SetupDescriptionPayload {
        let mut message = capnp::message::Builder::new_default();
        let mut builder = message.init_root::<schema::setup_description_qccs::Builder<'_>>();

        let mut instruments = builder.reborrow().init_instruments(1);
        let mut instrument = instruments.reborrow().get(0);
        instrument.set_uid("shfqc");
        instrument.set_device_type(device_type);
        instrument.set_reference_clock_source(schema::ReferenceClock::External);
        instrument.reborrow().init_options(1).set(0, "SHFQC6");

        let mut signals = builder.reborrow().init_signals(1);
        let mut signal = signals.reborrow().get(0);
        signal.set_uid("shfqc/measure");
        signal.set_instrument_uid("shfqc");
        signal
            .reborrow()
            .init_ports(1)
            .set(0, "QACHANNELS/0/OUTPUT");

        let mut connections = builder.reborrow().init_internal_connections(1);
        let mut connection = connections.reborrow().get(0);
        connection.set_from_instrument("ppc");
        connection.set_from_port("PPCHANNELS/0");
        connection.set_to_instrument("shfqc");
        connection.set_to_port("QACHANNELS/0/INPUT");

        let message: TypedBuilder<any_pointer::Owned> = message.into_typed();
        SetupDescriptionPayload::from_pointer(BACKEND_NAME, message.get_root_as_reader().unwrap())
            .unwrap()
    }

    #[test]
    fn a_setup_description_is_decoded_into_its_parts() {
        let mut id_store = NamedIdStore::new();
        let setup = read(&payload("SHFQC"), &mut id_store).unwrap();

        let instruments = setup.instruments();
        assert_eq!(instruments.len(), 1);
        assert_eq!(instruments[0].kind, InstrumentKind::Shfqc);
        assert_eq!(id_store.resolve(instruments[0].uid), Some("shfqc"));
        assert_eq!(
            instruments[0].reference_clock,
            Some(ReferenceClock::External)
        );
        assert_eq!(*instruments[0].options, ["SHFQC6"]);

        let channel = &setup.device_signals[0];
        assert_eq!(id_store.resolve(channel.uid), Some("shfqc/measure"));
        assert_eq!(channel.ports, ["QACHANNELS/0/OUTPUT"]);

        let connection = &setup.internal_connections()[0];
        assert_eq!(id_store.resolve(connection.from_instrument), Some("ppc"));
        assert_eq!(connection.from_port, "PPCHANNELS/0");
        assert_eq!(id_store.resolve(connection.to_instrument), Some("shfqc"));
        assert_eq!(connection.to_port, "QACHANNELS/0/INPUT");
    }

    #[test]
    fn an_unknown_device_type_is_reported() {
        let err = read(&payload("SHFXX"), &mut NamedIdStore::new()).unwrap_err();
        assert!(
            err.to_string().contains("Unknown device type: SHFXX"),
            "unexpected error: {err}"
        );
    }
}
