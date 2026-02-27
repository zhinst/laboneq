// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{NumericLiteral, ParameterUid};
use laboneq_scheduler::{ParameterStore, ParameterStoreBuilder};

pub(super) fn create_parameter_store(
    parameters: HashMap<String, f64>,
    id_store: &NamedIdStore,
) -> ParameterStore {
    let mut builder = ParameterStoreBuilder::new();
    for (uid, value) in parameters.iter() {
        let param_uid = id_store
            .get(uid)
            .unwrap_or_else(|| panic!("Internal error: Parameter {uid} not found in id store."));
        builder = builder.with_parameter(ParameterUid(param_uid), NumericLiteral::Float(*value));
    }
    builder.build()
}
