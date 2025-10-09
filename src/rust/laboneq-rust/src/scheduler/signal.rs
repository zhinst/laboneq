// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::AwgKey;
use laboneq_scheduler::SignalInfo;
use laboneq_scheduler::ir::SignalUid;
use pyo3::prelude::*;

pub struct Signal {
    pub uid: SignalUid,
    pub sampling_rate: f64,
    pub awg_key: AwgKey,
}

impl SignalInfo for Signal {
    fn uid(&self) -> SignalUid {
        self.uid
    }
    fn awg_key(&self) -> AwgKey {
        self.awg_key
    }
    fn sampling_rate(&self) -> f64 {
        self.sampling_rate
    }
}

#[pyclass(name = "Signal", frozen)]
pub struct SignalPy {
    pub uid: String,
    pub sampling_rate: f64,
    pub awg_key: i64,
}

impl SignalPy {
    pub fn to_signal(&self, id_store: &mut NamedIdStore) -> Signal {
        Signal {
            uid: SignalUid(
                // We must insert the signal as Compiler may add dummy signals that do not exist in the experiment.
                id_store.get_or_insert(&self.uid),
            ),
            awg_key: AwgKey(self.awg_key as u64),
            sampling_rate: self.sampling_rate,
        }
    }
}

#[pymethods]
impl SignalPy {
    #[new]
    pub fn new(uid: String, sampling_rate: f64, awg_key: i64) -> Self {
        Self {
            uid,
            sampling_rate,
            awg_key,
        }
    }
}
