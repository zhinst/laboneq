// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::exceptions::PyValueError;
use pyo3::{intern, prelude::*, types::PyDict, types::PyModule};

use laboneq_common::types::DeviceKind;

use crate::scheduler::experiment::Device;
use crate::scheduler::qccs_feedback_calculator::feedback_calculator::FeedbackModel;

/// Wrapper around the QCCS feedback model.
///
/// This struct provides an interface to calculate feedback latency using the QCCS feedback model implemented in Python.
/// It interacts with the `zhinst.timing_models` Python module to build the feedback model.
pub(crate) struct QCCSFeedbackModel<'py> {
    module: Bound<'py, PyModule>,
}

impl FeedbackModel<'_> for QCCSFeedbackModel<'_> {
    fn get_latency(
        &self,
        acquisition_end_samples: i64,
        acquisition_device: &Device,
        generator_device: &Device,
        local_feedback: bool,
    ) -> anyhow::Result<i64> {
        let latency = self.get_latency(
            acquisition_end_samples,
            acquisition_device,
            generator_device,
            local_feedback,
        )?;
        Ok(latency)
    }
}

impl QCCSFeedbackModel<'_> {
    pub(super) fn new(py: Python<'_>) -> PyResult<QCCSFeedbackModel<'_>> {
        let module = PyModule::import(py, intern!(py, "zhinst.timing_models"))?;
        Ok(QCCSFeedbackModel { module })
    }

    fn get_sg_type(&self, py: Python<'_>, device: &Device) -> PyResult<Py<PyAny>> {
        let sg_device = if device.is_shfqc {
            return Ok(self
                .module
                .getattr(intern!(py, "SGType"))?
                .getattr(intern!(py, "SHFQC"))?
                .into());
        } else {
            match &device.kind {
                DeviceKind::Hdawg => self
                    .module
                    .getattr(intern!(py, "SGType"))?
                    .getattr(intern!(py, "HDAWG"))?
                    .into(),
                DeviceKind::Shfsg => self
                    .module
                    .getattr(intern!(py, "SGType"))?
                    .getattr(intern!(py, "SHFSG"))?
                    .into(),
                _ => Err(PyValueError::new_err(format!(
                    "Device: '{}' cannot be used to generate feedback pulses",
                    device.kind
                )))?,
            }
        };
        Ok(sg_device)
    }

    fn get_qa_type(&self, py: Python<'_>, device: &Device) -> PyResult<Option<Py<PyAny>>> {
        let qa_type = if device.is_shfqc {
            Some(
                self.module
                    .getattr(intern!(py, "QAType"))?
                    .getattr(intern!(py, "SHFQC"))?
                    .into(),
            )
        } else {
            match &device.kind {
                DeviceKind::Shfqa => Some(
                    self.module
                        .getattr(intern!(py, "QAType"))?
                        .getattr(intern!(py, "SHFQA"))?
                        .into(),
                ),
                DeviceKind::Uhfqa => None,
                _ => panic!("Unsupported device kind for feedback QA device model"),
            }
        };
        Ok(qa_type)
    }

    fn build_model(
        &self,
        acquisition_device: &Device,
        generator_device: &Device,
        local_feedback: bool,
    ) -> PyResult<Bound<'_, PyAny>> {
        let qa_type = self.get_qa_type(self.module.py(), acquisition_device)?;
        let sg_type = self.get_sg_type(self.module.py(), generator_device)?;
        let model_class = self
            .module
            .getattr(intern!(self.module.py(), "QCCSFeedbackModel"))?;
        let system_description_func = self
            .module
            .getattr(intern!(self.module.py(), "get_feedback_system_description"))?;
        let trigger_source = self
            .module
            .getattr(intern!(self.module.py(), "TriggerSource"))?
            .getattr(intern!(self.module.py(), "ZSYNC"))?;
        let pqsc_mode = if !local_feedback {
            let pqsc_mode = self
                .module
                .getattr(intern!(self.module.py(), "PQSCMode"))?
                .getattr(intern!(self.module.py(), "REGISTER_FORWARD"))?;
            Some(pqsc_mode)
        } else {
            None
        };
        let feedback_path = if local_feedback {
            let feedback_path = self
                .module
                .getattr(intern!(self.module.py(), "FeedbackPath"))?
                .getattr(intern!(self.module.py(), "INTERNAL"))?;
            Some(feedback_path)
        } else {
            let feedback_path = self
                .module
                .getattr(intern!(self.module.py(), "FeedbackPath"))?
                .getattr(intern!(self.module.py(), "ZSYNC"))?;
            Some(feedback_path)
        };

        // Init the system description
        let kwargs = PyDict::new(self.module.py());
        kwargs.set_item(intern!(self.module.py(), "generator_type"), sg_type)?;
        kwargs.set_item(intern!(self.module.py(), "analyzer_type"), qa_type)?;
        kwargs.set_item(intern!(self.module.py(), "pqsc_mode"), pqsc_mode)?;
        kwargs.set_item(intern!(self.module.py(), "trigger_source"), trigger_source)?;
        kwargs.set_item(intern!(self.module.py(), "feedback_path"), feedback_path)?;
        let system_description = system_description_func.call((), Some(&kwargs))?;

        // Initialize `QCCSFeedbackModel` with the system description
        let model_instance: Bound<'_, PyAny> = model_class.call1((system_description,))?;
        Ok(model_instance)
    }

    pub(super) fn get_latency(
        &self,
        acquisition_end_samples: i64,
        acquisition_device: &Device,
        generator_device: &Device,
        local_feedback: bool,
    ) -> PyResult<i64> {
        let model = self.build_model(acquisition_device, generator_device, local_feedback)?;
        let get_latency_func = model.getattr(intern!(self.module.py(), "get_latency"))?;
        let latency_samples = get_latency_func.call1((acquisition_end_samples,))?;
        Ok(latency_samples
            .extract::<i64>()
            .expect("Expected feedback latency to be integer."))
    }
}
