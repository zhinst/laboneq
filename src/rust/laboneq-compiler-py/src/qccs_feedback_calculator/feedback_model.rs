// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::exceptions::PyValueError;
use pyo3::{intern, prelude::*, types::PyDict, types::PyModule};

use crate::qccs_feedback_calculator::feedback_calculator::{FeedbackDevice, FeedbackModel};

/// Wrapper around the QCCS feedback model.
///
/// This struct provides an interface to calculate feedback latency using the QCCS feedback model implemented in Python.
/// It interacts with the `zhinst.timing_models` Python module to build the feedback model.
pub struct QCCSFeedbackModel {}

impl FeedbackModel for QCCSFeedbackModel {
    fn get_latency(
        &self,
        acquisition_end_samples: i64,
        acquisition_device: &FeedbackDevice,
        generator_device: &FeedbackDevice,
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

impl QCCSFeedbackModel {
    pub(super) fn new() -> QCCSFeedbackModel {
        QCCSFeedbackModel {}
    }

    fn get_sg_type(
        &self,
        module: &Bound<'_, PyModule>,
        device: &FeedbackDevice,
    ) -> PyResult<Py<PyAny>> {
        let py = module.py();

        match device {
            FeedbackDevice::Shfqc => Ok(module
                .getattr(intern!(py, "SGType"))?
                .getattr(intern!(py, "SHFQC"))?
                .into()),
            FeedbackDevice::Hdawg => Ok(module
                .getattr(intern!(py, "SGType"))?
                .getattr(intern!(py, "HDAWG"))?
                .into()),
            FeedbackDevice::Shfsg => Ok(module
                .getattr(intern!(py, "SGType"))?
                .getattr(intern!(py, "SHFSG"))?
                .into()),
            _ => Err(PyValueError::new_err(format!(
                "Device: '{device}' cannot be used to generate feedback pulses"
            )))?,
        }
    }

    fn get_qa_type(
        &self,
        module: &Bound<'_, PyModule>,
        device: &FeedbackDevice,
    ) -> PyResult<Option<Py<PyAny>>> {
        let py = module.py();

        match device {
            FeedbackDevice::Shfqc => Ok(Some(
                module
                    .getattr(intern!(py, "QAType"))?
                    .getattr(intern!(py, "SHFQC"))?
                    .into(),
            )),
            FeedbackDevice::Shfqa => Ok(Some(
                module
                    .getattr(intern!(py, "QAType"))?
                    .getattr(intern!(py, "SHFQA"))?
                    .into(),
            )),
            FeedbackDevice::Uhfqa => Ok(None),
            _ => Err(PyValueError::new_err(format!(
                "Device: '{device}' cannot be used as feedback QA device"
            )))?,
        }
    }

    fn build_model<'py>(
        &self,
        module: Bound<'py, PyModule>,
        acquisition_device: &FeedbackDevice,
        generator_device: &FeedbackDevice,
        local_feedback: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = module.py();

        let qa_type = self.get_qa_type(&module, acquisition_device)?;
        let sg_type = self.get_sg_type(&module, generator_device)?;
        let model_class = module.getattr(intern!(py, "QCCSFeedbackModel"))?;
        let system_description_func =
            module.getattr(intern!(py, "get_feedback_system_description"))?;
        let trigger_source = module
            .getattr(intern!(py, "TriggerSource"))?
            .getattr(intern!(py, "ZSYNC"))?;
        let pqsc_mode = if !local_feedback {
            let pqsc_mode = module
                .getattr(intern!(py, "PQSCMode"))?
                .getattr(intern!(py, "REGISTER_FORWARD"))?;
            Some(pqsc_mode)
        } else {
            None
        };
        let feedback_path = if local_feedback {
            let feedback_path = module
                .getattr(intern!(py, "FeedbackPath"))?
                .getattr(intern!(py, "INTERNAL"))?;
            Some(feedback_path)
        } else {
            let feedback_path = module
                .getattr(intern!(py, "FeedbackPath"))?
                .getattr(intern!(py, "ZSYNC"))?;
            Some(feedback_path)
        };

        // Init the system description
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "generator_type"), sg_type)?;
        kwargs.set_item(intern!(py, "analyzer_type"), qa_type)?;
        kwargs.set_item(intern!(py, "pqsc_mode"), pqsc_mode)?;
        kwargs.set_item(intern!(py, "trigger_source"), trigger_source)?;
        kwargs.set_item(intern!(py, "feedback_path"), feedback_path)?;
        let system_description = system_description_func.call((), Some(&kwargs))?;

        // Initialize `QCCSFeedbackModel` with the system description
        let model_instance = model_class.call1((system_description,))?;
        Ok(model_instance)
    }

    pub(super) fn get_latency(
        &self,
        acquisition_end_samples: i64,
        acquisition_device: &FeedbackDevice,
        generator_device: &FeedbackDevice,
        local_feedback: bool,
    ) -> PyResult<i64> {
        Python::attach(|py| {
            let module = PyModule::import(py, intern!(py, "zhinst.timing_models"))?;
            let model = QCCSFeedbackModel::new().build_model(
                module,
                acquisition_device,
                generator_device,
                local_feedback,
            )?;
            let get_latency_func = model.getattr(intern!(py, "get_latency"))?;
            let latency_samples = get_latency_func.call1((acquisition_end_samples,))?;
            Ok(latency_samples
                .extract::<i64>()
                .expect("Expected feedback latency to be integer."))
        })
    }
}
