// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use codegenerator::string_sanitize;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PySequence},
};
use seqc_tracker::{
    Samples,
    compressor::{compress_generator, merge_generators, to_hash},
    prng_tracker::PRNGTracker,
    seqc_generator::{SeqCGenerator, seqc_generator_from_device_and_signal_type},
    seqc_statements::SeqCVariant,
    seqc_tracker::SeqCTracker,
    wave_index_tracker::WaveIndexTracker,
};

pyo3::import_exception!(laboneq.core.exceptions, LabOneQException);

#[pyfunction(name = "string_sanitize")]
pub fn string_sanitize_py(input: &str) -> String {
    string_sanitize(input)
}

#[pyclass]
#[pyo3(name = "WaveIndexTracker")]
pub(crate) struct WaveIndexTrackerPy {
    wave_index_tracker: WaveIndexTracker,
}

#[pymethods]
impl WaveIndexTrackerPy {
    #[new]
    fn new() -> Self {
        Self {
            wave_index_tracker: WaveIndexTracker::new(),
        }
    }

    pub fn lookup_index_by_wave_id(&self, wave_id: &str) -> Option<i32> {
        self.wave_index_tracker.lookup_index_by_wave_id(wave_id)
    }

    pub fn create_index_for_wave(
        &mut self,
        wave_id: &str,
        signal_type: &str,
    ) -> PyResult<Option<i32>> {
        self.wave_index_tracker
            .create_index_for_wave(wave_id, signal_type)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    pub fn add_numbered_wave(&mut self, wave_id: String, signal_type: String, index: i32) {
        self.wave_index_tracker
            .add_numbered_wave(wave_id, signal_type, index);
    }

    pub fn wave_indices(&self, py: Python) -> PyResult<PyObject> {
        // Cast entries to list. This will end up in the compiled experiment, and we want
        // invariance under serialization + deserialization, but the JSON decoder
        // produces lists, not tuples.
        let dict = PyDict::new(py);
        for (k, v) in self.wave_index_tracker.wave_indices.iter() {
            let list = PyList::new(py, vec![v.0]).unwrap();
            list.append(v.1.clone()).unwrap();
            dict.set_item(k, list)?;
        }
        Ok(dict.into())
    }
}

#[pyfunction]
#[pyo3(name = "seqc_generator_from_device_and_signal_type")]
pub(crate) fn seqc_generator_from_device_and_signal_type_py(
    device: &str,
    signal_type: &str,
) -> PyResult<SeqCGeneratorPy> {
    let seq_c_generator = seqc_generator_from_device_and_signal_type(device, signal_type);
    match seq_c_generator {
        Ok(seq_c_generator) => Ok(SeqCGeneratorPy { seq_c_generator }),
        Err(err) => Err(pyo3::exceptions::PyValueError::new_err(err.to_string())),
    }
}

fn union_to_variant(obj: Bound<'_, PyAny>) -> Result<SeqCVariant, PyErr> {
    if obj.is_instance_of::<pyo3::types::PyInt>() {
        let int_value = obj.extract::<i64>()?;
        return Ok(SeqCVariant::Integer(int_value));
    }
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        let float_value = obj.extract::<f64>()?;
        return Ok(SeqCVariant::Float(float_value));
    }
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(SeqCVariant::Bool(obj.extract::<bool>()?));
    }
    if obj.is_instance_of::<pyo3::types::PyString>() {
        return Ok(SeqCVariant::String(obj.extract::<String>()?));
    }
    if obj.is_instance_of::<pyo3::types::PyNone>() {
        return Ok(SeqCVariant::None());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Unsupported argument type",
    ))
}

fn sequence_to_variants(seq: Bound<'_, PySequence>) -> Result<Vec<SeqCVariant>, PyErr> {
    let mut result = Vec::new();
    for item in seq.try_iter()? {
        let item = item?;
        result.push(union_to_variant(item)?);
    }
    Ok(result)
}

#[pyclass]
#[pyo3(name = "SeqCGenerator")]
pub(crate) struct SeqCGeneratorPy {
    pub(crate) seq_c_generator: SeqCGenerator,
}

#[pymethods]
impl SeqCGeneratorPy {
    // def create(self) -> "SeqCGenerator": ...
    pub fn create(&self) -> PyResult<SeqCGeneratorPy> {
        let seq_c_generator = self.seq_c_generator.create();
        Ok(SeqCGeneratorPy { seq_c_generator })
    }

    // def num_statements(self) -> int: ...
    pub fn num_statements(&self) -> usize {
        self.seq_c_generator.num_statements()
    }

    // def num_noncomment_statements(self) -> int: ...
    pub fn num_noncomment_statements(&self) -> usize {
        self.seq_c_generator.num_noncomment_statements()
    }

    // def append_statements_from(self, seq_c_generator: "SeqCGenerator") -> None: ...
    pub fn append_statements_from(&mut self, seq_c_generator: &mut SeqCGeneratorPy) {
        self.seq_c_generator
            .merge_statements_from(&seq_c_generator.seq_c_generator);
    }

    // def add_comment(self, comment_text: str) -> None: ...
    pub fn add_comment(&mut self, comment_text: String) {
        self.seq_c_generator.add_comment(comment_text);
    }

    // def add_function_call_statement(
    //     self,
    //     name: str,
    //     args: Sequence[bool | int | float | str] | None = None,
    //     assign_to: str | None = None,
    // ) -> None: ...
    #[pyo3(signature = (name, args = None, assign_to = None))]
    pub fn add_function_call_statement(
        &mut self,
        name: String,
        args: Option<Bound<'_, PySequence>>,
        assign_to: Option<String>,
    ) -> PyResult<()> {
        let args = match args {
            Some(args) => sequence_to_variants(args)?,
            None => vec![],
        };
        self.seq_c_generator
            .add_function_call_statement(name, args, assign_to);
        Ok(())
    }

    // def add_wave_declaration(
    //     self, wave_id: str, length: int, has_marker1: bool, has_marker2: bool
    // ) -> None: ...
    pub fn add_wave_declaration(
        &mut self,
        wave_id: String,
        length: Samples,
        has_marker1: bool,
        has_marker2: bool,
    ) -> PyResult<()> {
        self.seq_c_generator
            .add_wave_declaration(wave_id, length, has_marker1, has_marker2)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    // def add_zero_wave_declaration(self, wave_id: str, length: int) -> None: ...
    pub fn add_zero_wave_declaration(&mut self, wave_id: String, length: Samples) -> PyResult<()> {
        self.seq_c_generator
            .add_zero_wave_declaration(wave_id, length)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    // def add_constant_definition(
    //     self, name: str, value: bool | int | float | str, comment: str | None = None
    // ) -> None: ...
    #[pyo3(signature = (name, value, comment = None))]
    pub fn add_constant_definition(
        &mut self,
        name: String,
        value: Bound<'_, PyAny>,
        comment: Option<String>,
    ) -> PyResult<()> {
        let value = union_to_variant(value)?;
        self.seq_c_generator
            .add_constant_definition(name, value, comment);
        Ok(())
    }

    // def add_repeat(self, num_repeats: int, body: SeqCGenerator) -> None: ...
    pub fn add_repeat(&mut self, num_repeats: u64, body: &SeqCGeneratorPy) {
        // For now, we clone the object we receive from the Python side.
        // Later, we can just move the object into the function, since we do not
        // need the original object anymore.
        self.seq_c_generator
            .add_repeat(num_repeats, body.seq_c_generator.clone());
    }

    // def add_do_while(self, condition: str, body: SeqCGenerator) -> None: ...
    pub fn add_do_while(&mut self, condition: &str, body: &SeqCGeneratorPy) {
        // For now, we clone the object we receive from the Python side.
        // Later, we can just move the object into the function, since we do not
        // need the original object anymore.
        self.seq_c_generator
            .add_do_while(condition, body.seq_c_generator.clone());
    }

    // def add_if(
    //     self, conditions: Sequence[str | None], bodies: Sequence[SeqCGenerator]
    // ) -> None: ...
    pub fn add_if<'py>(
        &mut self,
        conditions: Bound<'py, PySequence>,
        bodies: Bound<'py, PySequence>,
    ) -> PyResult<()> {
        let n_conditions = conditions.len()?;
        if conditions.len()? != bodies.len()? {
            return Err(LabOneQException::new_err(
                "Conditions and bodies must have the same length",
            ));
        }
        if conditions
            .try_iter()
            .into_iter()
            .take(n_conditions - 1)
            .any(|c| c.is_none())
        {
            return Err(LabOneQException::new_err(
                "Conditions must not be None, except for the last one",
            ));
        }
        let conditions = sequence_to_variants(conditions)?;
        let conditions = conditions.iter().map(|c| c.to_string()).collect::<Vec<_>>();
        let py = bodies.py();
        let bodies: Vec<Py<SeqCGeneratorPy>> = bodies
            .try_iter()?
            .map(|b| b.unwrap().extract::<Py<SeqCGeneratorPy>>().unwrap())
            .collect::<Vec<_>>();
        let mut conditions_internal = vec![];
        let mut bodies_internal = vec![];
        for (condition, body) in conditions.iter().zip(bodies.iter()) {
            conditions_internal.push(condition.as_str());
            // For now, we clone the object we receive from the Python side.
            // Later, we can just move the object into the function, since we do not
            // need the original object anymore.
            bodies_internal.push(body.borrow(py).seq_c_generator.clone());
        }
        // The last condition is allowed to be None, so we remove it from the list.
        if conditions_internal.last().unwrap_or(&"").is_empty() {
            conditions_internal.pop();
        }
        self.seq_c_generator
            .add_if(conditions_internal, bodies_internal)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    // def add_function_def(self, text: str) -> None: ...
    pub fn add_function_def(&mut self, text: String) {
        self.seq_c_generator.add_function_def(text);
    }

    // def is_variable_declared(self, variable_name: str) -> bool: ...
    pub fn is_variable_declared(&self, variable_name: &str) -> bool {
        self.seq_c_generator.is_variable_declared(variable_name)
    }

    // def add_variable_declaration(
    //     self, variable_name: str, initial_value: bool | int | float | str | None = None
    // ) -> None: ...
    #[pyo3(signature = (variable_name, initial_value = None))]
    pub fn add_variable_declaration(
        &mut self,
        variable_name: String,
        initial_value: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let initial_value = match initial_value {
            Some(value) => Some(union_to_variant(value)?),
            None => None,
        };
        self.seq_c_generator
            .add_variable_declaration(variable_name, initial_value)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    // def add_variable_assignment(
    //     self, variable_name: str, value: bool | int | float | str
    // ) -> None: ...
    pub fn add_variable_assignment(
        &mut self,
        variable_name: String,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let value = union_to_variant(value)?;
        self.seq_c_generator
            .add_variable_assignment(variable_name, value);
        Ok(())
    }

    // def add_variable_increment(
    //     self, variable_name: str, value: int | float
    // ) -> None: ...
    pub fn add_variable_increment(
        &mut self,
        variable_name: String,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let value = union_to_variant(value)?;
        self.seq_c_generator
            .add_variable_increment(variable_name, value);
        Ok(())
    }

    // def add_assign_wave_index_statement(
    //     self, wave_id: str, wave_index: int, channel: int | None
    // ) -> None: ...
    pub fn add_assign_wave_index_statement(
        &mut self,
        wave_id: String,
        wave_index: u64,
        channel: Option<u16>,
    ) {
        self.seq_c_generator
            .add_assign_wave_index_statement(wave_id, wave_index, channel);
    }

    // def add_play_wave_statement(self, wave_id: str, channel: int | None) -> None: ...
    pub fn add_play_wave_statement(&mut self, wave_id: String, channel: Option<u16>) {
        self.seq_c_generator
            .add_play_wave_statement(wave_id, channel);
    }

    // def add_command_table_execution(
    //     self, ct_index: int | str, latency: int | str | None = None, comment: str = ""
    // ) -> None: ...
    #[pyo3(signature = (ct_index, latency = None, comment = ""))]
    pub fn add_command_table_execution(
        &mut self,
        ct_index: Bound<'_, PyAny>,
        latency: Option<Bound<'_, PyAny>>,
        comment: Option<&str>,
    ) -> PyResult<()> {
        let latency = match latency {
            Some(latency) => Some(union_to_variant(latency)?),
            None => None,
        };
        self.seq_c_generator.add_command_table_execution(
            union_to_variant(ct_index)?,
            latency,
            comment,
        );
        Ok(())
    }

    // def add_play_zero_statement(
    //     self, num_samples: int, deferred_calls: SeqCGenerator | None = None
    // ) -> None: ...
    #[pyo3(signature = (num_samples, deferred_calls = None))]
    pub fn add_play_zero_statement(
        &mut self,
        num_samples: Samples,
        deferred_calls: Option<&mut SeqCGeneratorPy>,
    ) -> PyResult<()> {
        self.seq_c_generator
            .add_play_zero_statement(
                num_samples,
                &mut deferred_calls.map(|d| &mut d.seq_c_generator),
            )
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    // def add_play_hold_statement(
    //     self, num_samples: int, deferred_calls: SeqCGenerator | None = None
    // ) -> None: ...
    #[pyo3(signature = (num_samples, deferred_calls = None))]
    pub fn add_play_hold_statement(
        &mut self,
        num_samples: Samples,
        deferred_calls: Option<&mut SeqCGeneratorPy>,
    ) -> PyResult<()> {
        self.seq_c_generator
            .add_play_hold_statement(
                num_samples,
                &mut deferred_calls.map(|d| &mut d.seq_c_generator),
            )
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }
    // def generate_seq_c(self) -> str: ...
    pub fn generate_seq_c(&self) -> String {
        self.seq_c_generator.generate_seq_c()
    }

    // def __hash__(self) -> int: ...
    pub fn __hash__(&self) -> u64 {
        to_hash(&self.seq_c_generator)
    }

    // def __eq__(self, other: object) -> bool: ...
    pub fn __eq__(&self, other: &Bound<'_, SeqCGeneratorPy>) -> bool {
        self.seq_c_generator
            == other
                .extract::<Py<SeqCGeneratorPy>>()
                .unwrap()
                .borrow(other.py())
                .seq_c_generator
    }

    // def __repr__(self) -> str: ...
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.seq_c_generator)
    }

    // def compressed(self) -> SeqCGenerator: ...
    pub fn compressed(&self) -> SeqCGeneratorPy {
        // For now, we clone the object we receive from the Python side.
        // Later, we can just move the object into the function, since we do not
        // need the original object anymore.
        SeqCGeneratorPy {
            seq_c_generator: compress_generator(self.seq_c_generator.clone()),
        }
    }
}

#[pyfunction]
#[pyo3(name = "merge_generators")]
pub(crate) fn merge_generators_py(
    seq_c_generators: Bound<'_, PyList>,
    compress: bool,
) -> SeqCGeneratorPy {
    let py = seq_c_generators.py();
    let extracted_generators: Vec<Py<SeqCGeneratorPy>> = seq_c_generators
        .iter()
        .map(|g| g.extract::<Py<SeqCGeneratorPy>>().unwrap())
        .collect();
    let seq_c_generators_as_pyref = extracted_generators
        .iter()
        .map(|g| g.borrow(py))
        .collect::<Vec<_>>();
    let seq_c_generators = seq_c_generators_as_pyref
        .iter()
        .map(|g| &g.seq_c_generator)
        .collect::<Vec<_>>();
    SeqCGeneratorPy {
        seq_c_generator: merge_generators(&seq_c_generators, compress),
    }
}

#[pyclass]
#[pyo3(name = "PRNGTracker")]
pub(crate) struct PRNGTrackerPy {
    // todo: Remove smart pointer once SampledEventHandler is converted to Rust
    // and the PRNGTracker is not used in the Python code anymore.
    prng_tracker: Arc<RwLock<PRNGTracker>>,
}

#[pymethods]
impl PRNGTrackerPy {
    #[new]
    fn new() -> Self {
        Self {
            prng_tracker: Arc::new(RwLock::new(PRNGTracker::new())),
        }
    }

    #[getter]
    pub fn get_offset(&self) -> u32 {
        self.prng_tracker
            .read()
            .expect("PRNGTracker is already locked")
            .offset()
    }

    #[setter]
    pub fn set_offset(&mut self, value: u32) {
        self.prng_tracker
            .write()
            .expect("PRNGTracker is already locked")
            .set_offset(value);
    }

    #[getter]
    pub fn get_active_sample(&self) -> Option<String> {
        self.prng_tracker
            .read()
            .expect("PRNGTracker is already locked")
            .active_sample()
            .cloned()
    }

    #[setter]
    pub fn set_active_sample(&mut self, value: String) {
        self.prng_tracker
            .write()
            .expect("PRNGTracker is already locked")
            .set_active_sample(value.to_string());
    }

    pub fn drop_sample(&mut self) {
        self.prng_tracker
            .write()
            .expect("PRNGTracker is already locked")
            .drop_sample();
    }

    pub fn is_committed(&self) -> bool {
        self.prng_tracker
            .read()
            .expect("PRNGTracker is already locked")
            .is_committed()
    }
}

#[pyclass]
#[pyo3(name = "SeqCTracker")]
pub(crate) struct SeqCTrackerPy {
    seq_c_tracker: Arc<RwLock<SeqCTracker>>,
}

impl SeqCTrackerPy {
    fn get_seq_c_tracker(&self) -> RwLockReadGuard<'_, SeqCTracker> {
        self.seq_c_tracker
            .read()
            .expect("SeqCTracker is already locked")
    }
    fn get_seq_c_tracker_mut(&mut self) -> RwLockWriteGuard<'_, SeqCTracker> {
        self.seq_c_tracker
            .write()
            .expect("SeqCTracker is already locked")
    }
}

#[pymethods]
impl SeqCTrackerPy {
    //     def __init__(
    //         self,
    //         init_generator: SeqCGenerator,
    //         deferred_function_calls: SeqCGenerator,
    //         sampling_rate: float,
    //         delay: float,
    //         device_type: DeviceType,
    //         signal_type: AWGSignalType,
    //         emit_timing_comments: bool,
    //         automute_playzeros_min_duration: float,
    //         automute_playzeros: bool = False,
    //     ) -> None: ...
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        init_generator: &SeqCGeneratorPy,
        deferred_function_calls: &SeqCGeneratorPy,
        sampling_rate: f64,
        delay: f64,
        device_type: String,
        signal_type: String,
        emit_timing_comments: bool,
        automute_playzeros_min_duration: f64,
        automute_playzeros: bool,
    ) -> PyResult<Self> {
        let init_generator = init_generator.seq_c_generator.clone();
        let deferred_function_calls = deferred_function_calls.seq_c_generator.clone();
        let seq_c_tracker = Arc::new(RwLock::new(
            SeqCTracker::new(
                init_generator,
                deferred_function_calls,
                sampling_rate,
                delay,
                device_type,
                signal_type,
                emit_timing_comments,
                automute_playzeros_min_duration,
                automute_playzeros,
            )
            .map_err(|err| LabOneQException::new_err(err.to_string()))?,
        ));
        Ok(SeqCTrackerPy { seq_c_tracker })
    }

    //     @property
    //     def automute_playzeros(self) -> bool: ...
    #[getter]
    fn automute_playzeros(&self) -> bool {
        self.get_seq_c_tracker().automute_playzeros()
    }

    //     def add_required_playzeros(self, start: int) -> int: ...
    fn add_required_playzeros(&mut self, start: i64) -> PyResult<Samples> {
        self.get_seq_c_tracker_mut()
            .add_required_playzeros(start)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    //     def flush_deferred_function_calls(self) -> None: ...
    fn flush_deferred_function_calls(&mut self) {
        self.get_seq_c_tracker_mut().flush_deferred_function_calls();
    }

    //     def force_deferred_function_calls(self) -> None: ...
    fn force_deferred_function_calls(&mut self) -> PyResult<()> {
        self.get_seq_c_tracker_mut()
            .force_deferred_function_calls()
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    //     def flush_deferred_phase_changes(self) -> None: ...
    fn flush_deferred_phase_changes(&mut self) {
        self.get_seq_c_tracker_mut().flush_deferred_phase_changes();
    }

    //     def discard_deferred_phase_changes(self) -> None: ...
    fn discard_deferred_phase_changes(&mut self) {
        self.get_seq_c_tracker_mut()
            .discard_deferred_phase_changes();
    }

    //     def has_deferred_phase_changes(self) -> bool: ...
    fn has_deferred_phase_changes(&self) -> bool {
        self.get_seq_c_tracker().has_deferred_phase_changes()
    }

    //     def add_timing_comment(self, end_samples: int) -> None: ...
    fn add_timing_comment(&mut self, end_samples: Samples) {
        self.get_seq_c_tracker_mut().add_timing_comment(end_samples);
    }

    //     def add_comment(self, comment: str) -> None: ...
    fn add_comment(&mut self, comment: String) {
        self.get_seq_c_tracker_mut().add_comment(comment);
    }

    //     def add_function_call_statement(
    //         self,
    //         name: str,
    //         args: list[str] | None = None,
    //         assign_to: str | None = None,
    //         deferred: bool = False,
    //     ) -> None: ...
    #[pyo3(signature = (name, args = None, assign_to = None, deferred = false))]
    fn add_function_call_statement(
        &mut self,
        name: String,
        args: Option<Bound<'_, PySequence>>,
        assign_to: Option<String>,
        deferred: bool,
    ) -> PyResult<()> {
        let args = match args {
            Some(args) => sequence_to_variants(args)?,
            None => vec![],
        };
        self.get_seq_c_tracker_mut()
            .add_function_call_statement(name, args, assign_to, deferred);
        Ok(())
    }

    //     def add_play_zero_statement(
    //         self, num_samples: int, increment_counter: bool = False
    //     ) -> None: ...
    #[pyo3(signature = (num_samples, increment_counter = false))]
    fn add_play_zero_statement(
        &mut self,
        num_samples: Samples,
        increment_counter: bool,
    ) -> PyResult<()> {
        self.get_seq_c_tracker_mut()
            .add_play_zero_statement(num_samples, increment_counter)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }
    //     def add_play_hold_statement(self, num_samples: int) -> None: ...
    fn add_play_hold_statement(&mut self, num_samples: Samples) -> PyResult<()> {
        self.get_seq_c_tracker_mut()
            .add_play_hold_statement(num_samples)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    //     def add_play_wave_statement(self, wave_id: str, channel: int | None) -> None: ...
    fn add_play_wave_statement(&mut self, wave_id: String, channel: Option<u16>) {
        self.get_seq_c_tracker_mut()
            .add_play_wave_statement(wave_id, channel);
    }

    //     def add_command_table_execution(
    //         self, ct_index: int | str, latency: int | str | None = None, comment: str = ""
    //     ) -> None: ...
    #[pyo3(signature = (ct_index, latency = None, comment = ""))]
    fn add_command_table_execution(
        &mut self,
        ct_index: Bound<'_, PyAny>,
        latency: Option<Bound<'_, PyAny>>,
        comment: Option<&str>,
    ) -> PyResult<()> {
        let latency = match latency {
            Some(latency) => Some(union_to_variant(latency)?),
            None => None,
        };
        self.get_seq_c_tracker_mut()
            .add_command_table_execution(union_to_variant(ct_index)?, latency, comment)
            .map_err(|err| LabOneQException::new_err(err.to_string()))?;
        Ok(())
    }

    //     def add_phase_change(self, ct_index: int, comment: str = "") -> None: ...
    fn add_phase_change(&mut self, ct_index: i64, comment: Option<String>) {
        self.get_seq_c_tracker_mut()
            .add_phase_change(SeqCVariant::Integer(ct_index), comment);
    }

    //     def add_variable_assignment(
    //         self, variable_name: str, value: bool | int | float | str
    //     ) -> None: ...
    fn add_variable_assignment(
        &mut self,
        variable_name: String,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let value = union_to_variant(value)?;
        self.get_seq_c_tracker_mut()
            .add_variable_assignment(variable_name, value);
        Ok(())
    }
    //     def add_variable_increment(
    //         self, variable_name: str, value: int | float
    //     ) -> None: ...
    fn add_variable_increment(
        &mut self,
        variable_name: String,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let value = union_to_variant(value)?;
        self.get_seq_c_tracker_mut()
            .add_variable_increment(variable_name, value);
        Ok(())
    }
    //     def append_loop_stack_generator(
    //         self, always: bool = False, generator: SeqCGenerator | None = None
    //     ) -> SeqCGenerator: ...
    #[pyo3(signature = (always = false, generator = None))]
    fn append_loop_stack_generator(
        &mut self,
        always: bool,
        generator: Option<&SeqCGeneratorPy>,
    ) -> PyResult<()> {
        let generator = generator.map(|g| g.seq_c_generator.clone());
        self.get_seq_c_tracker_mut()
            .append_loop_stack_generator(generator, always)
            .map_err(|err| LabOneQException::new_err(err.to_string()))?;
        Ok(())
    }

    //     def push_loop_stack_generator(
    //         self, generator: SeqCGenerator | None = None
    //     ) -> None: ...
    #[pyo3(signature = (generator = None))]
    fn push_loop_stack_generator(&mut self, generator: Option<&SeqCGeneratorPy>) -> PyResult<()> {
        let generator = generator.map(|g| g.seq_c_generator.clone());
        self.get_seq_c_tracker_mut()
            .push_loop_stack_generator(generator)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    //     def pop_loop_stack_generators(self) -> list[SeqCGenerator]: ...
    fn pop_loop_stack_generators(&mut self) -> Option<Vec<SeqCGeneratorPy>> {
        self.get_seq_c_tracker_mut()
            .pop_loop_stack_generators()
            .map(|generators| {
                generators
                    .into_iter()
                    .map(|g| SeqCGeneratorPy { seq_c_generator: g })
                    .collect()
            })
    }

    //     def setup_prng(
    //         self,
    //         seed: int | None = None,
    //         prng_range: int | None = None,
    //     ) -> None: ...
    #[pyo3(signature = (seed = None, prng_range = None))]
    fn setup_prng(&mut self, seed: Option<u32>, prng_range: Option<u32>) -> PyResult<()> {
        self.get_seq_c_tracker_mut()
            .setup_prng(seed, prng_range)
            .map_err(|err| LabOneQException::new_err(err.to_string()))?;
        Ok(())
    }

    //     def drop_prng(self) -> None: ...
    fn drop_prng(&mut self) -> PyResult<()> {
        self.get_seq_c_tracker_mut()
            .drop_prng()
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    //     def commit_prng(self) -> None: ...
    fn commit_prng(&mut self) {
        self.get_seq_c_tracker_mut().commit_prng();
    }

    //     def add_prng_match_command_table_execution(self, offset: int) -> None: ...
    fn add_prng_match_command_table_execution(&mut self, offset: i64) -> PyResult<()> {
        self.get_seq_c_tracker_mut()
            .add_prng_match_command_table_execution(offset)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    //     def sample_prng(self, declarations_generator: SeqCGenerator) -> None: ...
    fn sample_prng(&mut self, declarations_generator: &mut SeqCGeneratorPy) -> PyResult<()> {
        self.get_seq_c_tracker_mut()
            .sample_prng(&mut declarations_generator.seq_c_generator)
            .map_err(|err| LabOneQException::new_err(err.to_string()))
    }

    //     def prng_tracker(self) -> PRNGTracker | None: ...
    fn prng_tracker(&self) -> Option<PRNGTrackerPy> {
        self.get_seq_c_tracker()
            .prng_tracker()
            .map(|prng_tracker| PRNGTrackerPy { prng_tracker })
    }

    //     def add_set_trigger_statement(self, value: int, deferred: bool = True) -> None: ...
    #[pyo3(signature = (value, deferred = true))]
    fn add_set_trigger_statement(&mut self, value: u32, deferred: bool) {
        self.get_seq_c_tracker_mut()
            .add_set_trigger_statement(value, deferred);
    }

    //     def add_startqa_shfqa_statement(
    //         self,
    //         generator_mask: str,
    //         integrator_mask: str,
    //         monitor: int | None = None,
    //         feedback_register: int | None = None,
    //         trigger: int | None = None,
    //     ) -> None: ...
    #[pyo3(signature = (generator_mask, integrator_mask, monitor = None, feedback_register = None, trigger = None))]
    fn add_startqa_shfqa_statement(
        &mut self,
        generator_mask: String,
        integrator_mask: String,
        monitor: Option<u8>,
        feedback_register: Option<u32>,
        trigger: Option<u32>,
    ) {
        self.get_seq_c_tracker_mut().add_startqa_shfqa_statement(
            generator_mask,
            integrator_mask,
            monitor,
            feedback_register,
            trigger,
        );
    }

    //     def trigger_output_state(self) -> int: ...
    fn trigger_output_state(&self) -> u32 {
        self.get_seq_c_tracker().trigger_output_state()
    }

    //     def current_time(self) -> int: ...
    #[getter]
    fn get_current_time(&self) -> Samples {
        self.get_seq_c_tracker().current_time()
    }

    //     def current_time(self, value: int) -> None: ...
    #[setter]
    fn set_current_time(&mut self, value: Samples) {
        self.get_seq_c_tracker_mut().set_current_time(value);
    }

    //     def top_loop_stack_generators_have_statements(self) -> bool: ...
    fn top_loop_stack_generators_have_statements(&self) -> bool {
        let has_statements =
            if let Some(generators) = self.get_seq_c_tracker().top_loop_stack_generators() {
                generators.iter().any(|g| {
                    g.read()
                        .expect("SeqCGenerator is already locked")
                        .num_noncomment_statements()
                        > 0
                })
            } else {
                false
            };
        has_statements
            || self.get_seq_c_tracker().has_deferred_phase_changes()
            || self.get_seq_c_tracker().has_deferred_function_calls()
    }
}
