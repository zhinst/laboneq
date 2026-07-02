# LabOne Q 26.7.0b5 (2026-07-02)

## Features

- The automation web viewer now implements CSRF protection by rejecting state-changing requests (non-GET/HEAD) whose Origin or Referer header does not match the request host, and rejecting those sending neither.

## Bug Fixes

- Fixed a bug where deserialization of compiled experiments that use an SHFPCC device
  with null amplifier pump parameters would fail.
- Fixed a bug where attempting to serialize an experiment containing `reset_oscillator_phase` would raise an error. `ResetOscillatorPhase` operations are now correctly serialized and deserialized.
- Fixed a bug where a single-shot result of the form `np.array(X + jY)` was deserialized as `np.array([X + jY])`.
- Fixed a bug where a compiler error occurred in experiments with near-time sweeps and a static SHFPPC configuration.
- Fixed a bug where runtime type validation for FIR precompensation coefficients failed with NumPy 2.5.

# LabOne Q 26.7.0b4 (2026-06-18)

## Features

- Added an option for the caller of a quantum operation to override the pulse name used by create_pulse.

  The create_pulse function used inside many quantum operations allows the implementation of the quantum operation to directly supply a pulse name, which is used to identify the kind of pulse and as part of the pulse UID.

  However, a user may wish to generate different pulse objects even when using the same quantum operations. For example, in an experiment with M measure operations, a user may wish to replace a specific subset of those measurements with new pulses.

  To enable this, create_pulse now accepts a new special key named name in the dictionary of pulse options. This allows the user to override the supplied name by passing, for example, qop.measure(q, handle, readout_pulse={"name": "my-special-tag"}) when calling a quantum operation.
- Changed the device setup compatibility verification between compilation and execution. Experiments compiled with previous versions of LabOne Q may need recompilation.

## Bug Fixes

- Fixed a bug where outputs on different AWG cores of a single HDAWG were not synchronized (e.g. channels 1 and 3 out of sync).
- Fixed a bug where automute was not correctly applied when multiplexing output.
- Fixed a bug where "zi:fw:19" execution errors occurred with (auto-)chunking on HDAWG.

# LabOne Q 26.7.0b3 (2026-06-05)

## Features

- Added a `skip_passed` argument/attribute to the `AutomationLayer` class that makes it possible to skip running passed nodes.

## Bug Fixes

- Fixed a bug where a timeout error could occur during experiments using pipelined execution
  (chunked) with large result sizes. The timeout calculation now accounts for
  the GW-to-FW result transfer time, which was previously unaccounted for
  and could cause premature timeouts on experiments with many jobs and long readout
  blocks.
- Fixed a bug where explicitly setting `PortMode.RF` on a QA channel had no effect and the hardware used the LF path instead.

## Documentation

- Added reference documentation for `for_each` and `sweep_range`.

# LabOne Q 26.7.0b2 (2026-05-22)

## Features

- The compiler now raises an exception when an SHF oscillator intermediate frequency (IF) has an absolute value >= 1 GHz. SHF instruments do not support IF frequencies in this range.
- Added error tracking and visualization for failed layer executions. The exceptions from `run_layer_executable` are recorded via a new AutomationStatus.ERROR status and error attribute, with errored nodes blinking yellow-to-red in the web viewer and displaying the error message in the info panel on click.
- Multiple automation web viewers can now run simultaneously. Each `start_web_viewer` call launches an independent server on its own port for a separate `Automation` instance.
- Added the `quantum_element_uids` property to the `QPU` class, which returns the list of quantum element UIDs in the QPU.
- Added `reset_oscillator_phase` to `laboneq.simple.dsl` namespace.
  This function existed already but was not directly exposed.
  It resets the phase of the oscillator associated with a given logical signal.
- Improved the "Node results" image discovery for nodes with multiple quantum elements by searching for the target quantum element UID in the results image file names if the node key string is not found.
- Updated the Experiment serializer to support a richer set of values as pulse parameters.

  Previously the Experiment serializer supported only a limited basic set of pulse parameter types:

  - float, int, str, bool, complex
  - SweepParameter, LinearSweepParameter
  - list[SweepParameter], list[LinearSweepParameter]

  This set of types proved too restrictive for some use cases.

  Now the Experiment serializer supports an extended set of types for pulse parameters:

  - int, float, complex, str, bytes, bool, None, SweepParameter or LinearSweepParameter
  - lists of the above
  - dictionaries of the above with str keys

  The lists and dictionaries may be nested and need not have homogeneous element types.

  The previous Experiment serializer format was version 4. The new format is version 5.


## Bug Fixes

- Fixed a bug where compilation would fail when a chunked sweep contained both the driving parameter and the derived parameter.
- Fixed a bug where passing workflow OptionBuilder options directly to a task called outside a workflow context raised a ValueError.
- Fixed a bug where differences in comments produced excessively long, uncompressed SeqC.
- Fixed a bug in the automation web viewer, where the "Node results" section of the node info panel sometimes displayed the results of the previously-clicked node.
- Fixed a bug where a minor rounding error biased phase increments by up to +0.3 µrad.

## Documentation

- Added documentation and unit tests for the list of real-time sweepable parameters vs near-time only parameters.

## Removals from the Codebase

- Removed `LabOneQInstrumentor` traces: `laboneq.compiler.schedule` and `laboneq.compiler.generate-code`.

  Capturing compiler traces from Python now requires `laboneq.instrumentation.tracing.laboneq_tracing()`.

# LabOne Q 26.7.0b1 (2026-05-08)

## Features

- Improved the performance of the automation web viewer so that it runs smoothly with up to 200 quantum elements per layer. The front end has been simplified and all transitions are now entirely done natively by the browser.
- Changed exception type from `RuntimeError` to `LabOneQException` when an invalid device combination is provided.

## Bug Fixes

- Fixed a bug where the assignment vector calculation for multi-state discrimination on SHFQA was incorrect. The calculation is now delegated to zhinst-utils.

## Removals from the Codebase

- Removed device setup hot-reloading.

# LabOne Q 26.4.0 (2026-04-30)

## Features

- Added remote controller service for running experiments on remote machines.
- Released the LabOne Q Automation framework and added a [tutorial](https://docs.zhinst.com/labone_q_user_manual/core/functionality_and_concepts/07a_automation/tutorials/00_automation.html).

## Bug Fixes

- Fixed a bug where markers were not held during compressed pulse playback in the output simulator.
- Fixed a bug where PRNG match sections inside non-compressed sweep loops allocated duplicate command table entries on each sweep step, preventing the SeqC compressor from collapsing repeated bodies and potentially exhausting the command table entries.
- Fixed a vulnerability issue where a crafted serialized file could cause the deserialization engine to import and invoke arbitrary Python classes, resulting in arbitrary code execution.
- Fixed a bug where an incorrect exception was raised when injecting incompatible results in `run_experiment`.
- Fixed a bug where multiplexed SHFQA output channels used invalid oscillator frequencies (HBAR-2509)


## Removals from the Codebase

- Remote controller service: drop the admin reset endpoint, the `connection_state` field of the `/v1/info` response, and the implicit on-demand hardware connect.  The service now connects at startup via `ControllerContainer.create()` and aborts if the connection fails.
- internal cleanup: Remove some indirection enums that were adding no functionality.

# LabOne Q 26.4.0b5 (2026-04-09)

## Features

- Added `SectionTimingMode.STRICT` for sections and acquire loops. When enabled, the compiler raises an error if any timing value (section length, pulse length, repetition time) requires padding to fit the required grid, rather than silently padding. `RepetitionMode.AUTO` is incompatible with strict mode and also raises an error.
- The `inject_results` `paths` run experiment option is now of type `list[Path | str]`, i.e. a list of paths to serialized `Results` objects. If multiple file paths are provided, then the `Results` objects are combined.
- Renamed controller API endpoints to match new naming scheme, added stubs for sync controller API.

## Bug Fixes

- Fixed a bug where pulse replacements did not persist over near-time steps in situations where, besides the replacement(s), the near-time loop is used to sweep some pulse properties as well (the pulse being replaced, or another one).
- Fixed a bug where near-time pulse replacement replaced the wrong waveform if the pulse being replaced was in different positions in different near-time steps (can happen when matching against the near-time loop index and defining structurally different entries for different cases).
- Fixed a bug where delay compensation for input latency on SHFQA when using LRT was inaccurate. Users are advised to update their integration kernel estimations as this change affects signal latency.
- Fixed a bug where auto chunking, when chunking was not necessary, produced a different compilation result than when auto chunking was not enabled.

## Deprecation Notices

- Deprecated the methods `CompiledExperiment.replace_pulse` and `CompiledExperiment.replace_phase_increment`. NOTE: the methods `RuntimeContext.replace_pulse` and `RuntimeContext.replace_phase_increment` are not deprecated and can still be used to apply replacements in NT callbacks (these methods leave the compiled experiment unchanged).
- Deprecated `QPU.measure_section_length`. The method is only meaningful for certain types of qubits, so it will be moved to the implementations of `QuantumOperations` in the `laboneq-applications` repository instead.

# LabOne Q 26.4.0b4 (2026-03-27)

## Bug Fixes

- Fixed a bug where acquisition delay was not compensated in LRT mode. Users no longer need to manually add a port delay offset to the acquire signal when using long readout.
- Fixed a bug where hold-off errors from a previous experiment could cause subsequent experiments to crash.
- Fixed a bug where match-case against a near-time sweep parameter always executed the first case only.

# LabOne Q 26.4.0b3 (2026-03-13)

## Features

- Added a new attribute `result_properties` for `CompiledExperiment` that for each result handle contains information about the result array (the shape and the axis names).
- Added support for injecting serialized experiment results into workflows via the `inject_results` option, enabling reuse of previously recorded runs for emulation or testing.

## Bug Fixes

- Fixed a bug where instance variables were treated incorrectly.
- Fixed a bug where `axis_name` in experiment results contained names of internal derived parameters.
- Fixed a bug where near-time frequency sweep of QA acquire oscillator with hardware modulation did not work on SHFQA devices with the LRT option without requiring spectroscopy acquisition type.
- Fixed a bug where loops using SW oscillators were incorrectly compressed (re-rolled),
  causing all sweep iterations to reuse the same pre-computed waveform regardless of
  the accumulated oscillator phase.

# LabOne Q 26.4.0b2 (2026-02-27)

## Features

- Near-time Callbacks now receive an object implementing `RuntimeContext` as their first argument instead of a `Session`. No changes are required to existing callback function definitions except when they use the accessors to experiment data, calibrations, connection state or the device setup. In these cases, the required data must be submitted via the function arguments.  It is still recommended to rename the parameter from `session` to `runtime_context` and update the type hint, if present.
- Long readout pulses on SHFQA devices are now automatically compressed without requiring the `can_compress` parameter when using hardware with long readout time support.
- `ModulationType.AUTO` on QA devices in integration mode now resolves to `HARDWARE` only when the device has the LRT option and the acquisition length exceeds 2 µs; shorter acquisitions default to `SOFTWARE` to avoid silent phase-averaging errors and NCO limitations. Long acquisitions (> 2 µs) without LRT now raise a compile-time error. Spectroscopy mode is unchanged and always resolves to `HARDWARE`.
- Refactored the abstract base classes for the LabOne Q Automation framework (currently in beta), due to be officially released in v26.04.


## Bug Fixes

- Fixed a bug where UHFQA AWG status check failed after AWG ready logic change in 26.1.0.
- Fixed a bug where hardware modulation was not applied when using a short readout on SHFQA with an LTR option.
- Fixed a bug where stale SHFQA integration kernel downsampling factor persists across sequential experiment runs within the same session.
- Fixed a bug where deserialization of a compiled experiment that uses local feedback would fail.
- Fixed a bug where deserialization of a scheduled experiment with RAW acquisition would fail.

## Documentation

- Fix inaccurate description of the `add_reset` argument to `batch_experiment`.
  The argument adds a *reset* operation not an *active reset* operation.
  Whether the reset is active or passive is determined by the implementation
  provided by the quantum operations of the QPU.

## Deprecation Notices

- Deprecated the legacy serializer provided by `laboneq.dsl.serialization` and
  `laboneq.core.serialization`. These will be removed in LabOneQ 26.7.
  Use `laboneq.serializers` instead.

# LabOne Q 26.4.0b1 (2026-02-12)

## Features

- Session now supports automatic system profile fetching and caching when connecting to hardware. Use the new `system_profile` parameter in the Session constructor to explicitly provide a profile, or let it auto-load from cache when needed.
- Ported event list generation (required for Pulse Sheet Viewer) from Python to Rust for improved performance.

## Bug Fixes

- Fixed a bug where a read timeout occurred when running LRT after non-LRT, due to LabOne Q enabling MSD by default even for two states for non-LRT experiments. MSD is now explicitly disabled for LRT.
- Fixed a bug where OutputSimulator did not properly decompress long readout (LRT) waveforms on SHFQA, so that simulated output did not match the full hardware playback.
- Fixed a bug where PSV crashed if a zero-length trigger was present.
- Fixed a bug where automatic measure section length in QPU did not take readout pulse into account.
- Fixed a bug where creating an experiment in a different thread than where LabOne Q was originally imported caused a crash.

## Removals from the Codebase

- Removed the `update_qubits` and `update_quantum_elements` methods, which were deprecated in LabOne Q 26.1. Please use the `update` method instead.
- Removed the following methods from the `QPU` class: `copy_qubits` (deprecated in v2.52.0, please use `copy_quantum_elements` instead), `override_qubits` (deprecated in v2.52.0, please use `override_quantum_elements` instead), `qubit_by_uid` (deprecated in v2.52.0, please use `__getitem__` instead), `quantum_element_by_uid` (deprecated in v2.55.0, please use `__getitem__` instead). Removed the following attributes from the `QPU` class: `qubits` (deprecated in v2.52.0, please use `quantum_elements` instead), `_qubit_map` (deprecated in v2.52.0, please use `_quantum_element_map` instead).
- Removed the deprecated DataStore class and associated SQLite-based data storage functionality.


# LabOne Q 26.1.0 (2026-01-30)

## Features

- Produce more compact Sequencer C code in some situations involving nested sweeps.
- Introduced an async API for the controller, currently used internally only.

## Bug Fixes

- Fixed a bug where long readout (LRT) settings were not reset correctly for experiments where LRT is not used.
- Fixed a bug where conditional phase-increments at case-branch edges were not supported and where they leaked in case-branches with no defined pulses.
- Fixed a bug where the LabOne Q custom `showwarning` implementation could raise
  "RuntimeError: dictionary changed size during iteration" during imports that happen while
  traversing `sys.modules`. This is now much less likely.
- Renamed `signal_type` parameter in `create_connection()` in favor of `type`. The old parameter name will be removed in a future version.

# LabOne Q 26.1.0b4 (2026-01-15)

## Features

- Enable auto chunking of sweeps in case the hardware limitation is the amount of acquired results.
- Calculate result shapes at compile time and make them available via `compiled_experiment.scheduled_experiment.result_shape_info.shapes`.
- Transpose the last two dimensions of the shape of RAW acquisition results in case of multiple handles with the same name - before: `(..., handle, samples)`, after: `(..., samples, handle)`. This brings the axis corresponding to the multiple handles to the same location as for the case of non-RAW acquisition.
- Implemented Ramsey spectroscopy experiment under `laboneq.testing.experiments`.

## Bug Fixes

- Fixed a bug where file descriptors would leak when many sessions are created and destroyed within a single process. This can occur in long-running processes that repeatedly create sessions, or during testing when each test case creates its own session. Users are advised to call `session.disconnect()` to free resources.
- Fixed a bug where the compiler would output unnecessary log warnings about dropping the imaginary part of a waveform on RF signals.
- Fixed a bug where result shaping failed (array broadcasting error) in case of multiple different handles on the same signal and inside different case blocks.
- Fixed a bug where result shape and contents for acquisition commands inside match-case blocks were incorrect.
- Fixed a bug where result type was incorrect when there is only a single acquisition for a handle in the entire experiment. Before: `np.complex128`, After: `np.ndarray`.
- Fixed a bug where referencing an invalid section in 'play_after' in a right-aligned section did not raise an error.
- Fixed a bug where FolderStore deduplication treated numpy arrays that contain NaNs as unequal to themselves, storing them repeatedly.
- Fixed a bug where type validation did not support TypeAliasType from Python 3.12+ and numpy 2.4+.

## Documentation

- Added example and support code to perform subsampling delay adjustments.

# LabOne Q 26.1.0b3 (2025-12-18)

## Features

- Added type hint validation for signal calibration objects using typeguard. Assigning a value of the incorrect type to a calibration object will now immediately raise a TypeError rather than failing during compilation or execution.
- Improved unused sweep parameter detection. Compiler will now raise an error whenever experiment or device setup contains a sweep parameter not registered in any sweep,
  excluding derived sweep parameters where registering the parent is enough.

## Bug Fixes

- Fixed a bug where the HDAWG wouldn't start if previously-used cores were now unused.
- Fixed a bug where the local event loop was pinned to the thread instead of the session, preventing the same session from being invoked from multiple threads. The local event loop is now pinned to the session, enabling the same session to be invoked from multiple threads, as long as concurrent calls are properly synchronized by the user.
- Fixed a bug where execution would time out when reading large amounts of data, particularly in single-shot mode and over slow connections.
- Fixed a bug where data from unused integrators was accumulating until the end of the experiment, causing a potential memory overflow.
- Fixed a bug where Pipeliner was not executing when SHFQC was present in the setup, but not used in the chunked experiment.
- Fixed a bug where `play`-command without `pulse` and `markers` did not work when marker `waveform` was a sampled pulse.
- Fixed a bug where derived parameters were not registered correctly in near-time execution loops.
- Fixed a bug (introduced in v2.61.0) that prevented standalone HDAWG cores from syncing properly.
- Fixed a bug in the `FolderStore` serializer to support dictionaries of `QuantumElement` and `QuantumParameter` objects whose keys are tuples of strings, as explained in the FolderStore serializer documentation. Without this, supplying temporary topology edge parameters to `temporary_qpu` would fail to save the `temporary_parameters` passed, because edges are passed as tuples of the form (`tag`, `q0.uid`, `q1.uid`). For example, (`"coupler"`, `"q0"`, `"q1"`). With this change, `temporary_parameters` are saved correctly.
- Fixed a bug where using the first internal-only channel with the RTR option triggered a false exception.
- Fixed a bug where a SYNCIGNORED error in standalone SHFQC occurred when running a non-chunked experiment right after a chunked one in the same session.
- Fixed a bug where the error message was unclear when feedback acquisition or measure line 'port_delay' calibration is swept.

## Documentation

- Updated the FolderStore documentation to clarify that only dicts of QuantumElement or QuantumParameters whose keys are strings or tuples of strings may be serialized by the `FolderStore` serializer.

