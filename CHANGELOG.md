# LabOne Q 26.4.0b1 (2026-02-12)

## Features

- Session now supports automatic system profile fetching and caching when connecting to hardware. Use the new `system_profile` parameter in the Session constructor to explicitly provide a profile, or let it auto-load from cache when needed.
- Ported event list generation (required for Pulse Sheet Viewer) from Python to Rust for improved performance.

## Bug Fixes

- Fixed a bug where a read timeout occurred when running LRT after non-LRT, due to LabOne Q enabling MSD by default even for two states for non-LRT experiments. MSD is now explicitly disabled for LRT.
- Fixed a bug where hardware modulation was not applied when using a short readout on SHFQA with the LRT option.
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

