# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import zip_longest

import numpy as np

from laboneq.core.types.numpy_support import NumPyArray
from laboneq.data.experiment_results import AcquiredResult, ExperimentResults
from laboneq.data.recipe import NtStepKey
from laboneq.data.scheduled_experiment import (
    HandleResultShape,
    ResultSource,
    ScheduledExperiment,
)


class ResultsBuilder:
    def __init__(
        self,
        shapes: dict[str, HandleResultShape],
        result_handle_maps: dict[ResultSource, list[set[str]]],
        chunk_count: int | None,
    ):
        self._shapes = shapes
        self._result_handle_maps = result_handle_maps
        self._chunk_count = chunk_count

        self._results = ResultsBuilder._init_empty_result_by_shape(shapes)

    @property
    def results(self) -> ExperimentResults:
        return self._results

    @staticmethod
    def from_scheduled_experiment(
        scheduled_experiment: ScheduledExperiment,
    ) -> ResultsBuilder:
        result_shape_info = scheduled_experiment.result_shape_info
        chunk_count = scheduled_experiment.rt_loop_properties.chunk_count
        return ResultsBuilder(
            result_shape_info.shapes, result_shape_info.result_handle_maps, chunk_count
        )

    @staticmethod
    def _init_empty_result_by_shape(
        result_shapes: dict[str, HandleResultShape],
    ) -> ExperimentResults:
        results = ExperimentResults()
        for handle, shape_info in result_shapes.items():
            empty_result = AcquiredResult(
                data=np.full(
                    shape=tuple(shape_info.shape),
                    fill_value=np.nan,
                    dtype=np.complex128,
                ),
                axis_name=shape_info.axis_names,
                axis=shape_info.axis_values,
                handle=handle,
            )
            results.acquired_results[handle] = empty_result
        return results

    def _determine_slices(
        self, nt_step: NtStepKey, chunk_index: int | None, handle: str
    ) -> tuple[slice | list[int], ...]:
        shape_info = self._shapes[handle]
        slices: list[slice | list[int]] = []
        for i, (dim, nt_idx) in enumerate(
            zip_longest(shape_info.shape, nt_step.indices)
        ):
            slc: slice | list[int]
            if nt_idx is not None:
                slc = [nt_idx]
            elif (
                shape_info.match_case_mask is not None
                and (rows := shape_info.match_case_mask.get(i)) is not None
            ):
                slc = rows
            else:
                # take everything in the axis
                slc = slice(0, dim)

            if shape_info.chunked_axis_index == i and chunk_index is not None:
                # slice the dimension further, to account for a single chunk only
                assert self._chunk_count is not None
                if isinstance(slc, list):
                    chunk_len = len(slc) // self._chunk_count
                    slc = slc[chunk_index * chunk_len : (chunk_index + 1) * chunk_len]
                else:
                    start, stop = slc.start or 0, slc.stop
                    assert stop is not None
                    assert slc.step is None
                    chunk_len = (stop - start) // self._chunk_count
                    slc = slice(
                        start + chunk_index * chunk_len,
                        start + (chunk_index + 1) * chunk_len,
                    )
            slices.append(slc)
        return tuple(slices)

    def add_acquired_data(
        self,
        nt_step: NtStepKey,
        chunk_index: int | None,
        result_source: ResultSource,
        acquired_data: NumPyArray,
    ):
        """Add data acquired from a single full (i.e. all chunks) RT execution, or a single chunk (chunk_index not None).

        Populates self._results in-place with acquired_data. acquired_data is what is obtained from the instrument
        and can be thought of as a flat sequence of data instances, where each instance corresponds to an acquisition event.
        Most of the time each such instance is a single complex number, except for the case of RAW acquisition, in which case
        it is a list of complex numbers.
        self._results is a buffer for the complete result to be returned to the user, that contains NaNs in
        locations where this function is about to populate. acquired_data itself may contain NaNs when acquiring failed, e.g.
        execution of one chunk errored out.
        For each handle, the buffer is a multi-dimensional numpy array. This function determines the slices of this array where
        the incoming data should go, and populates accordingly.
        """
        mapping = self._result_handle_maps.get(result_source, [])
        unique_handles = set(h for m in mapping for h in m)
        for handle in unique_handles:
            result_for_handle = self._results.acquired_results[handle]
            handle_shape_info = self._shapes[handle]
            assert result_for_handle.data is not None, (
                "Result data shape is not prepared"
            )
            result_for_handle.last_nt_step = list(nt_step.indices)
            acquired_data_len = len(acquired_data)
            handle_mask = np.fromiter(
                (handle in mapping[i % len(mapping)] for i in range(acquired_data_len)),
                dtype=np.bool,
                count=acquired_data_len,
            )
            acquired_data_for_handle = acquired_data[handle_mask]

            slices = self._determine_slices(nt_step, chunk_index, handle)
            sliced_shape = result_for_handle.data[slices].shape

            if (acquired_len := np.prod(acquired_data_for_handle.shape)) != (
                expected_len := np.prod(sliced_shape)
            ):
                # This is allowed to happen only in RAW acquisition mode.
                # When multiple signals are feeding from the same source, the number of acquired RAW samples
                # is the max over all such signals. Thus, for some signals we have more samples than expected/requested.
                # In such a case we assume that the acquire events across signals were aligned at the beginning, so we
                # truncate the excess data from the end.
                assert acquired_len > expected_len
                assert acquired_data_for_handle.ndim == 2  # i.e. it is RAW acquisition
                truncation_index = expected_len // acquired_data_for_handle.shape[0]
                acquired_data_for_handle = acquired_data_for_handle[
                    :, :truncation_index
                ]

            if self._chunk_count is None or chunk_index is not None:
                # if chunking is not used, or used and the incomming
                # data is for a single chunk only

                shaped_acquired_data_for_handle = np.reshape(
                    acquired_data_for_handle,
                    sliced_shape,
                    copy=False,
                )
            else:
                # chunking is used, and we have the data for all the chunks

                chunked_axis_index = handle_shape_info.chunked_axis_index
                assert chunked_axis_index is not None
                assert chunk_index is None
                assert self._chunk_count is not None

                chunk_shape = tuple(
                    dim if i != chunked_axis_index else dim // self._chunk_count
                    for i, dim in enumerate(sliced_shape)
                )
                # rearrange the data as if it came from a non-chunked experiment
                shaped_acquired_data_for_handle = np.concatenate(
                    np.reshape(
                        acquired_data_for_handle,
                        (self._chunk_count, *chunk_shape),
                        copy=False,
                    ),
                    axis=chunked_axis_index,
                )

            result_for_handle.data[slices] = shaped_acquired_data_for_handle

    def add_pipeline_jobs_timestamps(self, signal, metadata):
        timestamps = self._results.pipeline_jobs_timestamps.setdefault(signal, [])

        for job_id, v in metadata.items():
            # make sure the list is long enough for this job id
            timestamps.extend([float("nan")] * (job_id - len(timestamps) + 1))
            timestamps[job_id] = v["timestamp"]

    def add_execution_error(self, error: tuple[list[int], str, str]):
        self._results.execution_errors.append(error)
