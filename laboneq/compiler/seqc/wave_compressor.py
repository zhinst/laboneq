# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from itertools import groupby
from typing import Dict, List, Tuple, Union

import numpy as np


@dataclass
class PlayHold:
    num_samples: int


@dataclass
class PlaySamples:
    samples: Dict[str, np.array]
    label: int


class WaveCompressor:
    def __init__(self):
        self.wave_number = 0

    def _samples_constant(self, samples: np.ndarray) -> bool:
        return np.all(samples[0] == samples)

    def _sample_dict_constant(self, sample_dict: Dict[str, np.array]) -> bool:
        return all(self._samples_constant(sample) for sample in sample_dict.values())

    def _stacked_samples_constant(self, stacked_samples: np.ndarray, lo: int, hi: int):
        return np.all(
            np.all(
                stacked_samples[:, lo + 1 : hi] == stacked_samples[:, lo : hi - 1],
                axis=0,
            )
        )

    def _frames_compatible(
        self, last_vals_frame: Dict[str, np.array], sample_dict: Dict[str, np.array]
    ) -> bool:
        return all(
            last_vals_frame[k] == sample_dict[k][0] for k in last_vals_frame.keys()
        )

    def _merge_samples(self, samples: List[Dict[str, np.array]]) -> Dict[str, np.array]:
        ret = {}
        for k in samples[0].keys():
            ret.setdefault(k, np.array([]))

        for sample in samples:
            for k in sample.keys():
                ret[k] = np.concatenate((ret[k], sample[k]))

        return ret

    def _get_frame(
        self, sample_dict: Dict[str, np.array], size: int, i: int, num_frames: int
    ) -> Dict[str, np.array]:
        if i == num_frames - 1:
            return {k: sample_dict[k][i * size :] for k in sample_dict}
        return {k: v[i * size : (i + 1) * size] for k, v in sample_dict.items()}

    def _get_frame_idx(
        self, sample_dict: Dict[str, np.array], size: int, i: int, num_frames: int
    ) -> Tuple[int, int]:
        if i == num_frames - 1:
            return (i * size, len(list(sample_dict.values())[0]))
        return (i * size, (i + 1) * size)

    def _runs_longer_than_threshold(self, stacked_samples: np.ndarray, threshold: int):
        runs = []

        # Calculate differences between consecutive columns
        diffs = np.diff(stacked_samples, axis=1)

        # Find the indices where consecutive columns change
        changes = np.nonzero(diffs != 0)[1] + 1

        # Add the first and last indices for runs longer than the threshold
        if stacked_samples.shape[1] > threshold:
            changes = np.concatenate(([0], changes, [stacked_samples.shape[1]]))

        # Compute the lengths of each run
        run_lengths = np.diff(changes)

        # Find indices of runs longer than the threshold
        long_runs_indices = np.where(run_lengths > threshold)[0]

        # Compute the start and end indices of long runs
        start_indices = changes[long_runs_indices]
        end_indices = start_indices + run_lengths[long_runs_indices] - 1

        # Create a list of tuples (start, end) for each long run
        runs = list(zip(start_indices, end_indices))

        return runs

    def _compress_wave_simple(
        self,
        samples: Dict[str, np.array],
        sample_multiple: int,
        ref_length: int,
        run: Tuple[int, int],
    ) -> Union[List[Union[PlayHold, PlaySamples]], None]:
        start_run, end_run = run

        start_run_on_grid = (start_run // sample_multiple + 1) * sample_multiple
        end_run_on_grid = (end_run // sample_multiple - 1) * sample_multiple

        compression_length = (
            end_run - start_run_on_grid
            if end_run == ref_length
            else end_run_on_grid - start_run_on_grid
        )

        if compression_length < sample_multiple:
            return None

        events = []
        events.append(
            PlaySamples(
                samples={k: v[0:start_run_on_grid] for k, v in samples.items()},
                label=self.wave_number,
            )
        )
        self.wave_number += 1
        if end_run == ref_length - 1:
            events.append(PlayHold(num_samples=int(ref_length - start_run_on_grid)))
            return events
        else:
            events.append(
                PlayHold(num_samples=int(end_run_on_grid - start_run_on_grid))
            )
            events.append(
                PlaySamples(
                    samples={
                        k: v[end_run_on_grid:ref_length] for k, v in samples.items()
                    },
                    label=self.wave_number,
                )
            )
            self.wave_number += 1
            return events

    def _compress_wave_general(
        self,
        samples: Dict[str, np.array],
        stacked_samples: np.ndarray,
        compressable_segments: List[Tuple[int, int]],
        num_sample_channles: int,
        num_frames: int,
        num_samples: int,
        sample_multiple: int,
    ) -> Union[List[Union[PlayHold, PlaySamples]], None]:
        last_vals = np.zeros((num_sample_channles, num_frames))
        for i in range(0, num_frames):
            _, hi = self._get_frame_idx(samples, sample_multiple, i, num_frames)
            last_vals[:, i] = stacked_samples[:, hi - 1]

        compressable_frames = []
        for seg_lo, seg_hi in compressable_segments:
            frame_lo = seg_lo // sample_multiple + 1
            frame_hi = (
                num_frames if seg_hi == num_samples else seg_hi // sample_multiple - 1
            )
            compressable_frames.append((frame_lo, frame_hi))

        can_compress = [False] * num_frames
        for frame_lo, frame_hi in compressable_frames:
            for i in range(frame_lo, frame_hi):
                lo, hi = self._get_frame_idx(samples, sample_multiple, i, num_frames)
                can_compress[i] = self._stacked_samples_constant(
                    stacked_samples, lo, hi
                ) and np.all(last_vals[:, i - 1] == stacked_samples[:, lo])

        if not any(can_compress):
            return None

        events = []
        sequences = [
            (key, list(group))
            for key, group in groupby(
                range(num_frames), lambda frame_idx: can_compress[frame_idx]
            )
        ]

        for can_compress, sequence in sequences:
            if can_compress:
                num_samples = 0
                for frame_idx in sequence:
                    lo, hi = self._get_frame_idx(
                        samples, sample_multiple, frame_idx, num_frames
                    )
                    num_samples += hi - lo
                events.append(PlayHold(num_samples=num_samples))
            else:
                play_samples = [
                    self._get_frame(samples, sample_multiple, frame_idx, num_frames)
                    for frame_idx in sequence
                ]
                events.append(
                    PlaySamples(
                        samples=self._merge_samples(samples=play_samples),
                        label=self.wave_number,
                    )
                )
                self.wave_number += 1

        return events

    def compress_wave(
        self,
        samples: Dict[str, np.array],
        sample_multiple: int,
        compressible_segments: List[Tuple[int, int]] | None = None,
    ) -> Union[List[Union[PlayHold, PlaySamples]], None]:
        ref_length = len(list(samples.values())[0])
        num_sample_channles = len(list(samples.values()))
        if not all(len(v) == ref_length for v in samples.values()):
            raise ValueError("All sample arrays must have the same length")
        num_frames = int(ref_length / sample_multiple)
        num_samples = ref_length

        compressible_segments = (
            [(0, num_samples)]
            if compressible_segments is None
            else compressible_segments
        )

        stacked_samples = np.array(list(samples.values()))

        if len(compressible_segments) > 1:
            return self._compress_wave_general(
                samples,
                stacked_samples,
                compressible_segments,
                num_sample_channles,
                num_frames,
                num_samples,
                sample_multiple,
            )

        compr_start, compr_end = compressible_segments[0]
        runs = self._runs_longer_than_threshold(
            stacked_samples[:, compr_start:compr_end], 32
        )
        if len(runs) == 0:
            return None
        if len(runs) == 1:
            runs = [(run[0] + compr_start, run[1] + compr_start) for run in runs]
            return self._compress_wave_simple(
                samples, sample_multiple, ref_length, runs[0]
            )
        return self._compress_wave_general(
            samples,
            stacked_samples,
            compressible_segments,
            num_sample_channles,
            num_frames,
            num_samples,
            sample_multiple,
        )
