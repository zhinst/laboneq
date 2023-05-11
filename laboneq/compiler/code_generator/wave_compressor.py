# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from itertools import groupby
from typing import Dict, List, Union

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

    def compress_wave(
        self, samples: Dict[str, np.array], sample_multiple: int
    ) -> Union[List, None]:
        ref_length = len(list(samples.values())[0])
        if not all(len(v) == ref_length for v in samples.values()):
            raise ValueError("All sample arrays must have the same length")
        num_frames = int(ref_length / sample_multiple)

        last_vals = []
        for i in range(0, num_frames):
            sample_frame = self._get_frame(samples, sample_multiple, i, num_frames)
            last_vals.append({k: sample_frame[k][-1] for k in sample_frame})

        can_compress = [False] * num_frames
        for i in range(1, num_frames):
            sample_frame = self._get_frame(samples, sample_multiple, i, num_frames)
            can_compress[i] = self._sample_dict_constant(
                sample_frame
            ) and self._frames_compatible(last_vals[i - 1], sample_frame)

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
                    frame = self._get_frame(
                        samples, sample_multiple, frame_idx, num_frames
                    )
                    num_samples += len(frame[next(iter(frame.keys()))])
                events.append(PlayHold(num_samples=num_samples))
            else:
                play_samples = []
                for frame_idx in sequence:
                    play_samples.append(
                        self._get_frame(samples, sample_multiple, frame_idx, num_frames)
                    )
                events.append(
                    PlaySamples(
                        samples=self._merge_samples(samples=play_samples),
                        label=self.wave_number,
                    )
                )
                self.wave_number += 1

        return events
