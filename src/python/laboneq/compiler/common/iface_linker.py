# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc


class ILinker(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def combined_from_single_run(output, step_indices: list[int]):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def merge_combined_compiler_runs(this, new, previous, step_indices: list[int]):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def repeat_previous(this, previous):
        raise NotImplementedError
