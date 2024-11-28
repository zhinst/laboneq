# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.compiler.seqc.seqc_generator import SeqCGenerator


class PRNGTracker:
    def __init__(self, seqc_gen: SeqCGenerator):
        self.seqc_gen: SeqCGenerator = seqc_gen
        self._range: int | None = None
        self._seed: int | None = None
        self._offset: int = 0
        self._committed: bool = False
        self._active_sample: str | None = None

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, value):
        assert not self._committed
        self._range = value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        assert not self._committed
        self._seed = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value: int):
        assert not self._committed
        self._offset = value

    @property
    def active_sample(self):
        return self._active_sample

    @active_sample.setter
    def active_sample(self, value: str):
        assert isinstance(value, str)
        assert self._active_sample is None, "must first drop existing sample"
        self._active_sample = value

    def drop_sample(self):
        assert self._active_sample is not None, "no sample to drop"
        self._active_sample = None

    def is_committed(self) -> bool:
        return self._committed

    def commit(self):
        assert not self._committed

        if self._seed is not None:
            self.seqc_gen.add_function_call_statement(
                name="setPRNGSeed", args=[self._seed]
            )
        if self._range is not None:
            self.seqc_gen.add_function_call_statement(
                name="setPRNGRange",
                args=[self._offset, self._offset + self._range - 1],
            )

        # the tracker now has been spent, so clear it to prevent another call to `commit()`
        self._committed = True
