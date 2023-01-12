# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from typing import Any, Iterable, List, Optional

from intervaltree import Interval, IntervalTree

from laboneq.compiler.fastlogging import NullLogger

_logger = logging.getLogger(__name__)
if _logger.getEffectiveLevel() == logging.DEBUG:
    _dlogger = _logger
else:
    _logger.info("Debug logging disabled for %s", __name__)
    _dlogger = NullLogger()


def are_cut_points_valid(interval_tree: IntervalTree, cut_points: List[int]):
    for cut_point in cut_points:
        at_cut_point = interval_tree.at(cut_point)
        for iv in at_cut_point:
            if iv.begin < cut_point < iv.end:
                _logger.warning("Cut point %s would cut interval %s", cut_point, iv)
                return False
    return True


def ceil(value: int, grid: int):
    return value + (-value) % grid


def floor(value: int, grid: int):
    return value - value % grid


def merge_overlaps(interval_tree: IntervalTree):
    interval_tree.merge_overlaps(data_reducer=lambda data1, data2: [*data1, *data2])


@dataclasses.dataclass
class MutableInterval:
    """Mutable proxy for intervaltree.Interval

    Cannot actually live in an IntervalTree."""

    begin: int
    end: Optional[int]
    data: Any = dataclasses.field(default=None)

    def overlaps(self, other):
        return self.immutable().overlaps(other.immutable())

    def immutable(self) -> Interval:
        return Interval(self.begin, self.end, self.data)


def _pass_left_to_right(
    chunk: List[MutableInterval],
    cut_interval: Interval,
    play_wave_size_hint: int,
    play_zero_size_hint: int,
    play_wave_maximum_size: int = 0,
) -> List[MutableInterval]:
    first_playback = chunk[0].begin
    if play_wave_maximum_size:
        assert play_wave_maximum_size > play_wave_size_hint

    if play_wave_maximum_size == 0:
        play_wave_maximum_size = float("inf")
    if 0 < first_playback - cut_interval.begin < play_zero_size_hint:
        # First playZero is too short. Extend first playWave to the left.
        extended_length = chunk[0].end - cut_interval.begin
        if extended_length <= play_wave_maximum_size:
            chunk[0].begin = cut_interval.begin

    new_intervals = []

    for iv, next_iv in zip(
        chunk, [*chunk[1:], MutableInterval(cut_interval.end, float("inf"))]
    ):
        next_iv: MutableInterval
        playback_length = iv.end - iv.begin
        if playback_length < play_wave_size_hint:
            iv.end = min(iv.begin + play_wave_size_hint, cut_interval.end)

        gap_length = next_iv.begin - iv.end
        if 0 < gap_length < play_zero_size_hint:
            extended_length = next_iv.begin - iv.begin
            if extended_length <= play_wave_maximum_size:
                iv.end = next_iv.begin

        if iv.overlaps(next_iv):
            merged_length = next_iv.end - iv.begin
            if merged_length > play_wave_maximum_size:
                iv.end = next_iv.begin
            else:
                next_iv.begin = iv.begin
                next_iv.data = [*iv.data, *next_iv.data]
                continue
        new_intervals.append(iv)

    return new_intervals


def _pass_right_to_left(
    chunk: List[MutableInterval],
    cut_interval: Interval,
    play_wave_size_hint: int,
    play_zero_size_hint: int,
) -> List[MutableInterval]:
    new_intervals = []
    for iv, previous_iv in zip(
        chunk[::-1],
        [*chunk[-2::-1], MutableInterval(-float("inf"), cut_interval.begin)],
    ):
        previous_iv: MutableInterval
        playback_length = iv.end - iv.begin
        if playback_length < play_wave_size_hint:
            iv.begin = max(iv.end - play_wave_size_hint, cut_interval.begin)

        gap_length = iv.begin - previous_iv.end
        if 0 < gap_length < play_zero_size_hint:
            iv.begin = max(iv.end - play_zero_size_hint, cut_interval.begin)

        if iv.overlaps(previous_iv):
            previous_iv.end = iv.end
            previous_iv.data = [*previous_iv.data, *iv.data]
            continue

        new_intervals.append(iv)

    new_intervals.reverse()
    return new_intervals


class MinimumWaveformLengthViolation(ValueError):
    pass


def calculate_intervals(
    interval_tree: IntervalTree,
    min_play_wave: int,
    play_wave_size_hint: int,
    play_zero_size_hint: int,
    play_wave_max_hint: int,
    cut_points: List[int],
    granularity: int = 16,
    force_command_table_intervals: Iterable[MutableInterval] = {},
):
    """
    Compute intervals (corresponding to eventual playWave statements in the code) from
    pulses. Merge pulses into waveforms.

    Args:
        interval_tree: The interval tree containing pulse playback as intervals
        min_play_wave: the hard lower limit on how long a playWave or playZero can be
        play_wave_size_hint: minimum length long we would like (but not require!)
            playWave() to be
        play_zero_size_hint: minimum length long we would like (but not require!)
            playZero() to be
        play_wave_max_hint: hint maximum length on how long a playWave or playZero can be
            (pass 0 to ignore)
        cut_points: Timestamps of events that (probably) emit code. A merged waveform
            must not span across a cut point.
        force_command_table_intervals: A collection of intervals for which all pulses
            must be merged to one interval because they are played as a single command
            table entry.
        granularity: The waveform granularity of the hardware, i.e. waveform lengths
            must be a multiple of this number.

    Returns:
        A new interval tree where intervals have been merged according to the rules
        below. Data of each interval is a list of the data of the original intervals
        that were merged together.

    Respecting `min_play_wave` and `max_play_wave` is hard requirement: if they cannot
    be enforced, the algorithm fails loudly. By comparison `play_wave_size_hint` and
    `play_zero_size_hint` are merely hints.

    The calculation happens in three passes: Passes 1 & 2 target the hard min & max
    waveform length, going over the segments from left-to-right and right-to-left.
    Pass 3 targets the length hints, left-to-right.

    In each pass, the algorithm keeps merging segments greedily until the total length
    exceeds the desired length, and then continues with the next segment. If it reaches
    the end of the pass, and no interval is left, it leaves the last interval at its
    current length. By combining a left-to-right with a right-to-left pass, all
    waveforms are guaranteed to exceed the `min_play_wave` length, if possible at all.
    If two consecutive cut points are spaced by less than `min_play_wave`, no solution
    may exist, and we fail.

    In the third pass we also enforce the _maximum_ length. (In the first 2 passes, this
    is unnecessary; if we can't make one waveform long enough without making another too
    long, all is lost.)

    Note that at this time, we do not support pulses overlapping cut points.
    """
    if interval_tree.is_empty():
        return interval_tree

    assert all(cut_point % granularity == 0 for cut_point in cut_points)
    assert min_play_wave % granularity == 0
    assert play_wave_max_hint % granularity == 0
    assert play_wave_size_hint % granularity == 0
    for interval in force_command_table_intervals:
        assert min_play_wave <= interval.end - interval.begin
    assert are_cut_points_valid(interval_tree, cut_points)
    assert all(
        iv.end <= cut_points[-1] for iv in interval_tree
    )  # last cut point is end of sequence

    new_tree = IntervalTree()

    for iv in sorted(interval_tree.items()):
        begin = floor(iv.begin, granularity)
        end = ceil(iv.end, granularity)
        new_tree.addi(begin, end, [iv.data])

    interval_tree = new_tree

    merge_overlaps(interval_tree)

    # These intervals mark the regions delimited by the cut points. They are
    # independent, and *cannot* be merged.
    cut_intervals = [
        Interval(begin, end)
        for begin, end in zip([0, *cut_points[:-1]], [*cut_points])
        if begin != end  # necessary if 0 already in the list of cut points
    ]

    retval = IntervalTree()
    command_table_intervals = set(
        (m.begin, m.end) for m in force_command_table_intervals
    )

    for cut_interval in cut_intervals:

        # We may merge intervals inside this chunk, but they must not extend past it.
        chunk = [
            MutableInterval(i.begin, i.end, i.data)
            for i in sorted(interval_tree.overlap(cut_interval.begin, cut_interval.end))
        ]
        if len(chunk) == 0:
            continue

        if cut_interval.length() < min_play_wave:
            # Need playback, but can't fit a single waveform? Not happening!
            raise MinimumWaveformLengthViolation

        mpw = (
            (cut_interval.end - cut_interval.begin)
            if (cut_interval.begin, cut_interval.end) in command_table_intervals
            else min_play_wave
        )
        chunk = _pass_left_to_right(chunk, cut_interval, mpw, mpw)
        chunk = _pass_right_to_left(chunk, cut_interval, mpw, mpw)
        chunk = _pass_left_to_right(
            chunk,
            cut_interval,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
        )

        for interval in chunk:
            retval.add(interval.immutable())

    # Check if the scheduling was successful
    all_intervals = sorted(retval.items())
    for iv, previous_iv in zip(
        all_intervals, [*all_intervals[1:], Interval(cut_points[-1], None)]
    ):
        # playWave not too short?
        assert iv.length() >= min_play_wave

        # gap (playZero) not too short?
        assert not (0 < previous_iv.begin - iv.end < min_play_wave)
        # Note: here we do not care about possible cut points in between. If that is
        # indeed a problem it will be caught when emitting code.
        # We do not check for this here because cut points might not *actually* emit
        # code (e.g. end points of an unrolled loop), so it *could* be fine.

    assert are_cut_points_valid(retval, cut_points)

    return retval
