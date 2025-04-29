// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::cmp;
use std::collections::{HashSet, VecDeque};

pub mod interval;
use crate::interval::OrderedRange;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("{0}")]
    MinimumWaveformLengthViolation(String),

    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl OrderedRange<i64> {
    fn length(&self) -> i64 {
        self.0.end - self.0.start
    }
}

fn round_to_grid(intervals: &mut [OrderedRange<i64>], granularity: i64) {
    fn floor(value: i64, grid: i64) -> i64 {
        value - value % grid
    }

    fn ceil(value: i64, grid: i64) -> i64 {
        value + (grid - (value % grid)) % grid
    }

    intervals.iter_mut().for_each(|x| {
        x.0.start = floor(x.0.start, granularity);
        x.0.end = ceil(x.0.end, granularity);
    });
}

fn pass_left_to_right(
    chunk: &mut Vec<OrderedRange<i64>>,
    cut_interval: &OrderedRange<i64>,
    play_wave_size_hint: i64,
    play_zero_size_hint: i64,
    play_wave_maximum_size: i64,
) {
    let mut chunk_stack: VecDeque<OrderedRange<i64>> = VecDeque::from_iter(std::mem::take(chunk));
    let playzero = chunk_stack[0].0.start - cut_interval.0.start;
    if (0 < playzero) && (playzero < play_zero_size_hint) {
        // First playZero is too short. Extend first playWave to the left.
        let extended_length = chunk_stack[0].0.end - cut_interval.0.start;
        if extended_length <= play_wave_maximum_size {
            let chunk_0 = &mut chunk_stack[0];
            chunk_0.0.start = cut_interval.0.start;
        }
    }
    while let Some(mut iv) = chunk_stack.pop_front() {
        if iv.length() < play_wave_size_hint {
            iv.0.end = cmp::min(iv.0.start + play_wave_size_hint, cut_interval.0.end);
        }
        if let Some(next_iv) = chunk_stack.front_mut() {
            let gap_length = next_iv.0.start - iv.0.end;
            if (0 < gap_length) && (gap_length < play_zero_size_hint) {
                let extended_iv = next_iv.0.start - iv.0.start;
                if extended_iv <= play_wave_maximum_size {
                    iv.0.end = next_iv.0.start;
                }
            }
            if iv.overlaps_range(next_iv.0.start, next_iv.0.end) {
                let merged_length = next_iv.0.end - iv.0.start;
                if merged_length > play_wave_maximum_size {
                    iv.0.end = next_iv.0.start;
                } else {
                    next_iv.0.start = iv.0.start;
                    continue;
                }
            }
        } else {
            let gap_length = cut_interval.0.end - iv.0.end;
            if (0 < gap_length) && (gap_length < play_zero_size_hint) {
                let extended_iv = cut_interval.0.end - iv.0.start;
                if extended_iv <= play_wave_maximum_size {
                    iv.0.end = cut_interval.0.end;
                }
            }
        }
        chunk.push(iv);
    }
}

fn pass_right_to_left(
    chunk: &mut Vec<OrderedRange<i64>>,
    cut_interval: &OrderedRange<i64>,
    play_wave_size_hint: i64,
    play_zero_size_hint: i64,
) {
    chunk.reverse();
    let mut chunk_stack: VecDeque<OrderedRange<i64>> = VecDeque::from_iter(std::mem::take(chunk));
    while let Some(mut iv) = chunk_stack.pop_front() {
        if iv.length() < play_wave_size_hint {
            iv.0.start = cmp::max(iv.0.end - play_wave_size_hint, cut_interval.0.start);
        }
        if let Some(iv_prev) = chunk_stack.front_mut() {
            let gap_length = iv.0.start - iv_prev.0.end;
            if (0 < gap_length) && (gap_length < play_zero_size_hint) {
                iv.0.start = cmp::max(iv.0.end - play_zero_size_hint, cut_interval.0.start);
            }
            if iv.overlaps_range(iv_prev.0.start, iv_prev.0.end) {
                iv_prev.0.end = iv.0.end;
                continue;
            }
        } else {
            let gap_length = iv.0.start - cut_interval.0.start;
            if (0 < gap_length) && (gap_length < play_zero_size_hint) {
                iv.0.start = cmp::max(iv.0.end - play_zero_size_hint, cut_interval.0.start);
            }
        }
        chunk.push(iv);
    }
    chunk.reverse();
}

/// Merge interval overlaps in-place
fn merge_overlaps(intervals: &mut Vec<OrderedRange<i64>>) {
    let mut intervals_deque: VecDeque<OrderedRange<i64>> =
        VecDeque::from_iter(std::mem::take(intervals));
    if let Some(first) = intervals_deque.pop_front() {
        intervals.push(first);
        while let Some(interval) = intervals_deque.front() {
            let mut top = intervals.pop().unwrap();
            if top.0.end <= interval.0.start {
                let iv = intervals_deque.pop_front().unwrap();
                intervals.push(top);
                intervals.push(iv);
            } else if top.0.end < interval.0.end {
                top.0.end = interval.0.end;
                intervals_deque.pop_front();
                intervals.push(top);
            } else {
                intervals_deque.pop_front();
                intervals.push(top);
            }
        }
    }
}

/// Calculate intervals
///
/// Computes intervals (corresponding to eventual playWave statements in the
/// code) from pulses.
///
/// # Arguments
///
/// * intervals - Pulse playbacks as intervals
/// * cut_points - Timestamps of events that (probably) emit code. `intervals`
///   must not span across a cut point. The function assumes `cut_points` are
///   ordered. The gap between `cut_points` must be equal or larger than
///   `min_play_wave`.
/// * granularity - The waveform granularity of the hardware, i.e. waveform
///   lengths, hints and cut points must be a multiple of this number.
/// * min_play_wave - The hard lower limit on how long a playWave or playZero
///   can be
/// * play_wave_size_hint - Minimum length long we would like (but not require!)
///   playWave() to be
/// * play_zero_size_hint - Minimum length long we would like (but not require!)
///   playZero() to be
/// * play_wave_max_hint - An optional hint for maximum length on how long a
///   playWave or playZero can be. Must be larger than `min_play_wave`.
/// * ct_intervals - An optional collection of intervals within the `cut_points`
///   for which all pulses must be merged to one interval because they are
///   played as a single command table entry.
///
/// # Returns
///
/// Intervals that have been merged according to the rules below.
///
/// Respecting `min_play_wave` and `play_wave_max_hint` is hard requirement: if
/// they cannot be enforced, the algorithm fails loudly. By comparison
/// `play_wave_size_hint` and `play_zero_size_hint` are merely hints.
///
/// The calculation happens in three passes: Passes 1 & 2 target the hard min &
/// max waveform length, going over the segments from left-to-right and
/// right-to-left. Pass 3 targets the length hints, left-to-right.
///
/// In each pass, the algorithm keeps merging segments greedily until the total
/// length exceeds the desired length, and then continues with the next segment.
/// If it reaches the end of the pass, and no interval is left, it leaves the
/// last interval at its current length. By combining a left-to-right with a
/// right-to-left pass, all waveforms are guaranteed to exceed the
/// `min_play_wave` length, if possible at all. If two consecutive cut points
/// are spaced by less than `min_play_wave`, no solution may exist, and we fail.
///
/// In the third pass we also enforce the _maximum_ length. (In the first 2
/// passes, this is unnecessary; if we can't make one waveform long enough
/// without making another too long, all is lost.)
///
/// Note that at this time, we do not support pulses overlapping cut points.
/// This is returns an error.
#[allow(clippy::too_many_arguments)]
pub fn calculate_intervals(
    intervals: &[OrderedRange<i64>],
    cut_points: &[i64],
    granularity: i64,
    min_play_wave: i64,
    play_wave_size_hint: i64,
    play_zero_size_hint: i64,
    play_wave_max_hint: Option<i64>,
    ct_intervals: Option<&HashSet<OrderedRange<i64>>>,
) -> Result<Vec<OrderedRange<i64>>> {
    // TODO: Proper validation
    if intervals.is_empty() {
        return Ok(vec![]);
    }
    let play_wave_max_hint = play_wave_max_hint.unwrap_or(i64::MAX);
    if play_wave_max_hint != 0 && play_wave_max_hint <= play_wave_size_hint {
        return Err(anyhow::anyhow!("play_wave_max_hint <= play_wave_size_hint").into());
    }
    let mut intervals = intervals.to_owned();
    round_to_grid(&mut intervals, granularity);
    intervals.sort();
    merge_overlaps(&mut intervals);
    if cut_points.is_empty() {
        return Ok(intervals.to_vec());
    }
    if *cut_points.last().unwrap() < intervals.last().unwrap().0.end {
        return Err(
            anyhow::anyhow!("Last cut point must be equal or larger than intervals end").into(),
        );
    }
    // These intervals mark the regions delimited by the cut points. They are
    // independent, and *cannot* be merged.
    let start_point = cmp::min(intervals[0].0.start, cut_points[0]);
    let cut_point_intervals: Vec<_> = std::iter::once(&start_point)
        .chain(cut_points.iter())
        .collect();
    let mut interval_deque = VecDeque::from_iter(intervals);
    let mut intervals_out = vec![];
    for cut_pts in cut_point_intervals.windows(2).take(cut_points.len()) {
        // Necessary if 0 already in the list of cut points
        if cut_pts[0] == cut_pts[1] {
            continue;
        }
        let cut_interval = OrderedRange(*cut_pts[0]..*cut_pts[1]);
        if (cut_interval.length()) % granularity != 0 {
            return Err(anyhow::anyhow!("Cut point does not match granularity").into());
        }
        let mut chunk: Vec<OrderedRange<i64>> = vec![];
        // Collect and drain overlapping intervals until cut point interval is exceeded
        while let Some(front) = interval_deque.front() {
            if cut_interval.overlaps_range(front.0.start, front.0.end) {
                if front.0.start < cut_interval.0.start || cut_interval.0.end < front.0.end {
                    return Err(anyhow::anyhow!("Cut points overlap intervals").into());
                };
                chunk.push(interval_deque.pop_front().unwrap());
            } else {
                break;
            }
        }
        if chunk.is_empty() {
            continue;
        }
        if cut_interval.length() < min_play_wave {
            return Err(Error::MinimumWaveformLengthViolation(
                "Cut points are spaced more closely than the minimum waveform length".to_string(),
            ));
        }
        let mpw = match ct_intervals {
            Some(x) => match x.contains(&cut_interval) {
                true => cut_interval.length(),
                false => min_play_wave,
            },
            None => min_play_wave,
        };
        pass_left_to_right(&mut chunk, &cut_interval, mpw, mpw, i64::MAX);
        pass_right_to_left(&mut chunk, &cut_interval, mpw, mpw);
        pass_left_to_right(
            &mut chunk,
            &cut_interval,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
        );
        intervals_out.extend(chunk);
    }
    intervals_out.sort();
    // TODO: Proper validation
    Ok(intervals_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_interval() {
        let ivs = vec![OrderedRange(0..1024)];
        let cut_points = vec![0, 2304];
        let granularity = 8;
        let min_play_wave = 16;
        let play_wave_size_hint = 640;
        let play_zero_size_hint = 640;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        let intervals = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            ct_intervals,
        )
        .unwrap();

        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].0, 0..1024);
    }

    #[test]
    fn test_touching_intervals_matches_granularity() {
        let ivs = vec![OrderedRange(0..16), OrderedRange(16..32)];
        let cut_points = vec![];
        let granularity = 16;
        let min_play_wave = 16;
        let play_wave_size_hint = 16;
        let play_zero_size_hint = 16;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        let intervals = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            ct_intervals,
        )
        .unwrap();

        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].0, 0..16);
        assert_eq!(intervals[1].0, 16..32);
    }

    #[test]
    fn test_intervals_merged_due_granularity() {
        let ivs = vec![
            OrderedRange(0..16),
            OrderedRange(16..32),
            OrderedRange(32..64),
        ];
        let cut_points = vec![];
        let granularity = 64;
        let min_play_wave = 16;
        let play_wave_size_hint = 16;
        let play_zero_size_hint = 16;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        let intervals = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            ct_intervals,
        )
        .unwrap();

        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].0, 0..64);
    }

    #[test]
    fn test_intervals_with_gap() {
        let ivs = vec![OrderedRange(0..32), OrderedRange(64..128)];

        let cut_points = vec![32, 128];
        let granularity = 8;
        let min_play_wave = 8;
        let play_wave_size_hint = 8;
        let play_zero_size_hint = 8;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        let intervals = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            ct_intervals,
        )
        .unwrap();

        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].0, 0..32);
        assert_eq!(intervals[1].0, 64..128);
    }

    #[test]
    fn test_overlapping_intervals() {
        let ivs = vec![
            OrderedRange(0..32),
            OrderedRange(16..24),
            OrderedRange(24..64),
        ];
        let cut_points = vec![0, 64];
        let granularity = 8;
        let min_play_wave = 8;
        let play_wave_size_hint = 8;
        let play_zero_size_hint = 8;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        let intervals = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            ct_intervals,
        )
        .unwrap();

        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].0, 0..64);
    }

    #[test]
    fn test_non_zero_begin() {
        let ivs = vec![OrderedRange(160..360)];
        let cut_points = vec![160, 400];
        let granularity = 16;
        let min_play_wave = 32;
        let play_wave_size_hint = 0;
        let play_zero_size_hint = 64;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        let intervals = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            ct_intervals,
        )
        .unwrap();

        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].0, 160..400);
    }

    #[test]
    fn test_short_interval_before_cut_point() {
        // This is for e.g. loops where at the end of the loop is a very short pulse
        let ivs = vec![
            OrderedRange(192..432),
            OrderedRange(432..434),
            OrderedRange(448..688),
            OrderedRange(688..690),
            OrderedRange(704..944),
            OrderedRange(944..946),
        ];

        let cut_points = vec![192, 448, 704, 960, 1216];
        let granularity = 16;
        let min_play_wave = 32;
        let play_wave_size_hint = 128;
        let play_zero_size_hint = 128;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        let intervals = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            ct_intervals,
        )
        .unwrap();

        assert_eq!(intervals.len(), 3);
        assert_eq!(intervals[0].0, 192..448);
        assert_eq!(intervals[1].0, 448..704);
        assert_eq!(intervals[2].0, 704..960);
    }

    #[test]
    fn test_cut_point_cuts_interval() {
        let ivs = vec![OrderedRange(0..8)];
        let cut_points = vec![4, 8];
        let granularity = 4;
        let min_play_wave = 4;
        let play_wave_size_hint = 4;
        let play_zero_size_hint = 4;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        assert!(
            calculate_intervals(
                &ivs,
                &cut_points,
                granularity,
                min_play_wave,
                play_wave_size_hint,
                play_zero_size_hint,
                play_wave_max_hint,
                ct_intervals,
            )
            .is_err_and(|e| e.to_string() == "Cut points overlap intervals".to_string())
        );
    }

    #[test]
    fn test_cut_point_granularity_mismatch() {
        let ivs = vec![OrderedRange(0..8)];
        let cut_points = vec![0, 9];
        let granularity = 4;
        let min_play_wave = 4;
        let play_wave_size_hint = 4;
        let play_zero_size_hint = 4;
        let play_wave_max_hint = None;
        let ct_intervals = None;
        assert!(
            calculate_intervals(
                &ivs,
                &cut_points,
                granularity,
                min_play_wave,
                play_wave_size_hint,
                play_zero_size_hint,
                play_wave_max_hint,
                ct_intervals,
            )
            .is_err_and(|e| e.to_string() == "Cut point does not match granularity".to_string())
        );
    }

    #[test]
    fn test_ct_intervals() {
        let ivs = vec![
            OrderedRange(170..700),
            OrderedRange(50000..51000),
            OrderedRange(52000..53000),
        ];
        let cut_points = vec![160, 2112, 2496, 53008];
        let granularity = 16;
        let min_play_wave = 32;
        let play_wave_size_hint = 512;
        let play_zero_size_hint = 512;
        let play_wave_max_hint = None;
        let mut ct_intervals = HashSet::new();
        ct_intervals.insert(OrderedRange(2496..53008));
        let out = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            Some(&ct_intervals),
        )
        .unwrap();
        // CT interval forces intervals to be merged
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 160..704);
        assert_eq!(out[1].0, 2496..53008);

        let out = calculate_intervals(
            &ivs,
            &cut_points,
            granularity,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            play_wave_max_hint,
            None,
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, 160..704);
        assert_eq!(out[1].0, 50000..51008);
        assert_eq!(out[2].0, 52000..53008);
    }
}
