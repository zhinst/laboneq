// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::sampled_event_handler::{
    AwgEvent,
    awg_events::{EventType, TriggerOutput},
};

use crate::ir;

/// Resample state at given times, when state changes are known at independently defined times.
///
/// Given a progression of state changes over discrete (integer) time, specified as a SortedDict of (time, state),
/// and a collection of events, given as an iterable containing the times at which to sample, determine, for each event,
/// the state at the time the event happens.
/// Returns a SortedDict of (time, state_tuple) where the times are the times from the events,
/// and state_tuple is the corresponding (time, state) tuple taken from the state_progression.
///
/// The algorithm iterates only once through both input collections.
///
fn resample_states<
    'a,
    S: Iterator<Item = &'a ir::Samples> + ExactSizeIterator,
    T: Copy + 'a,
    P: Iterator<Item = &'a (ir::Samples, T)>,
>(
    sample_times: &mut S,
    state_progression: &mut P,
) -> Vec<(ir::Samples, (ir::Samples, T))> {
    let mut retval = Vec::with_capacity(sample_times.len());

    // If there are no samples or no states, return an empty vector
    let mut current_state = match state_progression.next() {
        Some(state) => state,
        None => return retval,
    };
    let mut current_sample_time = match sample_times.next() {
        Some(time) => time,
        None => return retval,
    };
    // Check if the earliest known state change is after the first sample time
    if current_state.0 > *current_sample_time {
        panic!(
            "Internal error: state undefined at time {}, earliest known state at {}",
            current_sample_time, current_state.0
        );
    }
    let mut next_state = state_progression.next();
    loop {
        // If there is no next state change or the current sample time is before the next state change
        if next_state.is_none_or(|next_state| *current_sample_time < next_state.0) {
            // If the current sample time is after or equal to the current state change time
            assert!(*current_sample_time >= current_state.0);
            retval.push((*current_sample_time, *current_state));
            let previous_sample_time = current_sample_time;
            // Try to get the next sample time
            match sample_times.next() {
                Some(next_sample_time) => {
                    // Check if the sample times are in ascending order
                    if next_sample_time < previous_sample_time {
                        panic!(
                            "Internal error: Sample times must be sorted in ascending order, but current time {next_sample_time} is before previous time {previous_sample_time}"
                        );
                    }
                    current_sample_time = next_sample_time;
                }
                // Exit the loop if there are no more sample times
                None => break,
            }
        } else {
            let current_state_time = current_state.0;
            current_state = next_state.unwrap();
            next_state = state_progression.next();
            if let Some(next_state) = next_state {
                assert!(
                    current_state_time < next_state.0,
                    "Internal error: state progression must be sorted in ascending order"
                );
            }
        }
    }

    retval
}

/// Consolidate trigger output events happening at the same time and
/// track the state of the trigger output.
pub fn generate_trigger_states(events: &mut Vec<AwgEvent>) {
    if events.is_empty() {
        return;
    }
    let mut processed_events = Vec::with_capacity(events.len());
    let mut current_trigger_state: u16 = 0;
    let mut state_progression = vec![];
    let mut loop_push_timestamps = vec![];

    let mut iter = std::mem::take(events).into_iter().peekable();
    while let Some(next_event_peek) = iter.peek() {
        let current_timestamp = next_event_peek.start;
        let mut timestamp_had_trigger_bits = false;
        let mut events_at_this_timestamp = vec![];
        while let Some(next_event_peek) = iter.peek() {
            if next_event_peek.start == current_timestamp {
                let event = iter.next().unwrap();
                events_at_this_timestamp.push(match event.kind {
                    EventType::TriggerOutputBit(data) => {
                        timestamp_had_trigger_bits = true;
                        let mask = 1 << data.bit;
                        if data.set {
                            current_trigger_state |= mask;
                        } else {
                            current_trigger_state &= !mask;
                        }
                        AwgEvent {
                            start: event.start,
                            end: event.end,
                            kind: EventType::TriggerOutput(TriggerOutput {
                                state: current_trigger_state,
                            }),
                        }
                    }
                    EventType::PushLoop(ref data) => {
                        if data.compressed {
                            loop_push_timestamps.push(event.start);
                        }
                        event
                    }
                    _ => event,
                });
            } else {
                break;
            }
        }
        if timestamp_had_trigger_bits {
            // Remove all TriggerOutputs events except the last one
            // (which is the one with the final state).
            let mut have_seen_trigger_output = false;
            let filtered: Vec<_> = events_at_this_timestamp
                .into_iter()
                .rev()
                .filter(|event| {
                    if let EventType::TriggerOutput(t) = &event.kind {
                        if have_seen_trigger_output {
                            return false;
                        }
                        have_seen_trigger_output = true;
                        state_progression.push((event.start, t.state));
                    }
                    true
                })
                .collect();
            processed_events.extend(filtered.into_iter().rev());
        } else {
            processed_events.extend(events_at_this_timestamp);
        }
    }
    *events = processed_events;
    if let Some(first_state) = state_progression.first()
        && first_state.0 != 0
    {
        state_progression.insert(0, (0, 0));
    }
    // When the trigger is raised at the end of the averaging loop (which is not
    // unrolled), things get a bit dicey: the command to reset the trigger signal must
    // be deferred to the next iteration. Which means that the very first iteration must
    // already include this trigger reset, and that we must issue it again after the loop.

    let resampled_states = resample_states(
        &mut loop_push_timestamps.iter(),
        &mut state_progression.iter(),
    );
    if resampled_states.is_empty() {
        return;
    }
    let mut new_events: Vec<AwgEvent> = Vec::with_capacity(events.len() + resampled_states.len());
    let mut resampled_states_iter = resampled_states.iter();
    for event in events.drain(..) {
        if let EventType::PushLoop(ref data) = event.kind {
            if data.compressed {
                let (time_in_samples, (_state_time, state)) = *resampled_states_iter.next().expect("Internal error: Number of loop push timestamps does not match number of resampled states");
                new_events.push(event);
                new_events.push(AwgEvent {
                    start: time_in_samples,
                    end: time_in_samples,
                    kind: EventType::TriggerOutput(TriggerOutput { state }),
                });
            }
        } else {
            new_events.push(event);
        }
    }
    assert!(
        resampled_states_iter.next().is_none(),
        "Internal error: Not all resampled states were used"
    );
    *events = new_events;
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_resample_state() {
        let state_progression: Vec<(i64, &'static str)> = vec![
            (28, "d"),
            (43, "d"),
            (109, "g"),
            (111, "h"),
            (132, "f"),
            (142, "a"),
            (145, "j"),
            (164, "i"),
            (198, "d"),
            (206, "c"),
            (230, "i"),
            (235, "c"),
            (249, "i"),
            (253, "g"),
            (271, "f"),
            (275, "d"),
            (292, "c"),
            (330, "f"),
            (333, "a"),
            (337, "f"),
            (361, "f"),
            (375, "c"),
            (423, "i"),
            (429, "a"),
            (432, "b"),
            (433, "b"),
            (451, "a"),
            (460, "g"),
            (478, "d"),
        ];
        let sample_times: Vec<ir::Samples> =
            vec![32, 93, 268, 337, 356, 357, 470, 474, 704, 711, 762];
        let resampled_states = resample_states(
            &mut sample_times.clone().iter(),
            &mut state_progression.iter(),
        );
        assert_eq!(resampled_states.len(), sample_times.len());
        assert!(
            resampled_states
                .iter()
                .zip(sample_times.iter())
                .all(|(a, b)| a.0 == *b)
        );
        for (k, v) in resampled_states.iter() {
            assert!(state_progression.contains(v));

            // Compare with naive O(n^2) algorithm
            let mut correct_state: Option<(i64, &'static str)> = None;
            for state_time in state_progression.iter() {
                if state_time.0 <= *k
                    && (correct_state.is_none_or(|correct_state| state_time.0 > correct_state.0))
                {
                    correct_state = Some(*state_time);
                }
            }
            assert_eq!(correct_state, Some(*v));
        }
    }

    #[test]
    fn test_single_state() {
        let state_progression: Vec<(i64, &'static str)> = vec![(28, "d")];
        let sample_times: Vec<ir::Samples> =
            vec![32, 93, 268, 337, 356, 357, 470, 474, 704, 711, 762];
        let resampled_states = resample_states(
            &mut sample_times.clone().iter(),
            &mut state_progression.iter(),
        );
        assert_eq!(resampled_states.len(), sample_times.len());
        assert!(
            resampled_states
                .iter()
                .zip(sample_times.iter())
                .all(|(a, b)| a.0 == *b)
        );
        assert!(resampled_states.iter().all(|a| a.1 == (28, "d")))
    }

    #[test]
    fn test_undefined_state() {
        let sample_times: Vec<ir::Samples> = vec![32];
        let state_progression: Vec<(i64, &'static str)> = vec![(200, "d")];
        let sample_times_iter = sample_times.iter();
        let state_progression_iter = state_progression.iter();
        // Suppress panic output for this test
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(|| {
            resample_states(
                &mut sample_times_iter.clone(),
                &mut state_progression_iter.clone(),
            );
        });
        std::panic::set_hook(default_hook);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_samples() {
        let sample_times: Vec<ir::Samples> = vec![];
        let state_progression: Vec<(i64, &'static str)> = vec![
            (28, "d"),
            (43, "d"),
            (109, "g"),
            (111, "h"),
            (132, "f"),
            (142, "a"),
            (145, "j"),
            (164, "i"),
        ];
        let resampled_states = resample_states(
            &mut sample_times.clone().iter(),
            &mut state_progression.iter(),
        );
        assert_eq!(resampled_states.len(), 0);
    }

    #[test]
    fn test_no_states() {
        let sample_times: Vec<ir::Samples> = vec![];
        let state_progression: Vec<(i64, &'static str)> = vec![];
        let resampled_states = resample_states(
            &mut sample_times.clone().iter(),
            &mut state_progression.iter(),
        );
        assert_eq!(resampled_states.len(), 0);
    }

    #[test]
    fn test_non_ascending_sampling() {
        let sample_times: Vec<ir::Samples> = vec![30, 40, 50, 1];

        let state_progression: Vec<(i64, &'static str)> = vec![
            (28, "d"),
            (43, "d"),
            (109, "g"),
            (111, "h"),
            (132, "f"),
            (142, "a"),
            (145, "j"),
            (164, "i"),
        ];
        let sample_times_iter = sample_times.iter();
        let state_progression_iter = state_progression.iter();
        // Suppress panic output for this test
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(|| {
            resample_states(
                &mut sample_times_iter.clone(),
                &mut state_progression_iter.clone(),
            );
        });
        std::panic::set_hook(default_hook);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_ascending_states() {
        let sample_times: Vec<ir::Samples> = vec![32, 40, 50];

        let state_progression: Vec<(i64, &'static str)> = vec![(28, "d"), (43, "d"), (1, "g")];
        let sample_times_iter = sample_times.iter();
        let state_progression_iter = state_progression.iter();
        // Suppress panic output for this test
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(|| {
            resample_states(
                &mut sample_times_iter.clone(),
                &mut state_progression_iter.clone(),
            );
        });
        std::panic::set_hook(default_hook);
        assert!(result.is_err());
    }
}
