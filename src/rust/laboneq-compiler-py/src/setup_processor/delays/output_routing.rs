// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub(super) struct RoutedOutput {
    pub target: u16,
    pub channel: u16,
}

/// Delay in samples introduced by output routing on SHFSG devices.
pub(super) const OUTPUT_ROUTE_DELAY_SAMPLES: i64 = 52;

/// Calculate the delays introduced by output routing on SHFSG device.
///
/// Using output routing will introduce delay on both source and target channels,
/// where the both channels must be on the same device.
///
/// Returns an iterator of tuples where each tuple contains the channel and the corresponding delay in samples.
pub(super) fn calculate_output_route_delay(
    outputs: impl Iterator<Item = RoutedOutput>,
) -> impl Iterator<Item = (u16, i64)> {
    outputs.flat_map(|output| {
        vec![
            (output.target, OUTPUT_ROUTE_DELAY_SAMPLES),
            (output.channel, OUTPUT_ROUTE_DELAY_SAMPLES),
        ]
    })
}
