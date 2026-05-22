# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.dsl.experiment.experiment import Experiment


def convert_signal_map(experiment: Experiment) -> dict[str, str]:
    return {
        signal.uid: signal.mapped_logical_signal_path
        for signal in experiment.signals.values()
        if signal.mapped_logical_signal_path is not None
    }
