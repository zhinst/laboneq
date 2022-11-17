# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import copy
from math import ceil
from typing import Dict, Any, TYPE_CHECKING, NewType
from numpy import gcd
from laboneq.compiler.common.device_type import DeviceType

if TYPE_CHECKING:
    from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO

PrecompensationType = NewType("PrecompensationType", Dict[str, Dict[str, Any]])


def precompensation_is_nonzero(precompensation: PrecompensationType):
    """Check whether the precompensation has any effect"""
    return precompensation is not None and (
        precompensation.get("exponential") is not None
        and any([e["amplitude"] != 0 for e in precompensation["exponential"]])
        or precompensation.get("high_pass") is not None
        or precompensation.get("bounce") is not None
        and precompensation["bounce"]["amplitude"] != 0
        or precompensation.get("FIR") is not None
        and any(c != 0 for c in precompensation["FIR"]["coefficients"])
    )


def precompensation_delay_samples(precompensation: PrecompensationType):
    """Compute the additiontal delay (in samples) caused by the precompensation"""
    if not precompensation_is_nonzero(precompensation):
        return 0
    delay = 72
    try:
        delay += 88 * len(precompensation["exponential"])
    except KeyError:
        pass
    if precompensation.get("high_pass") is not None:
        delay += 96
    if precompensation.get("bounce") is not None:
        delay += 32
    if precompensation.get("FIR") is not None:
        delay += 136
    return delay


def _adapt_precompensations_of_awg(signal_ids, precompensations):
    # If multiple signals per AWG, find the union of all filter enables
    number_of_exponentials = 0
    has_high_pass = None
    has_bounce = False
    has_FIR = False
    for signal_id in signal_ids:
        precompensation = precompensations.get(signal_id) or {}
        hp = bool(precompensation.get("high_pass"))
        if has_high_pass is None:
            has_high_pass = hp
        else:
            if hp != has_high_pass:
                raise RuntimeError(
                    "All precompensation settings for "
                    + "outputs of the same AWG must have the high pass "
                    + f"filter enabled or disabled; see signal {signal_id}"
                )
        exp = precompensation.get("exponential")
        if exp is not None and number_of_exponentials < len(exp):
            number_of_exponentials = len(exp)
        has_bounce = has_bounce or bool(precompensation.get("bounce"))
        has_FIR = has_FIR or bool(precompensation.get("FIR"))
    # Add zero effect filters to get consistent timing
    if has_bounce or has_FIR or number_of_exponentials:
        for signal_id in signal_ids:
            old_pc = precompensations.get(signal_id, {}) or {}
            new_pc = copy.deepcopy(old_pc)
            if number_of_exponentials:
                exp = new_pc.setdefault("exponential", [])
                exp += [{"amplitude": 0, "timeconstant": 10e-9}] * (
                    number_of_exponentials - len(exp)
                )
            if has_bounce and not new_pc.get("bounce"):
                new_pc["bounce"] = {"delay": 10e-9, "amplitude": 0}
            if has_FIR and not new_pc.get("FIR"):
                new_pc["FIR"] = {"coefficients": []}
            precompensations[signal_id] = new_pc


def adapt_precompensations(precompensations: PrecompensationType, dao: ExperimentDAO):
    """Make sure that we have the same timing for rf_signals on the same AWG"""
    signals_by_awg = {}
    # Group by AWG
    for signal_id in precompensations.keys():
        signal_info = dao.signal_info(signal_id)
        device_id = signal_info.device_id
        device_type = DeviceType(signal_info.device_type)
        channel = signal_info.channels[0]
        awg = (
            0
            if device_type == DeviceType.UHFQA
            else channel // device_type.channels_per_awg
        )
        signals_by_awg.setdefault((device_id, awg), []).append(signal_id)
    for signal_ids in signals_by_awg.values():
        if len(signal_ids) > 1:
            _adapt_precompensations_of_awg(signal_ids, precompensations)


def compute_precompensations_and_delays(dao: ExperimentDAO):
    """Retrieve precompensations from DAO, adapt those on the same AWG and
    compute timing"""
    precompensations = {
        id: copy.deepcopy(dao.precompensation(id)) for id in dao.signals()
    }
    adapt_precompensations(precompensations, dao)
    for signal_id, pc in precompensations.items():
        delay = precompensation_delay_samples(pc)
        pc = precompensations.setdefault(signal_id, {})
        if pc is None:
            precompensations[signal_id] = {"computed_delay_samples": delay}
        else:
            pc["computed_delay_samples"] = delay
    return precompensations


def compute_precompensation_delays_on_grid(
    precompensations: PrecompensationType, dao: ExperimentDAO, use_2GHz: bool
):
    """Compute delay_signal and port_delay contributions for each signal so that delays
    are commensurable with the grid"""
    signals = dao.signals()
    if not signals:
        return
    signal_infos = {
        signal_id: dao.signal_info(signal_id) for signal_id in dao.signals()
    }
    unique_sequencer_rates = set()
    sampling_rates_and_multiples = {}
    for signal_id in signals:
        devtype = DeviceType(signal_infos[signal_id].device_type)
        sampling_rate = (
            devtype.sampling_rate_2GHz
            if use_2GHz and devtype == DeviceType.HDAWG
            else devtype.sampling_rate
        )
        sequencer_rate = sampling_rate / devtype.sample_multiple
        sampling_rates_and_multiples[signal_id] = (
            sampling_rate,
            devtype.sample_multiple,
        )
        unique_sequencer_rates.add(int(sequencer_rate))

    common_sequencer_rate = gcd.reduce(list(unique_sequencer_rates))
    system_grid = 1.0 / common_sequencer_rate

    max_delay = 0
    for signal_id, pc in precompensations.items():
        delay = (
            precompensations[signal_id]["computed_delay_samples"]
            / sampling_rates_and_multiples[signal_id][0]
        )
        if max_delay < delay:
            max_delay = delay
    max_delay = ceil(max_delay / system_grid) * system_grid

    for signal_id in signals:
        pc = precompensations.setdefault(signal_id, {})
        try:
            delay_samples = pc["computed_delay_samples"]
        except KeyError:
            delay_samples = 0
        sampling_rate, multiple = sampling_rates_and_multiples[signal_id]
        max_delay_samples = max_delay * sampling_rate
        compensation = max_delay_samples - delay_samples
        delay_signal = (compensation // multiple) / sampling_rate * multiple
        port_delay = (compensation % multiple) / sampling_rate
        assert port_delay == 0 or signal_infos[signal_id].device_type != "uhfqa"
        pc["computed_delay_signal"] = delay_signal if abs(delay_signal) > 1e-12 else 0
        pc["computed_port_delay"] = port_delay if abs(port_delay) > 1e-12 else 0
