# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Logs the resource usage of the QCCS backend's compiled programs."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, cast

from rich import box
from rich.console import Console
from rich.table import Table

from laboneq.core.types.enums.wave_type import WaveType
from laboneq.data.awg_info import AwgKey
from laboneq.laboneq_logging import get_logger

if TYPE_CHECKING:
    from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC
    from laboneq.data.artifacts_qccs import CodegenWaveform

_logger = get_logger(__name__)


def log_report(compiler_output: CombinedRTOutputSeqC):
    """Logs the resource usage of the compiled program."""
    require_long_readout = [
        device
        for device, signals in compiler_output.requires_long_readout.items()
        if signals
    ]
    if require_long_readout:
        _logger.info(f"Require(s) long readout: {', '.join(require_long_readout)}")
    for line in _render(_resource_usage_table(compiler_output)).splitlines():
        _logger.info(line)

    _logger.diagnostic("")
    _logger.diagnostic("  Waveform usage across pulses:")
    for line in _render(_pulse_map_table(compiler_output)).splitlines():
        _logger.diagnostic(line)


@dataclass(order=True)
class _ProgramUsage:
    nt_step_indices: tuple[int, ...]
    awg: AwgKey
    seqc_loc: int
    command_table_entries: int
    waveforms: int
    samples: int


def _resource_usage_table(compiler_output: CombinedRTOutputSeqC) -> Table:
    usages = sorted(_program_usages(compiler_output))
    include_nt_step = len({u.nt_step_indices for u in usages}) > 1

    table = Table(box=box.HORIZONTALS, show_footer=True)
    if include_nt_step:
        table.add_column("Step")
    table.add_column("Device", footer="TOTAL")
    table.add_column("AWG", justify="right")
    table.add_column(
        "SeqC LOC", justify="right", footer=str(sum(u.seqc_loc for u in usages))
    )
    table.add_column(
        "CT entries",
        justify="right",
        footer=str(sum(u.command_table_entries for u in usages)),
    )
    table.add_column("Waveforms", justify="right")
    table.add_column(
        "Samples", justify="right", footer=str(sum(u.samples for u in usages))
    )

    previous_nt_step = None
    for usage in usages:
        step_cell = []
        if include_nt_step:
            if usage.nt_step_indices != previous_nt_step:
                table.add_section()
                previous_nt_step = usage.nt_step_indices
                step_cell = [",".join(str(n) for n in usage.nt_step_indices)]
            else:
                step_cell = [""]
        table.add_row(
            *step_cell,
            usage.awg.device_id,
            str(usage.awg.awg_id),
            str(usage.seqc_loc),
            str(usage.command_table_entries),
            str(usage.waveforms),
            str(usage.samples),
        )
    return table


def _program_usages(compiler_output: CombinedRTOutputSeqC) -> list[_ProgramUsage]:
    command_tables = {
        ct["seqc"]: ct["ct"]["table"] for ct in compiler_output.command_tables
    }
    wave_indices = {
        cast("str", wi["filename"]): cast(
            "dict[str, tuple[int, WaveType]]", wi["value"]
        )
        for wi in compiler_output.wave_indices
    }
    return [
        _ProgramUsage(
            nt_step_indices=step.key.indices,
            awg=AwgKey(step.device_id, step.awg_id),
            seqc_loc=len(compiler_output.src[step.seqc_ref].src.splitlines()),
            command_table_entries=len(command_tables.get(step.seqc_ref, [])),
            waveforms=len(wave_indices[step.wave_indices_ref]),
            samples=sum(
                _count_samples(compiler_output.waves, wave_index)
                for wave_index in wave_indices[step.wave_indices_ref].items()
            ),
        )
        for step in compiler_output.neartime_steps
    ]


def _count_samples(
    waves: dict[str, CodegenWaveform], wave_index: tuple[str, tuple[int, WaveType]]
):
    multiplier = 1
    wave_name, (_, wave_type) = wave_index
    if wave_name == "precomp_reset":
        # for precomp reset we use an all-zero waveform that is not explicitly
        # listed in the waveform table
        if wave_type in (WaveType.IQ, WaveType.DOUBLE):
            return 64
        return 32
    if wave_type in (WaveType.IQ, WaveType.DOUBLE):
        waveform_name = f"{wave_name}_i.wave"
        multiplier = 2  # two samples per clock cycle
    elif wave_type == WaveType.COMPLEX:
        waveform_name = f"{wave_name}.wave"
        multiplier = 2  # two samples per clock cycle
    elif wave_type == WaveType.SINGLE:
        waveform_name = f"{wave_name}.wave"
    else:
        raise ValueError(f"invalid wave type {wave_type}")

    waveform = waves[waveform_name]
    return len(waveform.samples) * multiplier


def _pulse_map_table(compiler_output: CombinedRTOutputSeqC) -> Table:
    table = Table(box=box.HORIZONTALS)
    table.add_column("Pulse UID")
    table.add_column("Waveforms", justify="right")
    table.add_column("Offsets", justify="right")
    table.add_column("Amplitudes", justify="right")
    table.add_column("Phases", justify="right")
    table.add_column("Lengths", justify="right")

    for pulse_id, pulse_map in compiler_output.pulse_map.items():
        instances = [i for wf in pulse_map.waveforms.values() for i in wf.instances]
        variant_counts = [
            len({i.offset_samples for i in instances}),
            len({i.amplitude for i in instances}),
            len({i.iq_phase for i in instances}),
            len({i.length for i in instances}),
        ]
        table.add_row(
            pulse_id,
            str(len(pulse_map.waveforms)),
            *(str(n) if n > 1 else "" for n in variant_counts),
        )
    return table


def _render(table: Table) -> str:
    with StringIO() as buffer:
        Console(file=buffer, force_jupyter=False).print(table)
        return buffer.getvalue()
