# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table

from laboneq.compiler.common.iface_compiler_output import (
    CombinedOutput,
    RTCompilerOutputContainer,
)
from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.workflow.compiler_output import (
    CombinedRTCompilerOutputContainer,
)
from laboneq.data.scheduled_experiment import CodegenWaveform
from laboneq.laboneq_logging import get_logger

_logger = get_logger(__name__)


from laboneq.compiler.workflow.neartime_execution import (
    NtCompilerExecutorDelegate,
    NtCompilerExecutor,
)


def _count_samples(waves: dict[str, CodegenWaveform], wave_index):
    multiplier = 1
    wave_name, (_, wave_type) = wave_index
    if wave_name == "precomp_reset":
        # for precomp reset we use an all-zero waveform that is not explicitly
        # listed in the waveform table
        if wave_type in ("iq", "double", "multi"):
            return 64
        return 32
    if wave_type in ("iq", "double", "multi"):
        waveform_name = f"{wave_name}_i.wave"
        multiplier = 2  # two samples per clock cycle
    elif wave_type == "complex":
        waveform_name = f"{wave_name}.wave"
        multiplier = 2  # two samples per clock cycle
    elif wave_type == "single":
        waveform_name = f"{wave_name}.wave"
    else:
        raise ValueError("invalid wave type")

    waveform = waves[waveform_name]
    return len(waveform.samples) * multiplier


@dataclass(order=True)
class ReportEntry:
    nt_step_indices: tuple[int, ...]
    awg: Optional[AwgKey]
    seqc_loc: int
    command_table_entries: int
    wave_indices: int
    waveform_samples: int


class CompilationReportGenerator(NtCompilerExecutorDelegate):
    def __init__(self, settings: CompilerSettings):
        self._settings = settings
        self._data: list[ReportEntry] = []
        self._total: ReportEntry | None = None
        self._require_long_readout: str | None = None

        self._pulse_waveform_count = {}
        self._pulse_map = {}

    def after_compilation_run(self, new: RTCompilerOutputContainer, indices: list[int]):
        self.update(new, indices)

    def after_final_run(self, combined: CombinedRTCompilerOutputContainer):
        from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC

        for co in combined.combined_output.values():
            if isinstance(co, CombinedRTOutputSeqC):
                compiler_output = co
                break
        else:
            return
        if self._settings.LOG_REPORT:
            self.compute_pulse_map_statistics(compiler_output)
            self.calculate_total(compiler_output)
            self.log_report()

    def update(
        self, rt_compiler_output: RTCompilerOutputContainer, step_indices: list[int]
    ):
        from laboneq.compiler.seqc.linker import SeqCGenOutput

        for co in rt_compiler_output.codegen_output.values():
            if isinstance(co, SeqCGenOutput):
                compiler_output = co
                break
        else:
            return
        report = []
        for awg, awg_src in compiler_output.src.items():
            seqc_loc = len(awg_src["text"].splitlines())
            ct = compiler_output.command_tables.get(awg, {"ct": []})["ct"]
            ct_len = len(ct)
            wave_indices = compiler_output.wave_indices[awg]["value"]
            wave_indices_count = len(wave_indices)
            sample_count = 0
            for wave_index in wave_indices.items():
                sample_count += _count_samples(compiler_output.waves, wave_index)
            report.append(
                ReportEntry(
                    awg=awg,
                    nt_step_indices=tuple(step_indices),
                    seqc_loc=seqc_loc,
                    command_table_entries=ct_len,
                    wave_indices=wave_indices_count,
                    waveform_samples=sample_count,
                )
            )
        self._data += report

    def calculate_total(self, compiler_output: CombinedOutput):
        from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC

        assert isinstance(compiler_output, CombinedRTOutputSeqC)
        total_seqc = sum(len(s["text"].splitlines()) for s in compiler_output.src)
        total_ct = sum(len(ct["ct"]) for ct in compiler_output.command_tables)
        total_wave_idx = sum(len(wi) for wi in compiler_output.wave_indices)
        total_samples = sum(
            _count_samples(compiler_output.waves, wi)
            for wil in compiler_output.wave_indices
            for wi in wil["value"].items()
        )
        self._total = ReportEntry(
            nt_step_indices=(),
            awg=None,
            seqc_loc=total_seqc,
            command_table_entries=total_ct,
            wave_indices=total_wave_idx,
            waveform_samples=total_samples,
        )
        require_long_readout = [
            id
            for id, lrt in compiler_output.requires_long_readout.items()
            if len(lrt) > 0
        ]
        if len(require_long_readout) > 0:
            self._require_long_readout = ", ".join(require_long_readout)

    def compute_pulse_map_statistics(self, compiler_output: CombinedOutput):
        from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC

        assert isinstance(compiler_output, CombinedRTOutputSeqC)
        for pulse_id, pulse_map in compiler_output.pulse_map.items():
            self._pulse_map[pulse_id] = (
                len(pulse_map.waveforms),
                [i for wf in pulse_map.waveforms.values() for i in wf.instances],
            )

    def create_resource_usage_table(self) -> Table:
        entries = sorted(self._data)
        all_nt_steps = set(e.nt_step_indices for e in entries)
        include_nt_step = len(all_nt_steps) > 1

        table = Table(box=box.HORIZONTALS)

        if include_nt_step:
            table.add_column("Step")

        table.add_column("Device")
        table.add_column("AWG", justify="right")
        table.add_column("SeqC LOC", justify="right")
        table.add_column("CT entries", justify="right")
        table.add_column("Waveforms", justify="right")
        table.add_column("Samples", justify="right")

        previous_nt_step = None
        previous_device = None

        for entry in entries:
            cells = []
            if include_nt_step:
                if entry.nt_step_indices != previous_nt_step:
                    table.add_section()
                    cells.append(",".join(str(n) for n in entry.nt_step_indices))
                    previous_nt_step = entry.nt_step_indices
                else:
                    cells.append("")

            if previous_device != entry.awg.device_id or not include_nt_step:
                device_cell = f"{entry.awg.device_id}"
            else:
                device_cell = ""

            assert entry.awg is not None
            cells.extend(
                [
                    device_cell,
                    f"{entry.awg.awg_id}",
                    f"{entry.seqc_loc}",
                    f"{entry.command_table_entries}",
                    f"{entry.wave_indices}",
                    f"{entry.waveform_samples}",
                ]
            )
            table.add_row(*cells)

        if self._total is not None:
            table.show_footer = True
            cells = []
            if include_nt_step:
                cells.append("")
            cells.extend(
                [
                    "TOTAL",  # device id
                    "",  # AWG idx
                    f"{self._total.seqc_loc}",
                    f"{self._total.command_table_entries}",
                    "",  #  f"{self._total.wave_indices}",
                    f"{self._total.waveform_samples}",
                ]
            )
            for column, cell in zip(table.columns, cells):
                column.footer = cell

        return table

    def resource_table_as_str(self):
        table = self.create_resource_usage_table()

        with StringIO() as buffer:
            console = Console(file=buffer, force_jupyter=False)
            console.print(table)
            return buffer.getvalue()

    def create_pulse_map_table(self):
        table = Table(box=box.HORIZONTALS)
        table.add_column("Pulse UID")
        table.add_column("Waveforms", justify="right")
        table.add_section()
        table.add_column("Offsets", justify="right")
        table.add_column("Amplitudes", justify="right")
        table.add_column("Phases", justify="right")
        table.add_column("Lengths", justify="right")

        for pulse_id, (waveform_count, instances) in self._pulse_map.items():
            offsets_count = len({inst.offset_samples for inst in instances})
            amplitudes_count = len({inst.amplitude for inst in instances})
            phases_count = len({inst.iq_phase for inst in instances})
            lengths_count = len({inst.length for inst in instances})
            fields = [
                str(n) if n > 1 else ""
                for n in [offsets_count, amplitudes_count, phases_count, lengths_count]
            ]

            table.add_row(pulse_id, str(waveform_count), *fields)

        return table

    def pulse_map_table_as_str(self):
        table = self.create_pulse_map_table()

        with StringIO() as buffer:
            console = Console(file=buffer, force_jupyter=False)
            console.print(table)
            return buffer.getvalue()

    def log_report(self):
        if self._require_long_readout is not None:
            _logger.info(f"Require(s) long readout: {self._require_long_readout}")
        for line in self.resource_table_as_str().splitlines():
            _logger.info(line)

        _logger.diagnostic("")
        _logger.diagnostic("  Waveform usage across pulses:")
        for line in self.pulse_map_table_as_str().splitlines():
            _logger.diagnostic(line)


NtCompilerExecutor.register_hook(CompilationReportGenerator)
