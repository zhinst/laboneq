# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
import concurrent.futures
from dataclasses import dataclass
import logging
import os

import zhinst.core
from zhinst.core.errors import CoreError as LabOneCoreError

from laboneq.core.exceptions.laboneq_exception import LabOneQException

_logger = logging.getLogger(__name__)


@dataclass
class SeqCCompileItem:
    dev_type: str
    dev_opts: list[str]
    awg_index: int
    sequencer: str
    sampling_rate: float | None
    filename: str
    code: str
    elf: bytes | None = None


def seqc_compile_one(item: SeqCCompileItem):
    if item.code is None:
        return
    try:
        elf, extra = zhinst.core.compile_seqc(
            item.code,
            item.dev_type,
            options=item.dev_opts,
            index=item.awg_index,
            sequencer=item.sequencer,
            filename=item.filename,
            samplerate=item.sampling_rate,
        )
    except LabOneCoreError as exc:
        raise LabOneQException(
            f"{item.filename}: AWG compilation failed.\n{str(exc)}"
        ) from None

    compiler_warnings = extra["messages"]
    if compiler_warnings:
        raise LabOneQException(
            f"AWG compilation succeeded, but there are warnings:\n{compiler_warnings}"
        )

    item.elf = elf


async def seqc_compile_async(item: SeqCCompileItem):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, seqc_compile_one, item)


def awg_compile(awg_data: list[SeqCCompileItem]):
    # Compile in parallel:
    _logger.debug("Started compilation of AWG programs...")
    max_workers_str = os.environ.get("LABONEQ_AWG_COMPILER_MAX_WORKERS")
    max_workers = None if max_workers_str is None else int(max_workers_str)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(seqc_compile_one, item) for item in awg_data]
        concurrent.futures.wait(futures)
        exceptions = [
            future.exception() for future in futures if future.exception() is not None
        ]
        if len(exceptions) > 0:
            errors = "\n".join([str(e) for e in exceptions])
            raise LabOneQException(f"Compilation failed.\n{errors}")
    _logger.debug("Finished compilation.")
