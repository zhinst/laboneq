# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attr

from laboneq.compiler.common.shfppc_sweeper_config import (
    SweepCommand,
    SHFPPCSweeperConfig,
)


@attr.define(kw_only=True, slots=True)
class SweepStackFrame:
    count: int
    items: list[SweepCommand | SweepStackFrame] = attr.field(factory=list)


class SHFPPCSweeperConfigTracker:
    def __init__(self):
        self._command_tree: SweepStackFrame = SweepStackFrame(count=1)
        self._stack: list[SweepStackFrame] = [self._command_tree]
        self._ppc_device: str | None = None
        self._ppc_channel: int | None = None

    def has_sweep_commands(self) -> bool:
        return self._ppc_device is not None and self._ppc_channel is not None

    @property
    def _current_frame(self) -> SweepStackFrame:
        return self._stack[-1]

    def add_step(self, ppc_device: str, ppc_channel: int, **kwargs):
        self._current_frame.items.append(SweepCommand(**kwargs))
        self._ppc_device = ppc_device
        self._ppc_channel = ppc_channel

    def enter_loop(self, count: int):
        new_frame = SweepStackFrame(count=count)
        self._current_frame.items.append(new_frame)
        self._stack.append(new_frame)

    def exit_loop(self):
        closed_frame = self._stack.pop()
        # if the frame we are closing is empty, we don't need to keep it
        if len(closed_frame.items) == 0:
            self._current_frame.items.pop()

    def ppc_channel(self) -> int:
        assert self._ppc_channel is not None, "must add a step first"
        return self._ppc_channel

    def ppc_device(self) -> str:
        assert self._ppc_device is not None, "must add a step first"
        return self._ppc_device

    def _flatten(self, command_tree: SweepStackFrame) -> SHFPPCSweeperConfig:
        """Remove any nesting from the command tree

        After flattening, `items` will only hold `SweepCommand`s. The repetition
        count may get adjusted to efficiently represent any internal loops.
        """

        commands: list[SweepCommand | SHFPPCSweeperConfig] = []
        count = command_tree.count

        for c in command_tree.items:
            if isinstance(c, SweepStackFrame):
                assert len(c.items) > 0, "empty frames should have been dropped earlier"
                commands.append(self._flatten(c))
            else:
                commands.append(c)

        if len(commands) == 1 and isinstance(commands[0], SHFPPCSweeperConfig):
            child = commands[0]
            commands = child.commands
            count *= child.count
        else:
            if any(isinstance(c, SHFPPCSweeperConfig) for c in commands):
                raise AssertionError("Nesting rolled loops is not supported")
                # If we wanted to support them, we have to unroll them here.

        assert all(isinstance(c, dict) for c in commands), "items must be SweepCommand"
        return SHFPPCSweeperConfig(
            count=count,
            commands=commands,
            ppc_device=self.ppc_device(),
            ppc_channel=self.ppc_channel(),
        )

    def finish(self) -> SHFPPCSweeperConfig:
        return self._flatten(self._command_tree)
