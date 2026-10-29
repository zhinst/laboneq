# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import attrs

from laboneq.controller.api.async_controller_api import AsyncControllerAPI
from laboneq.controller.api.commons import SubmissionRegistry, to_results
from laboneq.controller.constants import DEFAULT_TIMEOUT_S, READY_TIMEOUT_S
from laboneq.controller.controller_core import ControllerCore

if TYPE_CHECKING:
    from collections.abc import Callable

    from laboneq.controller.api.commons import SubmissionHandle
    from laboneq.controller.callback_channel import CallbackChannel
    from laboneq.controller.controller_core import SubmissionStatus
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


_logger = logging.getLogger(__name__)


class AsyncLocalController(AsyncControllerAPI):
    """Asynchronous controller interface."""

    @staticmethod
    async def create(
        device_setup: DeviceSetup,
        ignore_version_mismatch: bool = False,
        neartime_callbacks: dict[str, Callable[..., Any]] | None = None,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
        instrument_topology: InstrumentTopology | None = None,
    ) -> AsyncLocalController:
        """Create an instance of the AsyncLocalController and connect it.

        Returns the instance even if connecting fails. `get_not_ready_reason`
        then reports the failure, and the controller tries again in the
        background when it needs the instruments, at most every
        `ControllerCore._RECONNECT_HOLDOFF_S` seconds. With `reset_devices`, the
        first successful connect resets the instruments, and no later one does.
        """
        if neartime_callbacks is None:
            neartime_callbacks = {}
        controller = ControllerCore(
            device_setup=device_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=neartime_callbacks,
        )
        local_controller = AsyncLocalController(
            device_setup=device_setup,
            controller=controller,
            instrument_topology=instrument_topology,
            do_emulation=do_emulation,
        )
        try:
            await local_controller.connect(
                reset_devices=reset_devices,
                disable_runtime_checks=disable_runtime_checks,
                timeout_s=DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s,
            )
        except Exception as e:
            _logger.warning("Cannot connect to the instruments: %s", e)
        return local_controller

    def __init__(
        self,
        device_setup: DeviceSetup,
        controller: ControllerCore,
        instrument_topology: InstrumentTopology | None = None,
        do_emulation: bool = True,
    ):
        self._device_setup = device_setup
        self._controller = controller
        self._instrument_topology = instrument_topology
        self._do_emulation = do_emulation
        self._submissions = SubmissionRegistry()

    async def connect(
        self,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ):
        """Connect to the instruments, or to the emulator under `do_emulation`.

        Raises if connecting fails. The interfaces do not declare this method:
        a remote implementation connects to the service in `create`, and the
        service connects to the instruments.
        """
        await self._controller.connect(
            do_emulation=self._do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )

    async def aclose(self):
        for handle in self._submissions:
            # A failed submission has already reached the caller through
            # `get_experiment`; teardown has to reach the disconnect regardless.
            with suppress(Exception):
                await self.cancel_experiment(handle)
        await self._controller.disconnect()

    async def get_default_devicesetup(self) -> DeviceSetup:
        return self._device_setup

    async def get_instrument_topology(self) -> InstrumentTopology | None:
        if self._instrument_topology is None:
            return None
        await self._controller.ensure_setup_description()
        # A client rebuilds a `DeviceSetup` from the topology, and compiles for it
        # with the setup description. Emulation has no instruments to fetch one from.
        setup_description = self._controller.setup_description
        if self._do_emulation or setup_description is None:
            return self._instrument_topology
        return attrs.evolve(
            self._instrument_topology, setup_description=setup_description
        )

    async def get_not_ready_reason(self) -> str | None:
        return await self._controller.get_not_ready_reason()

    async def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        await self._controller.wait_until_ready(timeout_s)

    @property
    def neartime_callbacks(self) -> dict[str, Callable]:
        return self._controller.neartime_callbacks

    async def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """
        self._controller.neartime_callbacks.update(neartime_callbacks)

    async def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        await self._controller.disable_outputs(
            device_uids, logical_signals, unused_only
        )

    async def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        callback_channel: CallbackChannel | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        handle = self._submissions.ensure_unique_handle(handle)
        self._submissions.add(
            handle,
            await self._controller.submit_compiled(
                scheduled_experiment=scheduled_experiment,
                callback_channel=callback_channel,
                coprocessor_payload_overrides=coprocessor_payload_overrides,
            ),
        )
        return handle

    async def get_callback_channel(
        self, handle: SubmissionHandle
    ) -> CallbackChannel | None:
        """The client callback channel for *handle*, if the experiment has one."""
        return self._submissions.get(handle).callback_channel

    async def wait_for_experiment(self, handle: SubmissionHandle):
        await self._controller.wait_submission(self._submissions.get(handle))

    async def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        return self._controller.submission_status(self._submissions.get(handle))

    async def get_experiment(self, handle: SubmissionHandle) -> Results:
        await self.wait_for_experiment(handle)
        return await self.get_experiment_results(handle)

    async def get_experiment_results(self, handle: SubmissionHandle) -> Results:
        return to_results(
            self._device_setup,
            self._controller.submission_results(self._submissions.get(handle)),
        )

    async def cancel_experiment(self, handle: SubmissionHandle):
        # TODO(2K): No mechanism to cancel yet. Implement proper cancellation.
        await self.wait_for_experiment(handle)

    async def close_submission(self, handle: SubmissionHandle):
        await self.cancel_experiment(handle)
        self._submissions.remove(handle)
