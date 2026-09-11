# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq._backend_plugin import BackendPlugin
from laboneq._rust import BACKEND_NAME


class QCCSBackendPlugin(BackendPlugin):
    """Backend plugin for QCCS."""

    def register(self) -> None:
        """Register QCCS backend with LabOne Q."""
        from laboneq.qccs.dsl.device.instruments import register_instruments

        _register_compiler_hooks()
        register_instruments()
        _register_artifact_deserializer()
        _register_signal_plots_backend()
        _register_target_setup_builder()


def _register_compiler_hooks() -> None:
    """Register the QCCS backend's compiler hooks under its backend name."""
    from laboneq.compiler.workflow.compiler_hooks import register_compiler_hooks
    from laboneq.qccs.compiler.hooks import CompilerHooksSeqC

    register_compiler_hooks(CompilerHooksSeqC)


def _register_artifact_deserializer() -> None:
    """Register the QCCS backend's `compiled_experiment` module as the deserializer for the
    bytes that backend produces.
    """
    from laboneq._rust import compiled_experiment as qccs
    from laboneq.core.compiler_output import register_deserializer

    register_deserializer(BACKEND_NAME, qccs.deserialize)


def _register_signal_plots_backend() -> None:
    """Register the signal plotting backend for QCCS"""
    from laboneq.qccs.simulator.signal_plots_backend import QCCSSignalPlotsBackend
    from laboneq.simulator.signal_plots import backends

    backends.register(backend=QCCSSignalPlotsBackend(), name=BACKEND_NAME)


def _register_target_setup_builder() -> None:
    """Register how the controller builds the QCCS view of a device setup."""
    from laboneq.controller.target_setup import register_target_setup_builder
    from laboneq.qccs.controller.target_setup import build

    register_target_setup_builder(BACKEND_NAME, build)
