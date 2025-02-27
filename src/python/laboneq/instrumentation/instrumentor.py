# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Instrumentors for LabOne Q."""

from typing import Callable
import wrapt
from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from laboneq import __version__
import laboneq.compiler.scheduler.scheduler
import laboneq.compiler.seqc.code_generator
import laboneq.compiler
import laboneq.dsl.session
import laboneq.controller.controller
import laboneq.controller.devices.device_collection
import laboneq.compiler.seqc.analyze_playback
import laboneq.core.utilities.laboneq_compile
from laboneq.instrumentation import otel


class _BaseLabOneQInstrumentor(BaseInstrumentor):
    """Base class for LabOne Q instrumentors."""

    def __init__(self):
        super().__init__()
        self._to_instrument: list[object, str, Callable] = []

    def instrumentation_dependencies(self):
        return [f"laboneq == {__version__}"]

    def _instrument(self, **kwargs) -> None:
        tracer_provider = kwargs.get("tracer_provider") or trace.get_tracer_provider()
        tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version=__version__,
            tracer_provider=tracer_provider,
        )
        for path, obj, wrapper in self._to_instrument:
            wrapper = wrapper(tracer)
            wrapt.wrap_function_wrapper(path, obj, wrapper)

    def _uninstrument(self, **kwargs) -> None:
        for path, obj, *_ in self._to_instrument:
            unwrap(path, obj)


class SessionInstrumentor(_BaseLabOneQInstrumentor):
    """Opentelemetry instrumentor for LabOne Q session."""

    def __init__(self):
        super().__init__()
        self._to_instrument = [
            (laboneq.dsl.session.Session, "connect", self._wrap_session_connect),
            (laboneq.dsl.session.Session, "compile", self._wrap_session_compile),
            (laboneq.dsl.session.Session, "run", self._wrap_session_run),
        ]

    def _wrap_session_connect(self, tracer: trace.Tracer):
        def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span("laboneq.session.connect") as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return wrapped(*args, **kwargs)

        return wrapper

    def _wrap_session_compile(self, tracer: trace.Tracer):
        def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span("laboneq.session.compile") as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return wrapped(*args, **kwargs)

        return wrapper

    def _wrap_session_run(self, tracer: trace.Tracer):
        def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span("laboneq.session.run") as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return wrapped(*args, **kwargs)

        return wrapper


class CompilerInstrumentor(_BaseLabOneQInstrumentor):
    """Opentelemetry instrumentor for LabOne Q compiler."""

    def __init__(self):
        super().__init__()
        self._to_instrument = [
            (
                laboneq.compiler.scheduler.scheduler.Scheduler,
                "run",
                self._wrap_scheduler,
            ),
            (
                laboneq.compiler.seqc.code_generator.CodeGenerator,
                "generate_code",
                self._wrap_code_generator,
            ),
        ]

    def _wrap_scheduler(self, tracer: trace.Tracer):
        def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span("laboneq.compiler.schedule") as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return wrapped(*args, **kwargs)

        return wrapper

    def _wrap_code_generator(self, tracer: trace.Tracer):
        def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span("laboneq.compiler.generate-code") as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return wrapped(*args, **kwargs)

        return wrapper


class ControllerInstrumentor(_BaseLabOneQInstrumentor):
    """Opentelemetry instrumentor for LabOne Q controller."""

    def __init__(self):
        super().__init__()
        self._to_instrument = [
            (
                laboneq.controller.controller.Controller,
                "_connect_async",
                self._wrap_controller_connect,
            ),
            (
                laboneq.controller.controller.Controller,
                "_execute_compiled_async",
                self._wrap_controller_execute,
            ),
            (
                laboneq.controller.controller.NearTimeRunner,
                "run",
                self._wrap_near_time_execution,
            ),
            (
                laboneq.controller.controller.Controller,
                "_execute_one_step",
                self._wrap_controller_step_execution,
            ),
            (
                laboneq.controller.controller.Controller,
                "_read_one_step_results",
                self._wrap_read_step_results,
            ),
            (
                laboneq.controller.controller.Controller,
                "_prepare_nt_step",
                self._wrap_prepare_step,
            ),
        ]

    def _wrap_controller_connect(self, tracer: trace.Tracer):
        async def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span("laboneq.controller.connect") as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return await wrapped(*args, **kwargs)

        return wrapper

    def _wrap_near_time_execution(self, tracer: trace.Tracer):
        async def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span(
                "laboneq.controller.execute-near-time"
            ) as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return await wrapped(*args, **kwargs)

        return wrapper

    def _wrap_controller_step_execution(self, tracer: trace.Tracer):
        async def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span(
                "laboneq.controller.execute-near-time-step"
            ) as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return await wrapped(*args, **kwargs)

        return wrapper

    def _wrap_controller_execute(self, tracer: trace.Tracer):
        async def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span("laboneq.controller.execute") as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return await wrapped(*args, **kwargs)

        return wrapper

    def _wrap_read_step_results(self, tracer: trace.Tracer):
        async def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span(
                "laboneq.controller.read-results"
            ) as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return await wrapped(*args, **kwargs)

        return wrapper

    def _wrap_prepare_step(self, tracer: trace.Tracer):
        async def wrapper(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span(
                "laboneq.controller.prepare-step"
            ) as span:
                span.set_attributes(otel.source_code_attributes(wrapped))
                return await wrapped(*args, **kwargs)

        return wrapper


class LabOneQInstrumentor(BaseInstrumentor):
    """Opentelemetry instrumentor for LabOne Q."""

    _instrumentors = [SessionInstrumentor, CompilerInstrumentor, ControllerInstrumentor]

    def instrumentation_dependencies(self):
        return [f"laboneq == {__version__}"]

    def _instrument(self, **kwargs) -> None:
        for instrumentor in self._instrumentors:
            instrumentor().instrument(**kwargs)

    def _uninstrument(self, **kwargs) -> None:
        for instrumentor in self._instrumentors:
            instrumentor().uninstrument(**kwargs)
