# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
from typing import Callable

import numpy as np
import orjson
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from laboneq import __version__ as laboneq_version
from laboneq import simple
from laboneq_benchmark.instrumentation import LabOneQInstrumentor


# extract span duration from benchmark results, ready for plotting
def extract_duration(benchmark_results, key):
    return np.array([bm.get_operation(key).duration for bm in benchmark_results])


# extract metadata from benchmark results, ready for plotting
def extract_metadata(benchmark_results, key):
    return np.array([bm.metadata[key] for bm in benchmark_results])


def _otel_dt_to_epoch(time: str) -> int:
    """Convert opentelemetry timestamp to epoch."""
    return datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()


class Operation:
    def __init__(
        self,
        name: str,
        start_time: float,
        duration: float,
        metadata: dict | None = None,
    ):
        self.name = name
        self.start_time = start_time
        self.duration = duration
        self.metadata = metadata or {}
        self._children: list[Operation] = []

    def to_dict(self):
        return {
            "name": self.name,
            "start_time": self.start_time,
            "duration": self.duration,
            "metadata": self.metadata,
            "operations": {child.name: child.to_dict() for child in self._children},
        }


def _otel_traces_to_tree(traces: list[dict]) -> tuple[Operation, dict]:
    """Convert OpenTelemetry data to nested operations."""
    root: Operation | None = None
    ops: dict[str, Operation] = {}
    runner_info = {}
    for span in reversed(traces):
        duration = _otel_dt_to_epoch(span["end_time"]) - _otel_dt_to_epoch(
            span["start_time"]
        )
        span_id = span["context"]["span_id"]
        parent_id = span["parent_id"]
        if parent_id is None:
            if root is not None:
                # Currently we only support one top level span.
                raise ValueError("Multiple root benchmark traces are not supported")
            metadata = {}
            for attr_name, attr_value in span["attributes"].items():
                metadata[_deformat_user_attribute(attr_name)] = attr_value
            runner_info = {
                "name": span["resource"]["attributes"]["service.name"],
                "laboneq_version": span["resource"]["attributes"]["laboneq.version"],
                "version": "0.0.1",
            }
            root = Operation(
                name=span["name"],
                start_time=span["start_time"],
                duration=duration,
                metadata=metadata,
            )
            ops[span_id] = root
        else:
            parent = ops.get(parent_id)
            operation = Operation(
                name=span["name"],
                start_time=span["start_time"],
                duration=duration,
                metadata={},
            )
            ops[span_id] = operation
            if parent is None:
                root._children.append(ops[span_id])
            else:
                parent._children.append(operation)
    return root, runner_info


class BenchmarkResult:
    """Benchmark result."""

    def __init__(self, traces: list[dict]):
        # Keep the original OTEL traces
        self._raw_traces = traces
        self._operation, self._resource = _otel_traces_to_tree(traces)
        self._operation_index: dict[str, list[Operation]] = (
            BenchmarkResult._index_operations(self._operation)
        )

    @staticmethod
    def _index_operations(operation: Operation) -> dict[str, list[Operation]]:
        result = {operation.name: [operation]}
        for child in operation._children:
            out = BenchmarkResult._index_operations(child)
            for k, v in out.items():
                if k in result:
                    result[k].extend(v)
                else:
                    result[k] = v
        return result

    def operation_names(self) -> set[str]:
        return set(self._operation_index.keys())

    def get_operation(self, name: str) -> Operation:
        ops = self._operation_index[name]
        op = Operation(
            name=ops[0].name,
            start_time=ops[0].start_time,
            duration=sum([x.duration for x in ops]),
            metadata=ops[0].metadata,
        )
        return op

    @property
    def metadata(self) -> dict:
        return {**self._operation.metadata, "resource": self._resource}

    def to_dict(self) -> list[dict]:
        """Return a dictionary representation of the benchmark result.

        Returns:
            A list of each individual benchmarking run.
        """
        out = {**self._operation.to_dict(), "resource": self._resource}
        return out


def _setup_otlp_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    span_processor = BatchSpanProcessor(exporter)
    resource = Resource(
        attributes={
            SERVICE_NAME: "laboneq-benchmark",
            "laboneq.version": laboneq_version,
        }
    )
    traceProvider = TracerProvider(resource=resource)
    traceProvider.add_span_processor(span_processor)
    return traceProvider, exporter


def _deformat_user_attribute(key: str) -> str:
    return key.replace("user-attribute.", "")


def _format_user_attributes(metadata: dict | None) -> dict | None:
    if metadata is None:
        return None
    return {f"user-attribute.{k}": v for k, v in metadata.items()}


_USE_LEGACY_SERIALIZER = True if laboneq_version < "2.54.0" else False
if _USE_LEGACY_SERIALIZER:
    from laboneq.dsl.serialization import Serializer


def _serialize(compiled_experiment: simple.CompiledExperiment) -> str:
    if _USE_LEGACY_SERIALIZER:
        return Serializer.to_json(compiled_experiment)
    return simple.to_json(compiled_experiment)


def _deserialize(compiled_experiment: simple.CompiledExperiment) -> None:
    if _USE_LEGACY_SERIALIZER:
        Serializer.from_json(compiled_experiment)
    simple.from_json(compiled_experiment)


def benchmark(
    device_setup: simple.DeviceSetup,
    experiment: simple.Experiment | Callable[[], simple.Experiment],
    metadata: dict | None = None,
) -> BenchmarkResult:
    """Benchmark an LabOne Q experiment.

    The benchmark includes the following operations:

        * Experiment creation time (if applicable)
        * Experiment compilation
        * Compiled experiment serialization to JSON
        * Compiled experiment deserialization from JSON

    The benchmark function will suppress errors raised during serialization due to the issues
    in backwards compatibility. This allows the benchmark to proceed when using older LabOne Q versions.

    Arguments:
        device_setup: The device setup to use for the experiment.
        experiment: The experiment to benchmark.
            Either `Experiment` or a callable that returns an `Experiment`.
            If `experiment` is a callable, the experiment creation time is included in the benchmarks.
        metadata: Optional metadata to include in the benchmark.

    Returns:
        BenchmarkResult: The result of the benchmark.
    """
    tracer_provider, exporter = _setup_otlp_exporter()
    LabOneQInstrumentor().instrument(tracer_provider=tracer_provider)
    from laboneq.core.utilities.laboneq_compile import laboneq_compile

    tracer = trace.get_tracer(
        instrumenting_module_name=__name__,
        instrumenting_library_version=laboneq_version,
        tracer_provider=tracer_provider,
    )
    with tracer.start_as_current_span(
        "experiment-benchmark",
        attributes=_format_user_attributes(metadata),
    ) as root_span:
        with tracer.start_as_current_span("experiment-creation"):
            if callable(experiment):
                experiment = experiment()
        root_span.set_attribute("experiment_name", experiment.uid or "unnamed")
        with tracer.start_as_current_span("experiment-compile"):
            compiled_experiment = laboneq_compile(device_setup, experiment)
        with tracer.start_as_current_span("experiment-serialize"):
            compiled_experiment_json = None
            with tracer.start_as_current_span("serialize"):
                # Serialization may or may not work due to backwards compatibility issue so we try to catch any exceptions
                try:
                    compiled_experiment_json = _serialize(compiled_experiment)
                except Exception:
                    print("Serialization failed")
            with tracer.start_as_current_span("deserialize"):
                try:
                    _deserialize(compiled_experiment_json)
                except Exception:
                    print("Deserialization failed")

    tracer_provider.force_flush()
    spans = [orjson.loads(span.to_json()) for span in exporter.get_finished_spans()]
    LabOneQInstrumentor().uninstrument()
    return BenchmarkResult(spans)
