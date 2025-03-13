# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry instrumentation for tracing LabOne Q.

LabOne Q will NEVER transmit telemetry data to Zurich Instruments (or any other party) on its
own. This Python package purely serves to enable optional instrumentation of LabOne Q internals,
so that user-supplied OpenTelemetry tracers can 'see inside' the LabOne Q. Users must
still provide their own tracers, and their own back-end for storing or visualizing the acquired
tracing data.

Using the feature requires the following dependencies, which
are not provided in the base installation:

- `opentelemetry-instrumentation`
- `opentelemetry-api`
- `wrapt`

# Usage

`laboneq.instrumentation` provides an instrumentor which follows OpenTelemetry `BaseInstrumentor` API.

The configuration of the tracer is left up to the user, `laboneq` will attach the recorded spans to the active span,
if one exists when `laboneq` code is executed.

```python
from laboneq.instrumentation import LabOneQInstrumentor

# <Tracer configuration>

LabOneQInstrumentor().instrument(tracer_provider=tracer_provider)

# <Some code that executes laboneq>

LabOneQInstrumentor().uninstrument()
```
"""

from .instrumentor import LabOneQInstrumentor

__all__ = "LabOneQInstrumentor"
