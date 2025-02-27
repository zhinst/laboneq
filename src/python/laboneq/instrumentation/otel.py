# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import inspect


def source_code_attributes(obj: object) -> dict[str, str]:
    """Create span source code attributes.

    OpenTelemetry semantic conventions v1.30.0
    """
    namespace = ".".join([obj.__module__, obj.__qualname__.split(".")[0]])
    return {
        "code.file.path": inspect.getfile(obj),
        "code.function.name": obj.__name__,
        "code.namespace": namespace,
        "code.line.number": inspect.getsourcelines(obj)[-1],
    }
