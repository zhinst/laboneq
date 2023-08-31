# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


def convert_dynamic(source_object, conversion_function_lookup: dict):
    if source_object is None:
        return None
    if type(source_object) in [int, float, str, bool]:
        return source_object

    if isinstance(source_object, dict):
        return {
            k: convert_dynamic(v, conversion_function_lookup)
            for k, v in source_object.items()
        }

    if isinstance(source_object, list):
        return [convert_dynamic(v, conversion_function_lookup) for v in source_object]

    conversion_function = conversion_function_lookup.get(type(source_object))
    if conversion_function is not None:
        return conversion_function(source_object)

    return source_object
