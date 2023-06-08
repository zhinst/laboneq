# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

_logger = logging.getLogger(__name__)


def convert_dynamic(
    source_object,
    source_type_hint=None,
    source_type_string=None,
    target_type_hint=None,
    target_type_string=None,
    orig_is_collection=False,
    conversion_function_lookup=None,
):
    if source_object is None:
        return None
    if type(source_object) in [int, float, str, bool]:
        return source_object

    if source_type_string == "Dict" and target_type_string == "Dict":
        _logger.info(f"Converting Dict with {len(source_object)} elements")
        retval = {}
        for k, v in source_object.items():
            conversion_function = conversion_function_lookup(type(v))

            if conversion_function is not None:
                retval[k] = conversion_function(v)
            else:
                retval[k] = v
        return retval

    conversion_function = conversion_function_lookup(type(source_object))
    if conversion_function is not None:
        # _logger.info(f"Found conversion function for type {type(source_object)}")
        return conversion_function(source_object)

    if source_type_string == "List":
        _logger.info(f"Converting List with {len(source_object)} elements")
        retval = []
        for s in source_object:
            conversion_function = conversion_function_lookup(type(s))
            retval.append(conversion_function(s))
        return retval

    if orig_is_collection:
        retval = []
        if source_object is not None:
            _logger.info(
                f"Converting collection with {len(source_object)} items  for type {source_type_hint} to type {target_type_hint}"
            )
            if isinstance(source_object, dict):
                source_collection = source_object.values()
            else:
                source_collection = source_object

            for s in source_collection:
                conversion_function = conversion_function_lookup(type(s))
                if conversion_function is None:
                    # _logger.info(
                    #    f"Conversion function not found for type {type(s)}, looking up by type hint {source_type_hint}"
                    # )
                    conversion_function = conversion_function_lookup(source_type_hint)
                if conversion_function is None:
                    raise Exception(
                        f"Conversion function not found for type hint {source_type_hint}"
                    )

                retval.append(conversion_function(s))
        _logger.info(f"List converted with {len(retval)} elements")
        return retval
    _logger.info(
        f"NOT doing anythiing , return source object of type {type(source_object)}, source_type_hint: {source_type_hint}, source_type_string: {source_type_string}, target_type_hint: {target_type_hint}, target_tpye_string: {target_type_string}, orig_is_collection: {orig_is_collection}, conversion_function_lookup: {conversion_function_lookup}"
    )
    return None
