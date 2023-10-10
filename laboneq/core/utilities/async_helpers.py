# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import asyncio

_event_loop_patched = False


# Required to keep sync interface callable from Jupyter Notebooks
# See https://blog.jupyter.org/ipython-7-0-async-repl-a35ce050f7f7
def _enable_event_loop_nesting_if_necessary():
    global _event_loop_patched
    if _event_loop_patched:
        return
    try:
        asyncio.get_running_loop()
        # Running event loop detected, nesting is necessary
        import nest_asyncio

        nest_asyncio.apply()
    except RuntimeError:
        pass  # No event loop is running
    _event_loop_patched = True


def run_async(coro):
    _enable_event_loop_nesting_if_necessary()
    return asyncio.run(coro)
