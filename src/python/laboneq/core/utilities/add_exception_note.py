# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Support for adding notes to exceptions.

PEP 678: Enriching Exceptions with Notes, added a new
feature that allows adding notes to exceptions. These
notes are presented to the user when the exception is
displayed in a traceback.

The `.add_note` method is available from Python 3.11.

This module adds a small utility that adds a note if
the version of Python supports it.

See https://peps.python.org/pep-0678/.

This module can be removed once support for Python 3.10
is dropped.
"""


def do_add_note(err: Exception, note: str):
    """Add the given note to the exception."""
    err.add_note(note)


def dont_add_note(err: Exception, note: str):
    """Don't add the given note to the exception."""
    pass


def do_assert_notes(err: Exception, notes):
    """Assert that the exception has the specified notes."""
    assert err.__notes__ == notes


def dont_assert_notes(err: Exception, notes):
    """Don't assert that the exception has the specified notes."""
    pass


if hasattr(Exception, "add_note"):
    add_note = do_add_note
    assert_notes = do_assert_notes
else:
    # Python 3.10 and before don't support add_note
    add_note = dont_add_note
    assert_notes = dont_assert_notes
