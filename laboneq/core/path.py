# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

Separator = "/"
InstrumentOutputs = "$OUTPUTS"
InstrumentInputs = "$INPUTS"
Results = "$RESULTS"


Instruments_Path = "instruments"
Instruments_Path_Abs = Separator + Instruments_Path
LogicalSignalGroups_Path = "logical_signal_groups"
LogicalSignalGroups_Path_Abs = Separator + LogicalSignalGroups_Path
PhysicalChannelGroups_Path = "physical_channel_groups"
PhysicalChannelGroups_Path_Abs = Separator + PhysicalChannelGroups_Path


def concat(*args):
    return Separator.join(args)


def is_abs(path: str):
    """Test if a path string is absolute i.e. starts with a leading separator.

    Args:
        path: The path string to test.

    Returns:
        True, if the string starts with a leading separator character. Else
        returns false.
    """
    return path.startswith(Separator)


def split(path: str):
    """Split a path string into a list of path elements.

    A path element is a portion of the path string that is separated by ``'/'``.

    Arguments:
        path: The path string to split.

    Returns:
        A list of path elements.
    """
    if is_abs(Separator):
        path = path[1:]
    return path.split(Separator)


def starts_with(path: str, prefix: str, ignore_abs_path: bool = False):
    """Test if a path string starts with a given prefix.

    Args:
        path: The path to test for prefix.
        prefix: The prefix string to look for in path.
        ignore_abs_path: If true, any leading path separator character will be
            ignored in both arguments, path and prefix. If false, a character by
            character comparison is done equivalent to `str.startswith()`.

    Returns:
        True, if the given path starts with the given prefix text. False, if it
        does not.
    """
    if ignore_abs_path and is_abs(path):
        path = path[1:]
    if ignore_abs_path and is_abs(prefix):
        prefix = prefix[1:]

    return path.startswith(prefix)
