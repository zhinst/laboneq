# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import fields, dataclass
from typing import Iterable, Callable, TypeVar, Protocol, cast


@dataclass
class DataClassType(Protocol):
    pass


GREEN = "\033[92m"
YELLOW = "\33[33m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
END = "\033[0m"


def _filter_field_names(field_name: str) -> bool:
    return (
        field_name[0] == "_"
        or field_name == "sections"
        or field_name == "children"
        or field_name == "uid"
    )


def _name_altered_fields(subnode: DataClassType) -> list[str]:
    """
    Returns field names that are different from the default value in the respective dataclass
    """
    dummy = subnode.__class__()
    return [
        f.name
        for f in fields(subnode)
        if not _filter_field_names(f.name)
        and getattr(subnode, f.name) != getattr(dummy, f.name)
    ]


_T = TypeVar("_T")


def _get_attr(
    node: DataClassType | object, children_attr: str | Iterable[str], default: _T
) -> _T:
    attributes = children_attr or []
    if isinstance(children_attr, str):
        attributes = [children_attr]
    for attr in attributes:
        if hasattr(node, attr):
            return getattr(node, attr)
    return default


def _format_title_laboneq(node: object) -> str:
    uid = _get_attr(node, ["uid", "signal"], default="N.A")
    return GREEN + f"{node.__class__.__name__}{MAGENTA} {uid}" + END


def _format_body_laboneq(node: object) -> list[str]:
    naf = _name_altered_fields(cast(DataClassType, node))
    return [f"{YELLOW}{n}{END}: {CYAN}{repr(getattr(node, n))}{END}" for n in naf]


def beauty_list(
    node: object,
    title_formatter: Callable[[object], str],
    extra_info_formatter: Callable[[object], list[str]],
    children_attr: Iterable[str],
) -> list[str]:
    """
    Recursive tree traversal to assemble string representations for each tree node

    Arguments:
        node:
            Root node of an tree.
        title_formatter:
            String formatter for the title.
                Input for the callable is a node object and it should return a short
                string representation of the said node.
        extra_info_formatter: Callable to produce extra information about the node.
            Should returns a list of strings, which are displayed below the node title.
        children_attr: Possible attribute names which points to the children of the node.
            E.g. `node.children`, `node.sections`, ...

    Returns:
        list of strings representing the tree structure of the node instance.

    Example:
        ```python
        PrintNode 1  # title
        │   ⮡  name: 'root'  # extra information
        ├─PrintNode 2  # children
        │      ⮡  name: 'child1'
        └─PrintNode 3
                ⮡  name: 'child2'
        ````
    """
    subs = _get_attr(node, children_attr, default=[])
    lines = [title_formatter(node)]
    naf = extra_info_formatter(node)
    if len(naf) > 0:
        pre = " │   ⮡  " if len(subs) > 0 else "     ⮡  "
        lines.append(pre + ", ".join(naf))
    for sub in subs[:-1]:
        bl = beauty_list(sub, title_formatter, extra_info_formatter, children_attr)
        lines.extend([" ├─" + bl[0]] + [" │ " + _ for _ in bl[1:]])
    if len(subs) > 0:
        bl = beauty_list(subs[-1], title_formatter, extra_info_formatter, children_attr)
        lines.extend([" └─" + bl[0]] + ["   " + _ for _ in bl[1:]] + [""])
    return lines


def bprint(
    node: DataClassType,
) -> None:
    """
    Experimental printout of a short-format representation of LabOne Q dataclasses (like the DSL Experiment, Section, etc ...)

    Arguments:
        node:
            An object to visually inspect.
            Works in principle with any object, but will try to identify field names like
            "children", "sections", etc. used in objects of Experiment, Section,
            or derived LabOne Q DSL subclasses.
    """
    rows = beauty_list(
        node, _format_title_laboneq, _format_body_laboneq, ["sections", "children"]
    )
    print("\n".join(rows))
