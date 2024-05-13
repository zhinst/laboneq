# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from dataclasses import fields, dataclass

GREEN = "\033[92m"
YELLOW = "\33[33m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
END = "\033[0m"


def filter_field_names(field_name: str) -> bool:
    return (
        field_name[0] == "_"
        or field_name == "sections"
        or field_name == "children"
        or field_name == "uid"
    )


def name_altered_fields(subnode: dataclass):
    """
    Returns field names that are different from the default value in the respective dataclass
    """
    dummy = subnode.__class__()
    return [
        f.name
        for f in fields(subnode)
        if not filter_field_names(f.name)
        and getattr(subnode, f.name) != getattr(dummy, f.name)
    ]


def beauty_list(node: dataclass) -> list[str]:
    """
    Recursive tree traversal to assemble string representations for each tree node

    Arguments:
        node:
            instance of any dataclass but will try to identify field names like
            "children", "sections", etc. used in objects of Experiment, Section,
            or derived subclasses of DSL.

    Returns:
        lines:
            list of strings representing the tree structure of the dataclass instance

    """
    uid = getattr(node, "uid", getattr(node, "signal", "N.A."))
    naf = name_altered_fields(node)
    subs = getattr(node, "children", getattr(node, "sections", []))
    lines = [GREEN + f"{node.__class__.__name__}{MAGENTA} {uid}" + END]
    if len(naf) > 0:
        pre = " │   ⮡  " if len(subs) > 0 else "     ⮡  "
        saf = [f"{YELLOW}{n}{END}: {CYAN}{repr(getattr(node, n))}{END}" for n in naf]
        lines.append(pre + ", ".join(saf))
    for sub in subs[:-1]:
        bl = beauty_list(sub)
        lines.extend([" ├─" + bl[0]] + [" │ " + _ for _ in bl[1:]])
    if len(subs) > 0:
        bl = beauty_list(subs[-1])
        lines.extend([" └─" + bl[0]] + ["   " + _ for _ in bl[1:]] + [""])
    return lines


def bprint(node: dataclass):
    """
    Experimental printout of a short-format representation of LabOne Q dataclasses (like the DSL Experiment, Section, etc ...)

    Arguments:
        node:
            A dataclass to visually inspect.
            Works in principle with any dataclass, but will try to identify field names like
            "children", "sections", etc. used in objects of Experiment, Section,
            or derived LabOne Q DSL subclasses.

    Returns:
        None
        Prints a representation of the `node` dataclass.

    """
    print("\n".join(beauty_list(node)))
