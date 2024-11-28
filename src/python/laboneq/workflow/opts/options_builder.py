# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""OptionBuilder class to build options for a workflow."""

from __future__ import annotations

import typing
from collections import UserList
from io import StringIO

from rich.console import Console
from rich.pretty import pprint

if typing.TYPE_CHECKING:
    from laboneq.workflow.opts.options_core import WorkflowOptions
    from laboneq.workflow.opts.options_base import BaseOptions


class OptionBuilder:
    """A class to build options for a workflow."""

    def __init__(self, base: WorkflowOptions) -> None:
        self._base = base
        self._flatten_opts = sorted(_get_all_fields(self._base))

    @property
    def base(self) -> WorkflowOptions:
        """Return the base options."""
        return self._base

    def __dir__(self):
        return self._flatten_opts

    def __getattr__(self, field_name: str) -> OptionNodeList:
        if field_name.startswith("_"):
            return super().__getattribute__(field_name)
        if field_name in self._flatten_opts:
            return _retrieve_option_attributes(self._base, field_name)
        return super().__getattribute__(field_name)

    def __setattr__(self, field_name: str, value: typing.Any) -> None:  # noqa: ANN401
        if field_name.startswith("_"):
            return super().__setattr__(field_name, value)
        if field_name in self._flatten_opts:
            raise TypeError(
                "Setting options by assignment is not allowed. "
                "Please use the method call."
            )
        return super().__setattr__(field_name, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptionBuilder):
            return NotImplemented
        return self._base == other._base

    def __str__(self):
        with StringIO() as buffer:
            console = Console(file=buffer)
            pprint(self._base, console=console, expand_all=True, indent_guides=True)
            return buffer.getvalue()

    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        p.text(str(self))

    def _overview(self) -> str:
        """Return the overview of all option fields."""
        header_color = "\033[95m"  # Magenta
        separator_color = "\033[94m"  # Blue
        field_color = "\033[92m"  # Green
        description_color = "\033[93m"  # Yellow
        value_color = "\033[96m"  # Cyan
        reset_color = "\033[0m"  # Reset to default color

        header = f"{header_color}Option Fields{reset_color}"
        separator = f"{separator_color}{'=' * len('Option Fields')}{reset_color}"
        rows = []
        for field in self._flatten_opts:
            rows.append(f"{field_color}{field}:{reset_color}")
            desc = self._get_field_descr(field)
            for d in desc:
                rows.append(
                    f"{description_color}\tDescription:{reset_color} "
                    f"{value_color}{d['description']}{reset_color}"
                )
                class_default = list(zip(d["options"], d["default"]))
                rows.append(
                    f"{description_color}\tClasses and Defaults:{reset_color} "
                    f"{value_color}{class_default}{reset_color}, "
                )
                rows.append("")
        fields_str = "\n".join(rows)
        return f"{header}\n{separator}\n{fields_str}"

    def _get_field_descr(self, field: str) -> str:
        """Combine fields with the same description."""
        # TODO: Optimize this method
        fields: OptionNodeList = getattr(self, field)
        d = {type(f.option).__name__: (f.description, f.default_value) for f in fields}
        d2 = {}
        # iterate over d and combine fields if they have the same description
        for k, v in d.items():
            if v[0] in d2:
                d2[v[0]].append((k, v[1]))
            else:
                d2[v[0]] = [(k, v[1])]
        return [
            {
                "description": k,
                "options": [i[0] for i in v],
                "default": [i[1] for i in v],
            }
            for k, v in d2.items()
        ]


def show_fields(opt: OptionBuilder) -> None:
    """Print the overview of the option fields."""
    print(opt._overview())  # noqa: T201


def _retrieve_option_attributes(
    option: WorkflowOptions,
    field_name: str,
    current_node: str = "base",
) -> OptionNodeList:
    """Return OptionNodeList that contains the option fields."""
    from laboneq.workflow.opts.options_core import WorkflowOptions

    option_list = OptionNodeList()
    opt_field = option.fields.get(field_name, None)
    if opt_field is not None:
        option_list.append(OptionNode(current_node, field_name, option))
    for task_name, opt in option._task_options.items():
        new_node = current_node + "." + task_name
        if isinstance(opt, WorkflowOptions):
            temp = _retrieve_option_attributes(opt, field_name, new_node)
            option_list.extend(temp)
        elif hasattr(opt, field_name):
            option_list.append(OptionNode(new_node, field_name, opt))
    return option_list


def _get_all_fields(option: WorkflowOptions) -> set[str]:
    """Return all fields in the task_options and the top level fields."""
    from laboneq.workflow.opts.options_core import WorkflowOptions

    all_fields = set(option.fields.keys())
    for opt in option._task_options.values():
        if isinstance(opt, WorkflowOptions):
            all_fields.update(_get_all_fields(opt))
        else:
            for field_name in opt.fields:
                all_fields.add(field_name)
    top_level_options = option.fields.keys()
    all_fields.update(top_level_options)
    return all_fields


class OptionNodeList(UserList):
    """A list of option nodes for setting and querying options values."""

    def __init__(self, elements: list[OptionNode] | OptionNodeList | None = None):
        super().__init__(elements or [])

    def __getitem__(self, item: typing.Any) -> OptionNode | OptionNodeList:  # noqa: ANN401
        if isinstance(item, slice):
            return type(self)(self.data[item])
        return self.data[item]

    def __call__(self, value: typing.Any, selector: str | None = None) -> None:  # noqa: ANN401
        """Set the value of option fields, selected by a path name of a task/workflow.

        Arguments:
            value: The value to set.
            selector: Path of the task or workflow.
                If None, set the value for all option fields in the list.

        Example:
            ```python
            opt = workflow.options()

            # To set the value of field `count` at the top-level
            opt.count(1, ".")

            # To set a value to field `count` of a sub-workflow at top-level
            opt.count(1, "sub_workflow")

            # a task at top-level
            opt.count(1, "task1)

            # a task nested in `sub_workflow`
            opt.count(1, "sub_workflow.task2")

            # Or a workflow nested in another workflow
            opt.count(1, "sub_workflow.nested_workflow")
            ```
        """
        filtered = self if selector is None else self._get_filtered(selector)
        for element in filtered:
            element(value)

    def _get_filtered(self, task_name: str) -> OptionNodeList:
        # OptionNode is represented using "full-path" format, aka base.wf1.task1
        # But we'd like to omit "base" when setting the fields; "wf1.task1"
        # would be sufficient.
        filter_name = "base." + task_name if task_name != "." else task_name
        filtered = OptionNodeList(
            [node for node in self if self._predicate(node, filter_name)]
        )
        if not filtered:
            raise ValueError(
                f"Task or workflow {task_name} not found to have the option "
                f"{self[0].field}."
            )
        return filtered

    def _predicate(self, item: OptionNode, name: str) -> bool:
        if name == ".":
            return item.is_top_level()
        return name == item.name or name == item.name.rsplit(".", 1)[0]

    def __str__(self):
        max_name_length = max([len(node.name) for node in self], default=0)
        max_value_length = max([len(str(node._value)) for node in self], default=0)
        name_field_width = max(max_name_length, 50)
        value_field_width = max(max_value_length, 10)
        formatted_nodes = []
        for node in self:
            # strip base. prefix:
            _, _, post_dot = node.name.partition(".")
            node_name = post_dot if post_dot != "" else "."
            formatted_nodes.append(
                f"{node_name.ljust(name_field_width)} | "
                f"{str(node._value).ljust(value_field_width)}"
            )
        header = (
            f"{'Task/Workflow'.ljust(name_field_width)}"
            f" | {'Value'.ljust(value_field_width)}"
        )
        separator = "-" * len(header)
        return "\n".join(
            [
                separator,
                header,
                separator,
                *formatted_nodes,
                separator,
            ]
        )

    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        p.text(str(self))


class OptionNode:
    """A class representing a node for the options in workflow.

    Arguments:
        task_name: Full path of the option field.
            Examples: nested_wf.task1
            or task1 if task1 is at the base of the option.
            "base" if it is a top-layer option field.
        field: Name of the option field.
        option: The option instance.
    """

    def __init__(self, task_name: str, field: str, option: BaseOptions) -> None:
        self.name = task_name
        self.field = field
        self.option = option
        self._value = str(getattr(self.option, self.field, self.option))
        self._description = option.fields[self.field].metadata.get("description", None)
        self._default_value = option.fields[self.field].default

    @property
    def description(self) -> str:
        """Return the description of the option field."""
        return self._description

    @property
    def default_value(self):  # noqa: ANN201
        """Return the default value of the option field."""
        return self._default_value

    def is_top_level(self) -> bool:
        """Return True if the option is a top-level field."""
        splitted = self.name.split(".")
        return splitted[0] == "base" and len(splitted) <= 2  # noqa: PLR2004

    def __call__(self, value: typing.Any) -> None:  # noqa: ANN401
        """Set the value of the option."""
        setattr(self.option, self.field, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptionNode):
            return NotImplemented
        return (
            self.name == other.name
            and self.field == other.field
            and self.option == other.option
            and self._value == other._value
        )

    def __str__(self) -> str:
        return f"{self.name} : {self._value}"

    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        p.text(str(self))
