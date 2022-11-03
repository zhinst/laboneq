# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


def add_to_dict_list(dictlist, key, item):
    if not key in dictlist:
        dictlist[key] = []
    dictlist[key].append(item)


def merge_dict_list(dl1, dl2):
    for k, v in dl2.items():
        if k in dl1:
            dl1[k].extend(v)
        else:
            dl1[k] = v
