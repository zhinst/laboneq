# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union


class DataStorageAPI(ABC):
    """
    The interface for a data storage service. This service is used to store and retrieve experiment, setup and result data from a database.
    """

    def get(
        self, key: str, with_metadata=False
    ) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        pass

    def get_metadata(self, key: str) -> Dict[str, Any]:
        pass

    def keys(self) -> Iterable[str]:
        """Return an iterable of all keys in the database."""
        raise NotImplementedError

    def store(
        self, data: Any, key: str = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store data in the database. Only data that can be serialized with the LabOne Q serializer can be stored.

        Args:
            key (str): The key to store the data under.
            data (any): The data to store.
            metadata (dict): Metadata to store with the data. Metadata can be used to search for data in the database.
            Metadata must have strings as keys, and values may be strings or python standard datetime objects.
        """
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """
        Delete data from the database.

        Args:
            key (str): The key of the data to delete.
        """
        raise NotImplementedError

    def find(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Iterable[str]:
        """
        Find data in the database.

        Args:
            metadata (dict): Metadata to search for. If not None, only data where all keys and values match the
                metadata will be returned.
                If None, all data which also matches the condition will be returned.

            condition (function): A function that takes a single argument (the metadata
                of a data entry) and returns True if the data entry should be returned.
                If None, all data matching the metadata will be returned.
        """
        raise NotImplementedError
