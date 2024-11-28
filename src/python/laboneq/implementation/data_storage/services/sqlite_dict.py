# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import copy
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Union
from uuid import uuid4

from laboneq.dsl.serialization.serializer import Serializer
from laboneq.interfaces.data_storage.data_storage_api import DataStorageAPI


class DataStorageServiceSqliteDict(DataStorageAPI):
    METADATA_TABLE = "metadata"
    DATA_TABLE = "data"

    def __init__(self, file_path: str | None = None):
        from sqlitedict import SqliteDict

        if file_path is None:
            file_path = "laboneq_data/data.db"
        self._file_path = Path(file_path)

        # Check if the directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        self._metadata_db = SqliteDict(
            str(self._file_path), tablename=self.METADATA_TABLE, autocommit=True
        )
        self._data_db = SqliteDict(
            str(self._file_path), tablename=self.DATA_TABLE, autocommit=True
        )

    def get(
        self, key: str, with_metadata: bool = False
    ) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        metadata = self.get_metadata(key)
        data_type = metadata["__type"]
        raw_data = self._data_db[key]
        deserialized_object = Serializer.load(raw_data, data_type)
        if with_metadata:
            return deserialized_object, metadata
        else:
            return deserialized_object

    def get_metadata(self, key: str) -> Dict[str, Any]:
        return self._convert_metadata(self._metadata_db[key])

    def keys(self) -> Iterable[str]:
        """Return an iterable of all keys in the database."""
        return self._metadata_db.keys()

    def store(
        self,
        data: Any,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store data in the database. Only data that can be serialized with the LabOne Q serializer can be stored.

        Args:
            key (str): The key to store the data under.
            data (any): The data to store.
            metadata (dict): Metadata to store with the data. Metadata can be used to search for data in the database.
            Metadata must have strings as keys, and values may be strings or python standard datetime objects.
        """
        if key is None:
            key = uuid4().hex
        self._validate_key(key)

        metadata = copy.deepcopy(metadata)
        if metadata is None:
            metadata = {}
        metadata["__type"] = type(data).__name__
        self._validate_metadata(metadata)
        self._metadata_db[key] = metadata
        serialized_data = Serializer.to_dict(data)
        self._data_db[key] = serialized_data
        return key

    def delete(self, key: str) -> bool:
        """
        Delete data from the database.

        Args:
            key (str): The key of the data to delete.
        """
        deleted = False
        try:
            del self._metadata_db[key]
            deleted = True
        except KeyError:
            pass
        try:
            del self._data_db[key]
            deleted = True
        except KeyError:
            pass

        return deleted

    def find(
        self,
        metadata: dict[str, Any] | None = None,
        condition: Callable[[dict[str, Any]], bool] | None = None,
    ) -> Iterable[str]:
        """
        Find data in the database.

        Args:
            metadata (dict): Metadata to search for. If not None, only data where all keys and values match the
                metadata will be returned.
                If None, returns all data which also matches the condition.

            condition (function): A function that takes a single argument (the metadata of a data entry) and returns True if the data entry should be returned. If None, ¨
                all data matching the metadata will be returned.
        """
        for key in self.keys():
            metadata_for_key = self.get_metadata(key)
            if metadata is not None:
                if not self._metadata_matches(metadata, metadata_for_key):
                    continue
            if condition is not None:
                if not condition(metadata_for_key):
                    continue
            yield key

    def _validate_key(self, key: str) -> None:
        if not isinstance(key, str):
            raise ValueError("Key must be a string.")

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        for metadata_key, metadata_value in metadata.items():
            if not isinstance(metadata_key, str):
                raise ValueError("Metadata keys must be strings.")
            if not isinstance(
                metadata_value, (str, int, float, bool, bytes, datetime.datetime)
            ):
                raise ValueError(
                    "Metadata values must be strings, ints, floats, bools, bytes or datetime objects."
                )
            if isinstance(metadata_value, datetime.datetime):
                metadata[metadata_key] = {"datetime": metadata_value.isoformat()}

    def _metadata_matches(
        self, metadata: Dict[str, Any], metadata_to_match: Dict[str, Any]
    ) -> bool:
        return all(
            metadata_key in metadata_to_match
            and metadata_value == metadata_to_match[metadata_key]
            for metadata_key, metadata_value in metadata.items()
        )

    @staticmethod
    def _convert_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        converted_metadata = {}
        for metadata_key, metadata_value in metadata.items():
            if isinstance(metadata_value, dict) and "datetime" in metadata_value:
                converted_metadata[metadata_key] = datetime.datetime.fromisoformat(
                    metadata_value["datetime"]
                )
            else:
                converted_metadata[metadata_key] = metadata_value
        return converted_metadata

    def close(self):
        self._metadata_db.close()
        self._data_db.close()
