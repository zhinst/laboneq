# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from laboneq.data.experiment_results import AcquiredResults, slice_acquired_result

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, ValuesView

    from laboneq.data.experiment_results import AcquiredResult


@dataclass(frozen=True)
class AcquiredResultStructure:
    """Layout and progress of one acquired result, without its data."""

    axis_name: list[str | list[str]]
    shape: tuple[int, ...]
    last_nt_step: list[int] | None


class ResultStructureFetcher(Protocol):
    """Fetches the layout of every declared handle, without any measurements."""

    def __call__(self) -> dict[str, AcquiredResultStructure]: ...


class AcquiredResultsFetcher(Protocol):
    """Fetches acquired results from the controller service.

    ``handles=None`` means all handles. An unknown handle raises ``KeyError``;
    with ``nt="latest"`` handles without a completed near-time step are omitted
    from the returned mapping.
    """

    def __call__(
        self,
        handles: tuple[str, ...] | None,
        nt: Literal["all", "latest"],
    ) -> dict[str, AcquiredResult]: ...


class LazyAcquiredResults(AcquiredResults):
    """Acquired results fetched on demand from the controller service.

    Reading one handle fetches and caches just that handle. Listing the
    handles, counting them or checking whether one is there asks the service
    for their layout and downloads no results at all. Reading every value
    downloads all of them, in a single request.
    """

    def __init__(
        self,
        fetch: AcquiredResultsFetcher,
        fetch_structure: ResultStructureFetcher,
    ):
        super().__init__()
        self._fetch = fetch
        self._fetch_structure = fetch_structure
        self._structure: dict[str, AcquiredResultStructure] | None = None
        self._sliced: dict[tuple[str, str], AcquiredResult] = {}
        self._complete_dataset = False

    def _get_sliced(
        self, handle: str, nt: Literal["all", "latest"]
    ) -> AcquiredResult | None:
        """Return the result for *handle* restricted to the *nt* selection.

        Returns ``None`` when no near-time step has completed yet for the
        handle; raises ``KeyError`` for an unknown handle.
        """
        if nt == "all":
            return self[handle]
        key = (handle, nt)
        if key not in self._sliced:
            if handle in self.data or self._complete_dataset:
                # Full data already available locally: slice it here rather than
                # paying for a round trip (which would also fail once the run
                # has ended and the service is gone).
                sliced = slice_acquired_result(self.data[handle], nt)
                if sliced is None:
                    return None
                self._sliced[key] = sliced
            else:
                fetched = self._fetch((handle,), nt)
                if handle not in fetched:
                    return None
                self._sliced[key] = fetched[handle]
        return self._sliced[key]

    def _known_handles(self) -> set[str] | None:
        if self._complete_dataset:
            return set(self.data)
        return set(self._structure) if self._structure is not None else None

    def _ensure_structure(self) -> dict[str, AcquiredResultStructure]:
        if self._structure is None:
            self._structure = self._fetch_structure()
        return self._structure

    def _handles(self) -> tuple[str, ...]:
        """Every handle the service knows about, without fetching results."""
        if self._complete_dataset:
            return tuple(self.data)
        return tuple(self._ensure_structure())

    def _ensure_complete_dataset(self) -> None:
        if not self._complete_dataset:
            self.data.update(self._fetch(None, "all"))
            self._complete_dataset = True

    def __getitem__(self, key: str) -> AcquiredResult:
        if key not in self.data and not self._complete_dataset:
            self.data.update(self._fetch((key,), "all"))
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        if key in self.data:
            return True
        if self._complete_dataset or not isinstance(key, str):
            return False
        return key in self._ensure_structure()

    def __iter__(self) -> Iterator[str]:
        return iter(self._handles())

    def __len__(self) -> int:
        return len(self._handles())

    def values(self) -> ValuesView[AcquiredResult]:
        # Reading every value costs one fetch, not one per handle.
        self._ensure_complete_dataset()
        return super().values()

    def items(self) -> ItemsView[str, AcquiredResult]:
        self._ensure_complete_dataset()
        return super().items()


class NeartimeCallbackResultsFetcher(Protocol):
    """Fetches all near-time callback results accumulated so far."""

    def __call__(self) -> dict[str, list[Any]]: ...


class LazyNeartimeCallbackResults(UserDict[str, list[Any]]):
    """Near-time callback results fetched on demand from the controller service.

    Any access triggers one full fetch, cached for the lifetime of the object
    (a single callback invocation). Does not refetch new results afterwards.
    """

    def __init__(self, fetch: NeartimeCallbackResultsFetcher):
        super().__init__()
        self._fetch = fetch
        self._fetched = False

    def _ensure_fetched(self) -> None:
        if not self._fetched:
            self.data.update(self._fetch())
            self._fetched = True

    def __getitem__(self, key: str) -> list[Any]:
        self._ensure_fetched()
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        self._ensure_fetched()
        return key in self.data

    def __iter__(self):
        self._ensure_fetched()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure_fetched()
        return super().__len__()
