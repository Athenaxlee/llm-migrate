"""Opt-in bounded refetching of cited evidence sources for V1.1 review support.

This adapter never crawls, never discovers URLs, and never stores page content.
It fetches only URLs already cited by a typed research artifact, on explicit
request, and records reachability plus a content hash so reviewers and audits
can tell whether a citation was real and whether the page changed. Retrieved
bytes are hashed and discarded; full webpage copies are never persisted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_migrate.core.knowledge import ResearchResult
from llm_migrate.core.models import StrictModel

SourceFetcher = Callable[[str, float], bytes]

_MAX_CONTENT_BYTES = 5_000_000


def _fetch(url: str, timeout: float) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("evidence refetching is limited to https URLs")
    request = Request(  # noqa: S310 - scheme is validated above
        url,
        headers={"Accept": "text/html,application/json", "User-Agent": "llm-migrate"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return bytes(response.read(_MAX_CONTENT_BYTES))


class RefetchStatus(StrEnum):
    FETCHED = "fetched"
    UNREACHABLE = "unreachable"
    REFUSED = "refused"


class SourceRefetchRecord(StrictModel):
    """Reachability evidence for one cited source; content is never stored."""

    schema_version: Literal["1"] = "1"
    source_id: str
    url: str
    status: RefetchStatus
    retrieved_at: datetime
    content_sha256: str | None = None
    content_length: int | None = None
    error: str | None = None


def refetch_cited_sources(
    research: ResearchResult,
    *,
    timeout: float = 5.0,
    fetcher: SourceFetcher | None = None,
    retrieved_at: datetime | None = None,
) -> list[SourceRefetchRecord]:
    """Refetch exactly the sources a research artifact cites, and nothing else.

    Network failures are recorded, not raised: an unreachable citation is
    itself review evidence (the reviewer should verdict such claims
    `insufficient_evidence`), and the offline path stays fully usable by
    injecting a fake fetcher or skipping this call entirely.
    """
    timestamp = retrieved_at or datetime.now(UTC)
    fetch = fetcher or _fetch
    records: list[SourceRefetchRecord] = []
    for source in sorted(research.sources, key=lambda item: item.id):
        url = str(source.url)
        try:
            payload = fetch(url, timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            records.append(
                SourceRefetchRecord(
                    source_id=source.id,
                    url=url,
                    status=RefetchStatus.UNREACHABLE,
                    retrieved_at=timestamp,
                    error=str(exc),
                )
            )
            continue
        except ValueError as exc:
            records.append(
                SourceRefetchRecord(
                    source_id=source.id,
                    url=url,
                    status=RefetchStatus.REFUSED,
                    retrieved_at=timestamp,
                    error=str(exc),
                )
            )
            continue
        records.append(
            SourceRefetchRecord(
                source_id=source.id,
                url=url,
                status=RefetchStatus.FETCHED,
                retrieved_at=timestamp,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                content_length=len(payload),
            )
        )
    return records
