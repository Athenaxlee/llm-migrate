from __future__ import annotations

from datetime import UTC, datetime

from llm_migrate.adapters.evidence import RefetchStatus, refetch_cited_sources
from llm_migrate.service import MigrationService
from tests.unit.v11_scenarios import target_research

FIXED_TIME = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def test_refetch_covers_exactly_the_cited_sources() -> None:
    research = target_research()
    seen: list[str] = []

    def fetcher(url: str, timeout: float) -> bytes:
        seen.append(url)
        assert timeout == 2.0
        return b"<html>official facts</html>"

    records = refetch_cited_sources(research, timeout=2.0, fetcher=fetcher, retrieved_at=FIXED_TIME)
    assert seen == [str(source.url) for source in research.sources]
    assert all(record.status is RefetchStatus.FETCHED for record in records)
    assert all(record.content_sha256 is not None for record in records)
    # content itself is never retained on the record
    assert all(not hasattr(record, "content") for record in records)


def test_unreachable_citation_is_recorded_not_raised() -> None:
    research = target_research()

    def offline(url: str, timeout: float) -> bytes:
        raise TimeoutError("offline")

    records = refetch_cited_sources(research, fetcher=offline, retrieved_at=FIXED_TIME)
    assert all(record.status is RefetchStatus.UNREACHABLE for record in records)
    assert all(record.content_sha256 is None for record in records)
    assert all(record.error for record in records)


def test_service_exposes_explicit_refetch(service: MigrationService) -> None:
    research = target_research()
    records = service.refetch_research_sources(
        research,
        fetcher=lambda url, timeout: b"payload",
        retrieved_at=FIXED_TIME,
    )
    assert [record.source_id for record in records] == [source.id for source in research.sources]
