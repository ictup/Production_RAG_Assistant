import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from backend.app.core.config import Settings
from backend.app.db.models import ChatLog, ExportJob
from backend.app.exporting.worker import (
    ExportJobExecutionResult,
    build_export_job_filename,
    cleanup_expired_export_files,
    execute_next_export_job,
    media_type_for_format,
    run_export_worker_loop,
    safe_filename_part,
)


class FakeExportJobRepository:
    def __init__(self, export_job: ExportJob | None) -> None:
        self.export_job = export_job
        self.reset_calls: list[dict[str, Any]] = []
        self.claim_calls: list[bool] = []
        self.complete_calls: list[dict[str, Any]] = []
        self.fail_calls: list[dict[str, Any]] = []

    async def reset_stale_running_export_jobs(
        self,
        *,
        timeout_seconds: float,
        commit: bool = False,
    ) -> int:
        self.reset_calls.append(
            {
                "timeout_seconds": timeout_seconds,
                "commit": commit,
            }
        )
        return 0

    async def claim_next_pending_export_job(
        self,
        *,
        workspace_id: str | None = None,
        commit: bool = False,
    ) -> ExportJob | None:
        del workspace_id
        self.claim_calls.append(commit)
        if self.export_job is None:
            return None
        self.export_job.status = "running"
        self.export_job.started_at = datetime(2026, 5, 20, 8, 1, tzinfo=UTC)
        return self.export_job

    async def complete_export_job(
        self,
        *,
        job_id: uuid.UUID,
        result_uri: str,
        result_media_type: str,
        result_size_bytes: int | None = None,
        commit: bool = False,
    ) -> ExportJob | None:
        self.complete_calls.append(
            {
                "job_id": job_id,
                "result_uri": result_uri,
                "result_media_type": result_media_type,
                "result_size_bytes": result_size_bytes,
                "commit": commit,
            }
        )
        if self.export_job is None:
            return None
        self.export_job.status = "succeeded"
        self.export_job.result_uri = result_uri
        self.export_job.result_media_type = result_media_type
        self.export_job.result_size_bytes = result_size_bytes
        self.export_job.completed_at = datetime(2026, 5, 20, 8, 2, tzinfo=UTC)
        return self.export_job

    async def fail_export_job(
        self,
        *,
        job_id: uuid.UUID,
        error_message: str,
        commit: bool = False,
    ) -> ExportJob | None:
        self.fail_calls.append(
            {
                "job_id": job_id,
                "error_message": error_message,
                "commit": commit,
            }
        )
        if self.export_job is None:
            return None
        self.export_job.status = "failed"
        self.export_job.error_message = error_message
        self.export_job.completed_at = datetime(2026, 5, 20, 8, 2, tzinfo=UTC)
        return self.export_job


class FakeChatLogRepository:
    def __init__(self, logs: list[ChatLog]) -> None:
        self.logs = logs
        self.list_calls: list[dict[str, Any]] = []

    async def list_recent_chat_logs(
        self,
        *,
        workspace_id: str = "public",
        limit: int = 10,
        offset: int = 0,
        session_id: uuid.UUID | None = None,
        request_id: str | None = None,
        refusal_only: bool = False,
        citation_valid: bool | None = None,
    ) -> list[ChatLog]:
        self.list_calls.append(
            {
                "workspace_id": workspace_id,
                "limit": limit,
                "offset": offset,
                "session_id": session_id,
                "request_id": request_id,
                "refusal_only": refusal_only,
                "citation_valid": citation_valid,
            }
        )
        return self.logs


def make_export_job_model(
    *,
    export_format: str = "jsonl",
    export_type: str = "chat_logs",
    filters: dict[str, Any] | None = None,
) -> ExportJob:
    return ExportJob(
        id=uuid.UUID("55555555-5555-5555-5555-555555555555"),
        workspace_id="tenant-a",
        request_id="request-1",
        actor_hash="a" * 64,
        export_type=export_type,
        format=export_format,
        status="pending",
        filters_=filters or {"limit": 25, "offset": 50, "refusal_only": True},
        result_uri=None,
        result_media_type=None,
        result_size_bytes=None,
        error_message=None,
        created_at=datetime(2026, 5, 20, 8, 0, tzinfo=UTC),
        updated_at=datetime(2026, 5, 20, 8, 0, tzinfo=UTC),
        started_at=None,
        completed_at=None,
    )


def make_chat_log_model() -> ChatLog:
    return ChatLog(
        id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        request_id="chat-request-1",
        workspace_id="tenant-a",
        session_id=None,
        question="What problem does FlashAttention solve?",
        answer="FlashAttention reduces memory traffic. [1]",
        sources=[{"source_id": "1", "title": "FlashAttention Notes"}],
        retrieval={"mode": "hybrid_rrf_rerank"},
        usage={"model": "fake-llm"},
        refusal=None,
        citation_valid=True,
        latency_ms=12,
        created_at=datetime(2026, 5, 18, 8, 0, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_execute_next_export_job_writes_jsonl_and_marks_succeeded(
    tmp_path: Path,
) -> None:
    export_job = make_export_job_model()
    export_job_repository = FakeExportJobRepository(export_job)
    chat_log_repository = FakeChatLogRepository([make_chat_log_model()])

    result = await execute_next_export_job(
        export_job_repository=export_job_repository,  # type: ignore[arg-type]
        chat_log_repository=chat_log_repository,  # type: ignore[arg-type]
        settings=Settings(export_storage_dir=str(tmp_path)),
    )

    assert result is not None
    assert result.job.status == "succeeded"
    assert result.result_path is not None
    assert result.result_path.exists()
    assert result.result_path.name == build_export_job_filename(export_job)
    exported = json.loads(result.result_path.read_text(encoding="utf-8").strip())
    assert exported["request_id"] == "chat-request-1"
    assert exported["workspace_id"] == "tenant-a"
    assert chat_log_repository.list_calls == [
        {
            "workspace_id": "tenant-a",
            "limit": 25,
            "offset": 50,
            "session_id": None,
            "request_id": None,
            "refusal_only": True,
            "citation_valid": None,
        }
    ]
    assert export_job_repository.claim_calls == [True]
    assert export_job_repository.reset_calls == [
        {"timeout_seconds": 3600.0, "commit": True}
    ]
    assert export_job_repository.complete_calls[0]["commit"] is True
    assert export_job_repository.complete_calls[0]["result_media_type"] == (
        "application/x-ndjson; charset=utf-8"
    )
    assert export_job_repository.complete_calls[0]["result_size_bytes"] == (
        result.result_path.stat().st_size
    )
    assert export_job_repository.complete_calls[0]["result_uri"].startswith("file://")
    assert export_job_repository.fail_calls == []


@pytest.mark.asyncio
async def test_execute_next_export_job_writes_csv(tmp_path: Path) -> None:
    export_job = make_export_job_model(
        export_format="csv",
        filters={"limit": 1000, "offset": 0, "refusal_only": False},
    )
    export_job_repository = FakeExportJobRepository(export_job)
    chat_log_repository = FakeChatLogRepository([make_chat_log_model()])

    result = await execute_next_export_job(
        export_job_repository=export_job_repository,  # type: ignore[arg-type]
        chat_log_repository=chat_log_repository,  # type: ignore[arg-type]
        settings=Settings(export_storage_dir=str(tmp_path)),
    )

    assert result is not None
    assert result.result_path is not None
    assert result.result_path.suffix == ".csv"
    csv_text = result.result_path.read_text(encoding="utf-8")
    assert "request_id,workspace_id" in csv_text
    assert "chat-request-1" in csv_text
    assert export_job_repository.complete_calls[0]["result_media_type"] == (
        "text/csv; charset=utf-8"
    )
    assert export_job_repository.reset_calls == [
        {"timeout_seconds": 3600.0, "commit": True}
    ]


@pytest.mark.asyncio
async def test_execute_next_export_job_returns_none_without_pending_job(
    tmp_path: Path,
) -> None:
    export_job_repository = FakeExportJobRepository(None)
    chat_log_repository = FakeChatLogRepository([])

    result = await execute_next_export_job(
        export_job_repository=export_job_repository,  # type: ignore[arg-type]
        chat_log_repository=chat_log_repository,  # type: ignore[arg-type]
        settings=Settings(export_storage_dir=str(tmp_path)),
    )

    assert result is None
    assert export_job_repository.reset_calls == [
        {"timeout_seconds": 3600.0, "commit": True}
    ]
    assert export_job_repository.claim_calls == [True]
    assert export_job_repository.complete_calls == []
    assert export_job_repository.fail_calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_execute_next_export_job_marks_unsupported_job_failed(
    tmp_path: Path,
) -> None:
    export_job = make_export_job_model(export_type="workspace_audit")
    export_job_repository = FakeExportJobRepository(export_job)
    chat_log_repository = FakeChatLogRepository([])

    result = await execute_next_export_job(
        export_job_repository=export_job_repository,  # type: ignore[arg-type]
        chat_log_repository=chat_log_repository,  # type: ignore[arg-type]
        settings=Settings(export_storage_dir=str(tmp_path)),
    )

    assert result is not None
    assert result.job.status == "failed"
    assert result.result_path is None
    assert "unsupported export type" in result.job.error_message
    assert export_job_repository.complete_calls == []
    assert export_job_repository.fail_calls[0]["commit"] is True
    assert export_job_repository.reset_calls == [
        {"timeout_seconds": 3600.0, "commit": True}
    ]
    assert chat_log_repository.list_calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_execute_next_export_job_uses_configured_running_timeout(
    tmp_path: Path,
) -> None:
    export_job = make_export_job_model()
    export_job_repository = FakeExportJobRepository(export_job)
    chat_log_repository = FakeChatLogRepository([make_chat_log_model()])

    await execute_next_export_job(
        export_job_repository=export_job_repository,  # type: ignore[arg-type]
        chat_log_repository=chat_log_repository,  # type: ignore[arg-type]
        settings=Settings(
            export_storage_dir=str(tmp_path),
            export_job_running_timeout_seconds=12.5,
        ),
    )

    assert export_job_repository.reset_calls == [
        {"timeout_seconds": 12.5, "commit": True}
    ]


@pytest.mark.asyncio
async def test_execute_next_export_job_cleans_expired_export_files(
    tmp_path: Path,
) -> None:
    expired_export = tmp_path / "chat-logs-tenant-a-old.jsonl"
    expired_export.write_text("old\n", encoding="utf-8")
    old_timestamp = (datetime.now(UTC) - timedelta(seconds=120)).timestamp()
    os.utime(expired_export, (old_timestamp, old_timestamp))

    export_job_repository = FakeExportJobRepository(None)
    chat_log_repository = FakeChatLogRepository([])

    result = await execute_next_export_job(
        export_job_repository=export_job_repository,  # type: ignore[arg-type]
        chat_log_repository=chat_log_repository,  # type: ignore[arg-type]
        settings=Settings(
            export_storage_dir=str(tmp_path),
            export_file_retention_seconds=60,
        ),
    )

    assert result is None
    assert not expired_export.exists()
    assert export_job_repository.claim_calls == [True]


def test_export_filename_parts_are_sanitized() -> None:
    export_job = make_export_job_model()
    export_job.workspace_id = "../Tenant A"
    export_job.export_type = "chat logs"

    assert safe_filename_part("../Tenant A") == "Tenant-A"
    assert build_export_job_filename(export_job) == (
        "chat-logs-Tenant-A-55555555-5555-5555-5555-555555555555.jsonl"
    )


def test_media_type_for_format_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unsupported export format"):
        media_type_for_format("xml")


def test_cleanup_expired_export_files_deletes_only_expired_export_files(
    tmp_path: Path,
) -> None:
    expired_jsonl = tmp_path / "old.jsonl"
    expired_csv = tmp_path / "old.csv"
    fresh_jsonl = tmp_path / "fresh.jsonl"
    ignored_text = tmp_path / "old.txt"
    nested_dir = tmp_path / "nested.csv"
    nested_dir.mkdir()

    for path in (expired_jsonl, expired_csv, fresh_jsonl, ignored_text):
        path.write_text("payload", encoding="utf-8")

    old_timestamp = (datetime.now(UTC) - timedelta(seconds=120)).timestamp()
    fresh_timestamp = datetime.now(UTC).timestamp()
    for path in (expired_jsonl, expired_csv, ignored_text):
        os.utime(path, (old_timestamp, old_timestamp))
    os.utime(fresh_jsonl, (fresh_timestamp, fresh_timestamp))

    result = cleanup_expired_export_files(
        storage_dir=tmp_path,
        retention_seconds=60,
    )

    assert result.files_deleted == 2
    assert result.errors == 0
    assert {path.name for path in result.deleted_paths} == {"old.csv", "old.jsonl"}
    assert not expired_jsonl.exists()
    assert not expired_csv.exists()
    assert fresh_jsonl.exists()
    assert ignored_text.exists()
    assert nested_dir.is_dir()


def test_cleanup_expired_export_files_ignores_missing_storage_dir(
    tmp_path: Path,
) -> None:
    result = cleanup_expired_export_files(
        storage_dir=tmp_path / "missing",
        retention_seconds=60,
    )

    assert result.files_deleted == 0
    assert result.errors == 0
    assert result.deleted_paths == ()


def test_cleanup_expired_export_files_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retention_seconds"):
        cleanup_expired_export_files(storage_dir=tmp_path, retention_seconds=0)

    storage_file = tmp_path / "exports"
    storage_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="storage_dir"):
        cleanup_expired_export_files(storage_dir=storage_file, retention_seconds=60)


@pytest.mark.asyncio
async def test_run_export_worker_loop_sleeps_between_idle_iterations() -> None:
    export_job = make_export_job_model()
    results: list[ExportJobExecutionResult | None] = [
        None,
        ExportJobExecutionResult(job=export_job, result_path=Path("exports/job.jsonl")),
        None,
    ]
    sleep_calls: list[float] = []

    async def run_once(settings: Settings) -> ExportJobExecutionResult | None:
        del settings
        return results.pop(0)

    async def sleep(delay: float) -> None:
        sleep_calls.append(delay)

    stats = await run_export_worker_loop(
        Settings(export_worker_poll_interval_seconds=0.25),
        max_iterations=3,
        run_once=run_once,
        sleep=sleep,
    )

    assert stats.iterations == 3
    assert stats.jobs_processed == 1
    assert stats.failed_jobs == 0
    assert stats.idle_iterations == 2
    assert stats.errors == 0
    assert sleep_calls == [0.25]


@pytest.mark.asyncio
async def test_run_export_worker_loop_counts_failed_jobs() -> None:
    export_job = make_export_job_model()
    export_job.status = "failed"

    async def run_once(settings: Settings) -> ExportJobExecutionResult | None:
        del settings
        return ExportJobExecutionResult(job=export_job)

    async def sleep(delay: float) -> None:
        raise AssertionError(f"unexpected sleep: {delay}")

    stats = await run_export_worker_loop(
        Settings(),
        max_iterations=1,
        run_once=run_once,
        sleep=sleep,
    )

    assert stats.iterations == 1
    assert stats.jobs_processed == 1
    assert stats.failed_jobs == 1
    assert stats.idle_iterations == 0
    assert stats.errors == 0


@pytest.mark.asyncio
async def test_run_export_worker_loop_counts_iteration_errors() -> None:
    calls = 0
    sleep_calls: list[float] = []

    async def run_once(settings: Settings) -> ExportJobExecutionResult | None:
        nonlocal calls
        del settings
        calls += 1
        if calls == 1:
            raise RuntimeError("database temporarily unavailable")
        return None

    async def sleep(delay: float) -> None:
        sleep_calls.append(delay)

    stats = await run_export_worker_loop(
        Settings(export_worker_poll_interval_seconds=0.5),
        max_iterations=2,
        run_once=run_once,
        sleep=sleep,
    )

    assert stats.iterations == 2
    assert stats.jobs_processed == 0
    assert stats.failed_jobs == 0
    assert stats.idle_iterations == 1
    assert stats.errors == 1
    assert sleep_calls == [0.5]
