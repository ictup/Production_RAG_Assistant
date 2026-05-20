import argparse
import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings, get_settings
from backend.app.db.models import ExportJob
from backend.app.db.repositories import ChatLogRepository, ExportJobRepository
from backend.app.db.session import get_sessionmaker
from backend.app.exporting.chat_logs import (
    serialize_chat_logs_csv,
    serialize_chat_logs_jsonl,
)
from backend.app.schemas.export_jobs import ChatLogExportFilters

EXPORT_MEDIA_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "jsonl": "application/x-ndjson; charset=utf-8",
}
EXPORT_CLEANUP_SUFFIXES = frozenset({".csv", ".jsonl"})
LOGGER = logging.getLogger(__name__)
RunWorkerIteration = Callable[[Settings], Awaitable["ExportJobExecutionResult | None"]]
SleepCallable = Callable[[float], Awaitable[Any]]


@dataclass(frozen=True)
class ExportJobExecutionResult:
    job: ExportJob
    result_path: Path | None = None


@dataclass(frozen=True)
class ExportCleanupResult:
    files_deleted: int = 0
    errors: int = 0
    deleted_paths: tuple[Path, ...] = ()


@dataclass
class ExportWorkerLoopStats:
    iterations: int = 0
    jobs_processed: int = 0
    failed_jobs: int = 0
    idle_iterations: int = 0
    errors: int = 0


async def execute_next_export_job(
    *,
    export_job_repository: ExportJobRepository,
    chat_log_repository: ChatLogRepository,
    settings: Settings,
) -> ExportJobExecutionResult | None:
    cleanup_result = cleanup_expired_export_files(
        storage_dir=Path(settings.export_storage_dir),
        retention_seconds=settings.export_file_retention_seconds,
    )
    if cleanup_result.files_deleted:
        LOGGER.info("deleted %s expired export file(s)", cleanup_result.files_deleted)
    if cleanup_result.errors:
        LOGGER.warning(
            "failed to delete %s expired export file(s)",
            cleanup_result.errors,
        )

    reset_count = await export_job_repository.reset_stale_running_export_jobs(
        timeout_seconds=settings.export_job_running_timeout_seconds,
        commit=True,
    )
    if reset_count:
        LOGGER.warning("reset %s stale running export job(s)", reset_count)

    export_job = await export_job_repository.claim_next_pending_export_job(
        commit=True,
    )
    if export_job is None:
        return None
    return await execute_export_job(
        export_job=export_job,
        export_job_repository=export_job_repository,
        chat_log_repository=chat_log_repository,
        settings=settings,
    )


async def execute_export_job(
    *,
    export_job: ExportJob,
    export_job_repository: ExportJobRepository,
    chat_log_repository: ChatLogRepository,
    settings: Settings,
) -> ExportJobExecutionResult:
    try:
        result_path = await write_export_job_file(
            export_job=export_job,
            chat_log_repository=chat_log_repository,
            storage_dir=Path(settings.export_storage_dir),
        )
        payload_size = result_path.stat().st_size
        completed_job = await export_job_repository.complete_export_job(
            job_id=export_job.id,
            result_uri=result_path.resolve().as_uri(),
            result_media_type=media_type_for_format(export_job.format),
            result_size_bytes=payload_size,
            commit=True,
        )
        return ExportJobExecutionResult(
            job=completed_job or export_job,
            result_path=result_path,
        )
    except Exception as exc:
        failed_job = await export_job_repository.fail_export_job(
            job_id=export_job.id,
            error_message=str(exc),
            commit=True,
        )
        return ExportJobExecutionResult(job=failed_job or export_job)


async def write_export_job_file(
    *,
    export_job: ExportJob,
    chat_log_repository: ChatLogRepository,
    storage_dir: Path,
) -> Path:
    if export_job.export_type != "chat_logs":
        raise ValueError(f"unsupported export type: {export_job.export_type}")

    filters = ChatLogExportFilters.model_validate(dict(export_job.filters_))
    logs = await chat_log_repository.list_recent_chat_logs(
        workspace_id=export_job.workspace_id,
        limit=filters.limit,
        offset=filters.offset,
        session_id=filters.session_id,
        request_id=filters.request_id,
        refusal_only=filters.refusal_only,
        citation_valid=filters.citation_valid,
    )

    if export_job.format == "csv":
        payload = serialize_chat_logs_csv(logs).encode("utf-8")
    elif export_job.format == "jsonl":
        payload = serialize_chat_logs_jsonl(logs).encode("utf-8")
    else:
        raise ValueError(f"unsupported export format: {export_job.format}")

    storage_dir.mkdir(parents=True, exist_ok=True)
    result_path = storage_dir / build_export_job_filename(export_job)
    result_path.write_bytes(payload)
    return result_path


def media_type_for_format(export_format: str) -> str:
    try:
        return EXPORT_MEDIA_TYPES[export_format]
    except KeyError as exc:
        raise ValueError(f"unsupported export format: {export_format}") from exc


def build_export_job_filename(export_job: ExportJob) -> str:
    workspace = safe_filename_part(export_job.workspace_id)
    export_type = safe_filename_part(export_job.export_type)
    export_format = safe_filename_part(export_job.format)
    return f"{export_type}-{workspace}-{export_job.id}.{export_format}"


def safe_filename_part(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip())
    return normalized.strip(".-") or "export"


def cleanup_expired_export_files(
    *,
    storage_dir: Path,
    retention_seconds: float,
) -> ExportCleanupResult:
    if retention_seconds <= 0:
        raise ValueError("retention_seconds must be greater than zero")

    storage_root = storage_dir.resolve()
    if not storage_root.exists():
        return ExportCleanupResult()
    if not storage_root.is_dir():
        raise ValueError("storage_dir must be a directory")

    cutoff_timestamp = datetime.now(UTC).timestamp() - retention_seconds
    deleted_paths: list[Path] = []
    errors = 0
    for entry in storage_root.iterdir():
        if entry.suffix not in EXPORT_CLEANUP_SUFFIXES or not entry.is_file():
            continue
        try:
            if entry.stat().st_mtime > cutoff_timestamp:
                continue
            entry.unlink()
            deleted_paths.append(entry)
        except OSError:
            errors += 1
            LOGGER.exception("failed to delete expired export file %s", entry)

    return ExportCleanupResult(
        files_deleted=len(deleted_paths),
        errors=errors,
        deleted_paths=tuple(deleted_paths),
    )


async def run_export_worker_once(settings: Settings | None = None) -> str:
    result = await run_export_worker_iteration(settings=settings)
    return format_export_worker_result(result)


async def run_export_worker_iteration(
    settings: Settings | None = None,
) -> ExportJobExecutionResult | None:
    settings = settings or get_settings()
    async with get_sessionmaker()() as session:
        return await execute_next_export_job(
            export_job_repository=ExportJobRepository(session=session),
            chat_log_repository=ChatLogRepository(session=session),
            settings=settings,
        )


def format_export_worker_result(result: ExportJobExecutionResult | None) -> str:
    if result is None:
        return "no pending export job"
    if result.result_path is None:
        return f"export job {result.job.id} failed"
    return f"export job {result.job.id} wrote {result.result_path}"


async def run_export_worker_loop(
    settings: Settings | None = None,
    *,
    max_iterations: int | None = None,
    run_once: RunWorkerIteration = run_export_worker_iteration,
    sleep: SleepCallable = asyncio.sleep,
) -> ExportWorkerLoopStats:
    settings = settings or get_settings()
    stats = ExportWorkerLoopStats()

    while max_iterations is None or stats.iterations < max_iterations:
        try:
            result = await run_once(settings)
        except Exception:
            stats.iterations += 1
            stats.errors += 1
            LOGGER.exception("export worker iteration failed")
            if max_iterations is None or stats.iterations < max_iterations:
                await sleep(settings.export_worker_poll_interval_seconds)
            continue

        stats.iterations += 1
        if result is None:
            stats.idle_iterations += 1
            LOGGER.debug("no pending export job")
            if max_iterations is None or stats.iterations < max_iterations:
                await sleep(settings.export_worker_poll_interval_seconds)
            continue

        stats.jobs_processed += 1
        if result.result_path is None:
            stats.failed_jobs += 1
            LOGGER.warning("export job %s failed", result.job.id)
        else:
            LOGGER.info("export job %s wrote %s", result.job.id, result.result_path)

    return stats


async def run_export_worker_forever(settings: Settings | None = None) -> None:
    await run_export_worker_loop(settings=settings)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run export job worker")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously, polling for pending export jobs.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=positive_float,
        default=None,
        help="Override EXPORT_WORKER_POLL_INTERVAL_SECONDS for loop mode.",
    )
    return parser.parse_args(argv)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    if args.poll_interval_seconds is not None:
        settings = settings.model_copy(
            update={
                "export_worker_poll_interval_seconds": args.poll_interval_seconds,
            }
        )
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.loop:
        LOGGER.info(
            "starting export worker loop with %.3fs poll interval",
            settings.export_worker_poll_interval_seconds,
        )
        try:
            asyncio.run(run_export_worker_forever(settings=settings))
        except KeyboardInterrupt:
            LOGGER.info("export worker stopped")
        return

    print(asyncio.run(run_export_worker_once(settings=settings)))


if __name__ == "__main__":
    main()
