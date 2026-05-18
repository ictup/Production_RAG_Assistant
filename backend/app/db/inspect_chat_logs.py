import argparse
import asyncio
from dataclasses import dataclass

from sqlalchemy import func, select

from backend.app.db.models import ChatLog
from backend.app.db.repositories import ChatLogRepository
from backend.app.db.session import get_sessionmaker


@dataclass(frozen=True)
class ChatLogSnapshot:
    workspace_id: str
    logs_count: int
    recent_logs: list[ChatLog]


def truncate_text(value: str, *, max_length: int = 80) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."


async def load_chat_log_snapshot(
    *,
    workspace_id: str = "public",
    limit: int = 5,
) -> ChatLogSnapshot:
    workspace_id = workspace_id.strip() or "public"
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        logs_count = await session.scalar(
            select(func.count(ChatLog.id)).where(ChatLog.workspace_id == workspace_id)
        )
        recent_logs = await ChatLogRepository(session).list_recent_chat_logs(
            workspace_id=workspace_id,
            limit=limit,
        )

    return ChatLogSnapshot(
        workspace_id=workspace_id,
        logs_count=logs_count or 0,
        recent_logs=recent_logs,
    )


def format_chat_log_snapshot(snapshot: ChatLogSnapshot) -> str:
    lines = [
        f"workspace: {snapshot.workspace_id}",
        f"chat logs: {snapshot.logs_count}",
        "recent logs:",
    ]
    if not snapshot.recent_logs:
        lines.append("- none")
        return "\n".join(lines)

    for log in snapshot.recent_logs:
        refusal_reason = log.refusal.get("reason") if log.refusal else None
        lines.extend(
            [
                f"- request_id: {log.request_id}",
                f"  created_at: {log.created_at}",
                f"  citation_valid: {log.citation_valid}",
                f"  refusal_reason: {refusal_reason or 'none'}",
                f"  latency_ms: {log.latency_ms}",
                f"  question: {truncate_text(log.question)}",
            ]
        )
    return "\n".join(lines)


def validate_chat_log_snapshot(
    snapshot: ChatLogSnapshot,
    *,
    min_logs: int,
) -> None:
    if snapshot.logs_count < min_logs:
        raise SystemExit(
            f"chat log check failed: logs {snapshot.logs_count} < required {min_logs}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect recent chat logs.")
    parser.add_argument("--workspace-id", default="public")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-logs", type=int, default=0)
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = await load_chat_log_snapshot(
        workspace_id=args.workspace_id,
        limit=args.limit,
    )
    print(format_chat_log_snapshot(snapshot))
    validate_chat_log_snapshot(snapshot, min_logs=args.min_logs)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
