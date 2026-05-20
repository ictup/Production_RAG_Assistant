# Secret Manager Mapping

This guide defines which runtime values must be stored as managed secrets,
which values can remain plain deployment configuration, and how to validate the
runtime environment before promotion.

Use this document for shared staging, production-style remote deployments, and
real production. Local development can continue to use `.env`, but `.env`
must never be committed.

## Deployment Rule

Production deployments should inject configuration as environment variables at
process start. The storage backend can be a cloud secret manager, a deployment
platform secret store, Kubernetes secrets, Docker secrets projected into
environment variables, or another managed equivalent.

Do not mount a long-lived `.env` file as the source of truth for shared or
production deployments. Treat `.env.example` only as a schema template.

## Managed Secrets

Store these values in a secret manager or platform secret store:

| Variable | Why it is secret | Notes |
| --- | --- | --- |
| `API_KEYS` | Contains Bearer tokens accepted by the API. | Use long random tokens. In production, do not use `dev-key` or short placeholders. |
| `API_KEY_ROLES` | Contains API key values as the left side of the role mapping. | Use `admin`, `operator`, or `viewer`. Keep it secret because it repeats token material. |
| `API_KEY_WORKSPACE_ACCESS` | Contains API key values as the left side of the mapping. | Required when keys should be scoped to specific workspaces. Keep it secret because it repeats token material. |
| `POSTGRES_PASSWORD` | Database password used by Compose-managed Postgres. | For managed databases, the password is usually embedded in the database URLs instead. |
| `DATABASE_URL` | Contains database username, password, host, and database name. | Runtime async database URL used by the API and worker. |
| `SYNC_DATABASE_URL` | Contains database username, password, host, and database name. | Sync database URL used by Alembic migrations. |
| `OPENAI_API_KEY` | Provider credential for OpenAI-compatible calls. | Required only when an OpenAI provider is enabled. |

`PROVIDER_PRICE_TABLE` is not normally a secret, but it is deployment-specific
configuration. Keep real pricing outside code because provider prices change
over time.

## Plain Runtime Configuration

These values can usually be plain deployment configuration rather than secrets:

| Variable | Notes |
| --- | --- |
| `APP_NAME`, `APP_VERSION`, `ENV`, `LOG_LEVEL` | Operational labels and logging behavior. |
| `POSTGRES_USER`, `POSTGRES_DB`, `POSTGRES_PORT`, `POSTGRES_LOG_MIN_DURATION_STATEMENT_MS` | Non-secret database metadata and local Compose port/log settings. |
| `API_PORT` | Host port published by Compose. |
| `EXPORT_STORAGE_DIR`, `EXPORT_WORKER_POLL_INTERVAL_SECONDS`, `EXPORT_JOB_RUNNING_TIMEOUT_SECONDS`, `EXPORT_FILE_RETENTION_SECONDS` | Export worker storage and lifecycle behavior. |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` | Embedding provider selection and schema-compatible dimension. |
| `OPENAI_BASE_URL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_TIMEOUT_SECONDS`, `OPENAI_MAX_RETRIES`, `OPENAI_RETRY_DELAY_SECONDS`, `OPENAI_MAX_OUTPUT_TOKENS` | Provider endpoint, model names, timeout, and retry behavior. Treat `OPENAI_BASE_URL` as secret only if it embeds private credentials. |
| `CORS_ALLOWED_ORIGINS`, `CORS_ALLOWED_ORIGIN_REGEX`, `CORS_ALLOW_CREDENTIALS` | Browser boundary configuration. |
| `RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`, `RATE_LIMIT_EXCLUDED_PATHS` | API rate limiting behavior. |
| `QUERY_REWRITER_PROVIDER`, `QUERY_REWRITE_MODEL`, `QUERY_REWRITE_MAX_OUTPUT_TOKENS`, `QUERY_CONTEXT_HISTORY_LIMIT` | Query rewrite behavior. |
| `RERANKER_PROVIDER`, `RERANKER_MODEL`, `RERANK_TOP_N` | Reranker behavior. |
| `VECTOR_TOP_K`, `SPARSE_TOP_K`, `FUSED_TOP_K`, `RRF_K` | Retrieval and fusion behavior. |
| `GENERATOR_PROVIDER`, `LLM_MODEL`, `REFUSAL_SCORE_THRESHOLD` | Generation and refusal behavior. |

## Injection Patterns

For production-style local Compose, `.env` is acceptable because it stays on
the local machine:

```powershell
Copy-Item .env.example .env
docker compose -f docker-compose.prod.yml config --quiet
uv run python -m backend.app.core.config_check --production
```

For a managed platform, configure the same variable names in the platform
secret/config UI or CLI. The application does not need provider-specific code as
long as the process receives the variables in its environment.

For Kubernetes-style deployments, keep secret values in `Secret` objects and
non-secret values in `ConfigMap` objects, then expose both as environment
variables to the API, migration job, and export worker.

The API service, migration job, and export worker must receive consistent
database and provider settings:

- `api` requires `DATABASE_URL`, `API_KEYS`, provider settings, CORS, rate limit,
  and export settings.
- `migrate` requires `SYNC_DATABASE_URL`.
- `export-worker` requires `DATABASE_URL`, export settings, and the same
  workspace/API key configuration used by the API.

## Rotation Workflow

Rotate `API_KEYS` without downtime by temporarily accepting both old and new
tokens:

1. Generate a new long random API token.
2. Add the new token to `API_KEYS`.
3. Add or update the matching `API_KEY_ROLES` entry.
4. Add or update the matching `API_KEY_WORKSPACE_ACCESS` entry.
5. Deploy and run `uv run python -m backend.app.core.config_check --production`.
6. Move clients to the new token.
7. Remove the old token from `API_KEYS`, `API_KEY_ROLES`, and
   `API_KEY_WORKSPACE_ACCESS`.
8. Deploy again and verify authenticated API smoke tests.

Rotate `OPENAI_API_KEY` by updating the managed secret, redeploying, and running
the provider smoke commands from `docs/PROJECT_HANDOFF.md`.

Rotate database credentials by following the database provider's rotation
process, updating `DATABASE_URL` and `SYNC_DATABASE_URL` together, and running
the migration job before promoting traffic.

## Required Preflight

Before promoting a shared or production deployment, run these checks in an
environment that sees the same variables as the runtime process:

```powershell
uv run python -m backend.app.core.config_check --production
docker compose -f docker-compose.prod.yml config --quiet
```

The configuration preflight must have zero errors. Warnings require explicit
review before production promotion.

Never use a command that prints resolved environment values into logs when real
secrets are present.
