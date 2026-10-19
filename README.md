# Kasparro FSE I

Claim extraction and evaluation service for the Kasparro FSE I assessment.
Run PowerShell commands from the project root.

## Install

Prerequisites: Python 3.11 or newer, Docker Desktop with Compose, and an
OpenAI API key for live model requests and evaluations.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` before starting the database. Set `POSTGRES_PASSWORD` to a
locally chosen password and set `DATABASE_URL` to the SQLAlchemy URL for that
same local database:

```text
postgresql+psycopg2://<POSTGRES_USER>:<POSTGRES_PASSWORD>@localhost:5433/<POSTGRES_DB>
```

Set `OPENAI_API_KEY` for live model requests. Never commit `.env`. Optional
Langfuse tracing requires `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`;
`LANGFUSE_BASE_URL` may be set for a self-hosted service. Traces include
version/run/answer identifiers, accepted span offsets, validation error codes,
request latency, and provider token usage when available; they do not include
prompt text, answer text, or fact-result contents. Development scores are
attached only after deterministic matching. Check runs do not receive gold-
dependent scores. If Langfuse is not configured, trace IDs remain null and
remote tracing is disabled.

## Start and initialize PostgreSQL

The Compose service reads `POSTGRES_USER`, `POSTGRES_PASSWORD`, and
`POSTGRES_DB` from `.env`. Set the same values in `DATABASE_URL` before
starting the database.

```powershell
docker compose up -d db
docker compose ps
docker compose exec -T db pg_isready -U kasparro -d kasparro
```

FastAPI creates `evaluation_runs`, `answer_results`, and `claim_results`
idempotently when the application starts. No manual SQL initialization is
required.

## Validate data and run tests

```powershell
python scripts\validate_data.py
python -m pytest -q
```

API and PostgreSQL persistence tests only:

```powershell
python -m pytest tests\test_api.py tests\test_database.py -q
```

## Start the API

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The liveness endpoint is `http://127.0.0.1:8000/healthz`. Interactive API
documentation is at `http://127.0.0.1:8000/docs`.

## Run live evaluations

These commands call the configured model and may incur provider charges. Run
them from the project root after setting `OPENAI_API_KEY` in `.env`:

```powershell
python scripts\run_evaluation.py --version v1
python scripts\run_evaluation.py --version v2
python scripts\run_check.py --version v1
python scripts\run_check.py --version v2
```

The development commands evaluate the 40 development answers against gold.
The check commands process the 10 final-check answers without loading gold;
their gold-dependent metrics and correctness fields remain null. Each new run
uses a timestamped report and evidence directory, preserving existing reports.
Detailed run evidence is written under `artifacts\runs`. Each run's
`langfuse-traces.json` records the actual run trace ID, direct trace URL, base
URL, and answer observation IDs when tracing is enabled. Open the URL while
signed into the configured Langfuse project. Development trace scores include
per-answer extraction/joint F1, matched-claim `label_correct` scores, aggregate
metrics, and the missed/extra claim IDs. Check traces record predictions and
errors only. Disabled tracing records null IDs instead.

See [SUBMISSION.md](./SUBMISSION.md) for the final evidence checklist, measured
v1/v2 comparison, trace links, and remaining candidate-specific disclosures.
