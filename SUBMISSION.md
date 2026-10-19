# Kasparro FSE I submission checklist

## Verification status

| Assessment requirement | Status | Evidence |
|---|---|---|
| CSV parsing, exact answer text, Unicode offsets, span validation | Verified | Automated CSV and span tests; full suite result below. |
| Model-requested `get_brand_facts`, validation, and no gold in inference inputs | Verified | Fake-provider tests and trace metadata show real tool observations; gold is loaded only after development predictions are saved. |
| Deterministic extraction, joint, per-label and macro metrics | Verified | Development reports contain the full score records, missed claims, extra predictions, wrong labels and answer errors. |
| Frozen v1/v2 and four live-run artifacts | Verified | Four new timestamped reports and evidence directories are listed below. Historical reports were not overwritten. |
| FastAPI and PostgreSQL persistence | Verified | PostgreSQL test module: 13 passed; full suite includes persistence and joined retrieval. |
| Async limit, timeout, retry and failure isolation | Verified | Fake-provider concurrency/retry/timeout tests and full suite pass. |
| Actual Langfuse delivery and readback | Verified | Each new trace was fetched through the Langfuse read API. Sanitized exports are included in each run directory. |
| Check traces have no gold-dependent scores | Verified | Both check traces returned zero scores; their reports contain null quality metrics and correctness fields. |
| Clean-environment setup validation | Verified | Created a fresh temporary virtual environment, installed `requirements.txt`, and ran all 116 tests successfully. A separate machine/container was not used. |
| Candidate time spent and personal verification statement | Incomplete | The candidate's actual time and personal actions were not supplied and must not be guessed. Fill these in truthfully before submission. |
| Reviewer access to the evidence | Verified with limitation | The GitHub repository is public and sanitized trace exports are committed. Direct Langfuse UI URLs still require an authenticated member of that Langfuse project. |

## Final run artifacts

All runs used OpenAI `gpt-4o-mini`, temperature 0, a 15-second request timeout,
at most three concurrent provider requests, one retry per turn, and no more
than three model turns per answer. The reports record prompt, dataset, fact,
evaluator, and source-tree hashes. The historical report set remains unchanged.
Eight earlier assistant-generated verification runs are also present locally
from trace-instrumentation iterations; they are not the final comparison set.
Use only the four runs listed here as the final evidence set.

| Run | Report | Trace | Results |
|---|---|---|---|
| v1 development | [report](./artifacts/reports/v1_development_20261019T112543111486Z.json) | [Langfuse](https://cloud.langfuse.com/project/cmvf54g0p05rsad0kznah8fv8/traces/7b7a470bec4ada9bb2b573638cad23fc) · [sanitized export](./artifacts/runs/v1_development_20261019T112543111486Z/langfuse-export.json) | G=376, P=289, M=272, C=218; extraction F1 0.8180; joint F1 0.6556; macro F1 0.6546; 10 answer errors; 104 missed, 17 extra, 54 wrong-label claims. |
| v2 development | [report](./artifacts/reports/v2_development_20261019T112718371119Z.json) | [Langfuse](https://cloud.langfuse.com/project/cmvf54g0p05rsad0kznah8fv8/traces/8430c164937e296f999343ce300474a9) · [sanitized export](./artifacts/runs/v2_development_20261019T112718371119Z/langfuse-export.json) | G=376, P=321, M=302, C=248; extraction F1 0.8666; joint F1 0.7116; macro F1 0.7104; 6 answer errors; 74 missed, 19 extra, 54 wrong-label claims. |
| v1 final check | [report](./artifacts/reports/v1_check_20261019T112853340232Z.json) | [Langfuse](https://cloud.langfuse.com/project/cmvf54g0p05rsad0kznah8fv8/traces/af103d31d8ffbdc81e375cc4b95e60b1) · [sanitized export](./artifacts/runs/v1_check_20261019T112853340232Z/langfuse-export.json) | 10 scheduled, 10 successful, 0 errors, 99 claims; all gold-dependent metrics and correctness fields null; no Langfuse scores. |
| v2 final check | [report](./artifacts/reports/v2_check_20261019T112920687565Z.json) | [Langfuse](https://cloud.langfuse.com/project/cmvf54g0p05rsad0kznah8fv8/traces/5f79ffa75fb52d0a2877a1d16f474e73) · [sanitized export](./artifacts/runs/v2_check_20261019T112920687565Z/langfuse-export.json) | 10 scheduled, 8 successful, 2 errors, 77 claims; all gold-dependent metrics and correctness fields null; no Langfuse scores. |

The v2 development joint F1 increased by 0.0560 and extraction F1 by 0.0485
relative to this v1 run. Missed claims and answer errors decreased, while extra
predictions increased from 17 to 19 and the count of wrong-label claims stayed
at 54. These are measured outcomes, not a claim of perfect extraction or
classification.

## Trace-based diagnosis and limits

The targeted v2 change is limited to the evidence rule in
[prompts_v2.py](./app/agent/prompts_v2.py): supported/contradicted labels must
cite a retrieved fact that directly bears on the claim; otherwise use unknown.
In the v1 development trace, answer A02's span `A02:1469:1659` was labeled
supported using `cloudnote.studio.battery_50_percent`, while its development
gold label is unknown because the assertion adds a below-freezing condition.
Langfuse records `label_correct = 0` for that span. This pinpoints a scope and
condition-matching risk. The same span remained incorrectly supported in v2,
so the prompt change did **not** fix every instance; the aggregate metrics
improved but this regression remains.

Langfuse readback confirmed a run span, one answer span per scheduled answer,
model-generation observations, and `get_brand_facts` tool observations. Model
observations include request latency and available usage details. Tool traces
include call IDs, brand IDs, returned fact IDs and counts; answer spans include
accepted offsets/labels or sanitized failure codes. The exports intentionally
omit raw model inputs and outputs and redact SDK-injected key metadata. The
check traces contain no scores.

Check inputs are visible in the supplied dataset, so this is not a blind
benchmark. No prompt or behavior tuning was made from the final-check scores;
those scores are not available in these artifacts.

## Tests actually run

Full test suite:

```text
116 passed in 7.18s
```

Clean temporary virtual environment, dependencies freshly installed from
`requirements.txt`:

```text
116 passed in 13.93s
```

PostgreSQL persistence tests were also run explicitly:

```text
13 passed in 3.39s
```

The full suite and explicit database tests ran with `DATABASE_URL` configured.
The four new evaluations were run against the configured live model, exited
successfully, and their traces and scores were read back from Langfuse. These
checks were performed by the coding assistant in this session. Do not describe
them as personally run by the candidate unless the candidate independently
repeated them.

## Assistance and candidate disclosures

- Antigravity assistance was used earlier for environment/startup and Git
  setup/push coordination.
- An AI coding assistant made the focused tracing/persistence changes and ran
  the tests and four evaluations described above.
- Candidate hands-on time: **unknown**; no duration was provided.
- Candidate's own tests/checks: **none reported**. The tests and live
  evaluations above were run by the assistant in this session.
- The GitHub repository is public; direct Langfuse UI access still depends on
  the reviewer being a member of the Langfuse project. The committed sanitized
  exports provide trace evidence without requiring that access.
- No hosted deployment or separate-machine install was performed here.
- Optional Langfuse and model credentials stay in the ignored local `.env`;
  never stage that file. Ensure any previously exposed provider key has been
  revoked and replaced before submission.

## Suggested staging set

Stage the implementation/documentation changes and only the four final report
and run folders above. Do not stage `.env`, `.venv`, `.vscode`, the local
verification helper, or the eight earlier assistant-generated run folders.
From the repository root:

```powershell
git add README.md SUBMISSION.md app/agent/runner.py app/api/evaluation_runs.py app/evaluation/runs.py app/tracing.py scripts/run_check.py scripts/run_evaluation.py tests/test_tracing.py
git add artifacts/reports/v1_development_20261019T112543111486Z.json artifacts/reports/v2_development_20261019T112718371119Z.json artifacts/reports/v1_check_20261019T112853340232Z.json artifacts/reports/v2_check_20261019T112920687565Z.json
git add artifacts/runs/v1_development_20261019T112543111486Z artifacts/runs/v2_development_20261019T112718371119Z artifacts/runs/v1_check_20261019T112853340232Z artifacts/runs/v2_check_20261019T112920687565Z
```

Review `git diff --cached` and `git status` before committing or pushing.
