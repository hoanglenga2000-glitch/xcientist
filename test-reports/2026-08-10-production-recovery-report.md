# Test Results — 2026-08-10 Production Recovery Closure

## Verdict

- Engineering production candidate: **GO**.
- Formal Git publication: **GO — authorization received for `v0.3.0`**.
- No HPC connection, GPU action, Kaggle submission, external Agent execution, formal commit, tag, or push was performed during the clean-checkout closure.

## Clean-checkout results

| Gate | Result | Evidence |
| --- | --- | --- |
| Plaintext-secret scan | **PASS** | 1,926 final-candidate text files; `finding_count=0` |
| Python compile | **PASS** | All first-party Python compiles |
| Python imports | **PASS** | 170 modules import cleanly |
| Python tests | **PASS** | 2,722 collected; suite green |
| Python wrapper | **PASS** | `ALL GATES PASSED — 4/4` |
| Node tests | **PASS** | 66/66 |
| TypeScript | **PASS** | `tsc --noEmit --incremental false` |
| Lint | **PASS** | zero warnings under `--deny-warnings` |
| Next production build | **PASS** | Next.js 16.2.12; `src/proxy.ts` emitted as Proxy/Middleware |
| Prisma | **PASS** | client generation and SQLite schema push |
| Production dependency audit | **PASS** | 0 vulnerabilities |
| Candidate diff hygiene | **PASS** | alternate-index `git diff --cached --check` returned 0 |
| Dashboard build transaction fixture | **PASS** | 10/10; self-contained SQLite schema; runtime DB excluded from staging |

Python JUnit:

`D:\AI-Outputs\Codex\verification\evomind-production-candidate-20260810-final-exact.junit.xml`

## Candidate runtime smoke

Candidate source:

`D:\AI-Outputs\Codex\verification\evomind-production-candidate-20260810-1400`

Evidence root:

`D:\AI-Outputs\Codex\verification\evomind-production-runtime-smoke-019fe94a`

Managed lifecycle start:

- Dashboard: `127.0.0.1:18088`, PID `25740`, HTTP 200/ready.
- Runtime: `127.0.0.1:18765`, PID `22204`, HTTP 200/ready.
- Mode: `source-standalone`.
- `WORKSTATION_DISABLE_AGENT_EXECUTION=1`.

Authenticated verifier result: **11/11 checks passed**.

| Check | Result |
| --- | --- |
| `/api/healthz` | HTTP 200, ready |
| `/api/system/version` | HTTP 200 |
| `/api/runtime/health` | HTTP 200, ready |
| `/api/connectors/health` | HTTP 200 |
| `/api/hpc/job-lineage` | HTTP 200 |
| `/api/workstation-summary` | HTTP 200 |
| Create workstation run | `WAIT_PLAN_GATE` |
| Dispatch before plan Gate | `execution_started=false` |
| Required Gate | `plan_approval` |
| Persisted Gate decision | `pending`, `decided_at=null` |
| Summary projection | Run remains `WAIT_PLAN_GATE` |

Verified run: `wr_2026-08-10T02-43-15-869Z_sob1j`.

The summary projection used bounded polling because the production lightweight summary intentionally has a 1,000 ms cache TTL. The run appeared on attempt 5 without weakening the required state assertion.

## Shutdown and persistence

Managed stop result:

- Stopped PIDs: `25740`, `22204`.
- Dashboard listeners after stop: 0.
- Runtime listeners after stop: 0.
- Alive smoke PIDs after stop: 0.
- Bootstrap/automation auth files after stop: 0.
- SQLite `PRAGMA quick_check`: `ok`.
- Smoke fixture runs in `EXECUTING`: 0.
- All smoke fixture `process_id` values: `null`.

## Failure recovery contract

Result: **PASS**, `mock_used=false`.

Run: `recovery_crash_20260810_104332`.

Required failure bundle present exactly:

- `error.json`
- `traceback.txt`
- `environment.json`
- `node_status.json`
- `recovery_plan.json`

Recovery action suffix: `detect → analyze → repair → retry`.

Evidence:

`D:\AI-Outputs\Codex\verification\evomind-production-runtime-smoke-019fe94a\failure-recovery\recovery_crash_20260810_104332\verification.json`

## Memory and Evolution contract

Result: **PASS**, real Titanic data, `mock_used=false`.

| Candidate | Three-fold CV accuracy |
| --- | ---: |
| Dummy most frequent | 0.6161616161616161 |
| Logistic mixed features | 0.7946127946127945 |
| Random Forest mixed features | 0.8271604938271605 |

- Selected strategy: `random_forest_mixed_features`.
- Decision before memory: `dummy_most_frequent`.
- Decision after retrieval: `random_forest_mixed_features`.
- Fresh-task holdout accuracy: `0.8295964125560538`.
- Dummy holdout accuracy: `0.6143497757847534`.

Evidence:

`D:\AI-Outputs\Codex\verification\evomind-production-runtime-smoke-019fe94a\memory-evolution-report.json`

## Source and release boundary

- Real repository `HEAD`: `b4967e5e476545590e5dd1686801cec0ac27b623`.
- Existing real staged set: 35 files; preserved unchanged.
- Alternate index: `D:\AI-Outputs\Codex\verification\evomind-production-candidate-20260810-1100.index`.
- Final clean export: `D:\AI-Outputs\Codex\verification\evomind-production-candidate-20260810-final`.
- Exact final CI mirror: `D:\AI-Outputs\efci2`; temporary verification-only commit `cf5235e2383d6213671b7f55303ade38908b61d1`.
- Runtime/build/database/cache outputs are excluded from the final source candidate.
- Operator authorization was received for the formal commit, clean checkout, branch `codex/evomind-production-closure-20260810`, protected tag `v0.3.0`, and push to `origin`.
- Tag publication triggers the repository's Python, frontend, CodeQL, capability-certification, protected signing, and release-bundle verification gates.
