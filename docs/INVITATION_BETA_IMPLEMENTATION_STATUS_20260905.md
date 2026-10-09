# Invitation beta candidate: implementation and release boundaries

Status: LOCAL CANDIDATE, NOT DEPLOYED. Engineering acceptance and research gain
remain separate. This document does not certify current production or HPC state.

## Implemented candidate scope

| Area | Candidate implementation | Remaining evidence |
| --- | --- | --- |
| Navigation | Five persistent core entries; advanced routes retained; assistant viewport and draft recovery | Desktop/mobile Chrome journey |
| Execution | Versioned progress, source distinction, stale heartbeat, immutable attempts and resource locks | Real download/train/inference/failure/restart traces |
| Model download | Pinned official GPT-OSS revision, manifest hashes, strict ranges, bounded retries, persistent partials | Settled previous worker and guarded live execution |
| Dependencies | Pre-install official wheel hash lock and isolated installation | Real driver/wheel compatibility and failure testing |
| AIBuildAI-2 | Original paper-mechanism role orchestrator, opt-in invitation flag, durable approvals and role budgets | Real provider-driven guarded run and recovery proof |
| Knowledge | Project L1/L2 records, evidence-bound promotion, negative lessons | Measured effect on model delivery; not the author's full corpus |
| Research methods | MCGS Manager recommendation and claim audit wired in candidate | Same-budget independent ablation |
| Model evidence | Independent non-pickle linear NPZ reload and metric recomputation | Torch/joblib/competition adapters and bounded archive loading |
| Study budget | Persistent 27-trial/12-hour ledger foundation | Integration into actual trial orchestration; no study executed |
| Object access | Server-derived tenant AND user scope; durable additive ownership ACL | Public deployed session/API/worker end-to-end acceptance |

## Object-access contract

- Both assistant and generic runtime Web proxies derive scope from the verified
  session, overwrite identity headers, restrict upstream destinations to local
  HTTP runtime endpoints, and prohibit redirect forwarding.
- The runtime service bearer remains mandatory. Scope headers are not login
  credentials and cannot authenticate a caller on their own.
- Runs, sessions, events, approvals, artifact previews/bytes, upload chunks,
  completion, attachment reuse and conversation inheritance require ownership.
- Ownership is registered before a newly created session can start a worker.
  The additive `principal_access.sqlite3` ACL must travel with `runtime.sqlite3`
  in a quiesced backup/restore. It is not a replacement for the runtime database.
- Existing unowned data is fail-closed to invitation users. No migration assigns
  ownership from prompts, browser metadata or guessed historical HPC fields.
- Legacy service-token internal callers and verified loopback automation remain
  administrative. Their credentials must not reach a browser or untrusted code.
- The generic runtime proxy is read-only for browser sessions during beta.
  Mutations must use governed assistant endpoints. Global benchmark/super-agent
  data without tenant ownership is not exposed to invitation sessions.
- This slice scopes HTTP object access. It is not a claim that every legacy
  tool, arbitrary-code sandbox, knowledge store or administrative API has passed
  tenant-isolation/security acceptance.

## Validation scope

The targeted suite includes two tenants, different users in the same tenant,
real HTTP object reads/writes, real chunk uploads, approval denial, synthetic
artifact byte/preview authorization, ACL persistence, existing HTTP regressions,
execution/research scheduler tests and Web scope/header contracts. Test execution
results are reported separately by the implementation task. Test fixtures do not
start a GPU worker, make provider requests, install wheels or download models.

## Release gates still open

1. Obtain current read-only production/worker settlement. Do not replace or
   restart an unconfirmed active worker based on the historical download state.
2. Complete the source-derived release allowlist, immutable candidate packaging,
   quiesced database/ACL backup and rollback procedure. Do not use old 32-module
   release scripts for a candidate containing additional runtime modules.
3. Exercise real scoped public sessions, worker recovery, exact approvals and
   protected artifact paths. Ownership migration, if needed, requires a reviewed
   explicit allowlist; missing ACL data blocks access rather than granting it.
4. Complete Chrome desktop/mobile accessibility and authenticated user journeys.
   The previously reported Chrome plugin mismatch is not marked repaired here.
5. Establish five-user GPU serialization, API p95 and progress-latency evidence.
6. Add real model adapters and independent data/scorer contracts before the
   12-GPU-hour study. Ledger existence is not budget enforcement on real trials.

## Scientific reporting boundary

CURE remains authorized public validation/inference only, never official final
Phase 2. E2LMC needs independent groups or an organizer evaluator. Ariel's older
metric is not promoted without split/scorer/model reload evidence. No competition
submission, terms acceptance, private label access, new scientific score, or
claim of beating all three paper baselines is produced by this candidate.
