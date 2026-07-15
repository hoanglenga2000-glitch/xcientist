# XCIENTIST New-User Release Checklist

This checklist defines the launch boundary for a downloaded copy of the
AI Research Workstation.

Default gateway:

```text
http://127.0.0.1:8088/?page=control
```

## Release Target

The new-user release is considered ready when a fresh user can:

1. Install Python and frontend dependencies.
2. Install `evomind`, compatibility aliases, and `kaggle-official` wrappers.
3. Open the EvoMind workstation gateway on port `8088`.
4. Use the terminal research agent and see honest readiness status.
5. Use web pages for control, tasks, data, GPU status, evidence, literature,
   workflow, code, runtime, experiments, reports, gates, and settings.
6. Run non-training API and UI smoke tests.
7. Store credentials through DPAPI helpers instead of committing secrets.
8. See GPU/HPC, DeepSeek cache, and Kaggle submission blockers as gates, not as
   hidden failures.

## Acceptance Command

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_new_user_release_acceptance.ps1
```

Fast mode without browser smoke:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_new_user_release_acceptance.ps1 -SkipBrowserSmoke
```

The command writes:

```text
workspace/new_user_release_acceptance.json
reports/NEW_USER_RELEASE_ACCEPTANCE.md
```

Before creating a version tag, scan both the current tree and every blob
reachable from local Git refs:

```powershell
python scripts\verify_no_plaintext_secrets.py --history
```

This gate must pass after credential rotation and any required history rewrite.
Deleting a sensitive file only from the latest commit is not sufficient.
The tag-only `release-artifacts` CI job enforces this as
`history_secret_scan` before building or publishing assets.

## Required Passing Checks

The acceptance command must pass:

| Check | Requirement |
| --- | --- |
| `python_core_compile` | XSCI terminal agent and release verifier compile |
| `powershell_script_parse` | installer, launcher, secret-management, and acceptance PowerShell scripts parse cleanly |
| `installer_smoke_no_secrets` | `install.ps1` can run in no-secret smoke mode without starting training |
| `cli_tests` | CLI, setup, routing, menu, and stream tests pass |
| `frontend_typecheck` | TypeScript typecheck passes |
| `frontend_build` | Next.js production build passes |
| `restart_workstation_frontend` | `8088` starts and CSS loads |
| `new_user_release_readiness_live` | new-user live release gate passes |
| `workstation_launch_readiness` | launch gate has no critical failures |
| `browser_render_smoke` | browser renders workstation pages |
| `click_smoke` | clickable UI smoke has no runtime errors |
| `interactive_controls` | visible controls have action/test/component contracts |
| `secret_scan` | launch-critical files do not contain plaintext secrets |
| `cli_routing` | `evomind` resolves to `%USERPROFILE%\.xsci\bin` first |

## Acceptable Optional Training Blockers

These do not block the new-user control-plane release:

- `gpu_resource_blocked`
- `deepseek_cache_below_80_for_batch_generation`

They do block real GPU training and large-scale batch code generation until
the corresponding gate passes.

## Not Proved By This Release

This release does not prove:

- official Kaggle submission success;
- official score, rank, top-30 status, or medal;
- MLE-Bench-75 coverage or parity;
- remote GPU training availability;
- DeepSeek cache hit rate at or above 80 percent in current runtime.
- Codex or Claude Code research-parity without an externally anchored hidden-suite certificate;
- trustworthy self-evolution without an active strict-improvement campaign, runtime-tree canary, and rollback proof.

Those claims require separate official artifacts and claim audit evidence.

## Stable Research-Parity Gate

A stable tag must fail closed unless all of the following are true:

1. `EVOMIND_CAPABILITY_EVIDENCE_URL` points to an externally supplied evidence bundle.
2. Bundle, report, suite, and evaluator SHA-256 values are configured independently.
3. At least two named baselines include Codex and Claude Code.
4. The hidden suite contains at least 100 tasks, 8 domains, 3 tasks per domain, and 3 repeats.
5. The strict raw-trial matrix covers every task, repeat, domain, and agent exactly once; recomputed aggregates and paired tables match the signed report.
6. Candidate Wilson lower bound is at least `0.75`, paired non-inferiority passes, and timeout/scope/claim violations are zero.
7. A frozen-evaluator campaign evaluates at least two immutable candidates and promotes only a strict improvement.
8. Human approval, runtime-tree activation, CAS promotion, and rollback evidence all verify.
9. The active champion commit/tree and campaign evidence bytes match the external certificate bindings.
10. The release source ZIP is byte-identical to the externally certified source archive.

Runtime status commands:

```powershell
evomind certification-status
evomind upgrade-campaign status
evomind parity-status
```

`Engineering Beta` may remain usable while these commands report `blocked`.
The phrases `research parity certified`, `Claude-level`, and `Codex-level` are
reserved for a passing external certificate bound to the exact released source.

## Human Gate Policy

Official Kaggle submission is not automatic. A candidate can only be submitted
after:

1. submission audit passes;
2. claim audit allows an official-submission candidate;
3. CV/public-gap risk is acceptable;
4. the UI/action trace records the candidate;
5. a human explicitly approves the submission gate.

## Release Wording

Allowed:

```text
EvoMind / XCIENTIST AI Research Workstation is ready for new-user gateway
release. The terminal agent, web workstation, report/evidence/literature/code
pages, and launch gates are verifiable. Training resources and official Kaggle
submission remain gate-controlled.
```

Not allowed:

```text
The system guarantees Kaggle medals.
The system has completed MLE-Bench 75.
The system has exceeded MLEvolve.
The GPU training path is production-ready without a fresh GPU smoke.
```
