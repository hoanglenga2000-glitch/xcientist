# Path A-diff: keep 514 AppShell UI, backend/runtime P0 only

Date: 2026-10-10 (Asia/Shanghai)

## Goal
Ship P0 backend/runtime/knowledge fixes on invitation overlay without replacing the live
full workstation UI (`overlay-invitation-beta-5141838f452e` AppShell).

## WEB_PATCHES policy
Excluded UI shell / invitation workspace / visual chrome (see allowlist-removed.txt).
Included: API routes (incl. provenance healthz), server libs, non-shell screen logic, proxy, package manifests.
Baseline 514 files remain authoritative for shell routes.

## RUNTIME_PATCHES
75 modules under `src/evomind_runtime/` (knowledge, evidence, policy, assistants, etc.).

## Cannot enter via invitation overlay (needs sealed ExactRelease)
- `src/xsci/**` novice tool policy
- `src/research_os/**` score-integrity core outside evomind_runtime
- ReleaseCommit reseal away from 664a636 / tag v0.3.0

## Acceptance before apply
- candidate page.tsx identical to baseline AppShell (no TaskWorkspace homepage)
- login HTML shows DeepEvo Research Workstation / 进入工作台
- Claude default `claude_lt4net_temp` retained

## Rollback
`C:\ProgramData\EvoMind\backups\invitation-cutover-*` (restore web pointer + runtime rename + seal; do not restore DBs)
