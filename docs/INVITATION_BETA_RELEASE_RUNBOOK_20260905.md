# EvoMind invitation beta: release and rollback runbook

The user authorized deployment after successful testing on 2026-09-05. This is
authorization to deploy a safe candidate, not permission to interrupt unresolved
workers, ignore tenant ownership, or label local metrics official results.

## Repair contract

1. Dependency executor receipts use `evomind.dependency_install.v1`, exact requested
   pins and target, a SHA-256 lock identity, a real boolean reuse flag and the
   pre-install-hashing invariant. Plain text, duplicate keys, extra documents,
   missing fields, wrong targets and oversized output fail closed.
2. Repeated dependency preparation reuses the same hash-keyed base directory.
   A failed receipt does not update the effective dependency target.
3. Runtime close first prevents new local workers and deferred resumes, then
   waits for registered workers/waiters. `close(timeout)` returns `False` while
   work is active and leaves databases open. The operator must retry close only
   after work settles. No remote cancellation, signal or replacement is sent.
4. ACL database connections commit or roll back and then close explicitly on
   both success and exception paths.

## Local validation

Use the repository interpreter, not the unrelated system Python:

```powershell
& .venv/Scripts/python.exe -m pytest -o 'addopts=' -q -W error::pytest.PytestUnhandledThreadExceptionWarning tests/test_invitation_release_repairs.py tests/test_invitation_tenant_access.py tests/test_invitation_execution_progress.py tests/test_invitation_aibuild.py tests/test_invitation_budget_queue.py tests/test_hpc_execute_solution_workspace.py tests/test_evomind_runtime_http_server.py tests/test_goal_board_http.py tests/test_evomind_runtime.py tests/test_assistant_run_service.py
```

The new repair tests use local installation fixtures, not real downloaded wheels
or GPU jobs. Older tests must also remain free of unhandled background thread
exceptions. A passing fixture is not an HPC installation/driver acceptance test.

From `web/research-agent-workstation`, run the targeted Web contracts, typecheck
and production build. Preserve the actual test/build outputs separately.

## Pre-activation gates

- Read the current release and process identity through the pinned
  `evomind-shanghai` connection. Do not infer the runtime package from a Web
  health response's build metadata alone.
- Establish settlement of every relevant current worker/tool call. A stored
  `running` value does not prove training, and a stale timestamp does not prove
  exit. Require execution receipt/worker evidence before a runtime restart.
- Any HPC inspection still requires the designated proxy, current named profile,
  intended container and complete identity gate. Never probe allocation private
  IPs as a fallback. No competition submission or private labels are authorized.
- Build an immutable allowlisted candidate against the actual baseline. Old G21
  bootstrap/build scripts and old fixed-module-count installers are not suitable
  for the expanded invitation runtime without an independently reviewed update.
- Freeze a consistent quiesced backup of `runtime.sqlite3`, its SQLite sidecars
  as applicable, `principal_access.sqlite3`, worker state and immutable artifacts.
  Protect backup credentials and do not put them in source or evidence reports.
- Resolve historical ownership with an explicit reviewed migration allowlist.
  Do not grant new users all historical data by guessing from prompts/metadata.
- Complete scoped authenticated API acceptance and the real Chrome desktop and
  mobile journey. Unit/contract tests and health-only checks are insufficient.

## Activation

Only after all pre-activation gates pass, stage through the existing sealed
release mechanism, verify candidate identity, switch the service atomically,
and accept one invitation user first. Keep AIBuild flags off until its separate
real-provider/managed-executor acceptance is complete. Do not rewrite historic
scores, consume new evaluation labels, or start a replacement GPU worker as an
implicit side effect of deployment.

## Rollback triggers and procedure

Any cross-user access, duplicate worker, missing ownership data, secret exposure,
false completion, failed recovery or artifact corruption blocks activation or
triggers rollback. Stop admitting new work, settle current work through its
managed lifecycle, restore the previous sealed release pointer, then validate
its identity and scoped API behavior. Restore database/ACL backups only if the
approved migration requires it and no new writes would be lost. Do not use a
blanket directory overwrite or force-kill a GPU job to perform rollback.

## Scientific gate

This engineering release does not prove research gain or completion of three
competition trainings. The 12-GPU-hour independent study, real model reload
adapters, CURE public-only evaluation, E2LMC independent-group requirement and
Ariel split/scorer audit remain separate acceptance items.
