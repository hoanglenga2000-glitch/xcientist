# R116 Goal-only runtime deployment architecture audit

Status: this implementation thread performs local changes only. A separately owned production attempt reached Goal bootstrap, failed, and reported verified rollback to R115; this thread did not deploy or retry it.

## Frozen scope

- Fixed Run: `run_7b1efb878afb40f396db431e91f093a5`
- Allocation: `G21`
- Current runtime baseline: R115 ZIP SHA-256 `fc65d7134f9f53fa6eda21ee2fcd804a330d9184a83d5206a6b7469a217d89ba`
- Expected installed R115 target-root tree SHA-256: `b92cc96dae5ac1c1ed3641999eec56eda782b4959ac8709a0c0584ac1438955a`
- Web build remains `overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc`.
- The deployment may replace only `C:\ProgramData\EvoMind\bundle\runtime\evomind_runtime` and may add the fixed Goal through the authenticated loopback Goal API.

## Existing release protocol audit

The useful protocol elements are:

- `Deploy-RuntimePackageR1Remote.ps1`: target-root tree hashing, local-only HPC binding/profile/BYOA snapshots, transaction mutex, managed service wrapper, atomic runtime directory replacement, release reseal, exact health checks, and rollback structure.
- `Deploy-RuntimeSyncR1Remote.ps1`: separate current/candidate build identities, runtime/other-tree checks, and rollback health verification.
- `Deploy-RuntimePackageR30.ps1`: local and remote input hash verification plus PowerShell parser verification before invoking the remote transaction.

The following paths are explicitly not reusable for R116:

- `Deploy-RuntimePackageR113Remote.ps1` creates a temporary scheduled task and invokes the main task directly during start recovery. It also has a direct residual-listener process termination path. Both violate the R116 wrapper-only service mutation contract.
- `Recover-R112ServicesEmergency.ps1` and `CleanupAndStart-R112ServicesEmergency.ps1` directly control scheduled tasks; the latter also terminates processes. They are emergency historical evidence, not the R116 deployment protocol.
- The generic 30-file deployment scripts accept the old hotfix manifest and cannot safely be widened for a 32-file Goal runtime.

## Dedicated R116 protocol

1. `verify_g21_goal_runtime_r116_candidate.py` independently closes the R115 embedded and external manifests, recomputes all seven unified diffs and their aggregate SHA, verifies the production target-root tree hash, performs two isolated 32-file compile/HTTP rounds from the hash-bound bootstrap artifact, checks the relevant JUnit identities, and runs the mandatory secret and `git diff --check` gates.
2. `new_g21_goal_runtime_deployment_approval.py` accepts only the formal verifier receipt and emits `evomind.g21_goal_runtime_deployment_approval.v1`. It binds R115, R116, source manifest, build result, bootstrap evidence, fixed Run/G21, Web build, verifier/deployer hashes, expiry, and the retained rejection evidence for superseded candidate SHA `37929a1c...`.
3. `Deploy-G21GoalRuntimeR116.ps1` accepts only that approval artifact. `-ValidateOnly` performs local parsing and hash checks without SSH or deployment.
4. `Deploy-G21GoalRuntimeR116Remote.ps1` keeps extraction under `C:\SecureInput` while it performs all read-only production gates. The current Run must be terminal with effective active count zero and pending approval count zero. Therefore a nonterminal Run cannot stop a service, create a production backup, touch the bundle, or touch the database.
5. Only after the read-only gates pass does the remote transaction invoke `C:\SecureInput\Invoke-ServiceAccountAction.ps1 -Action Stop`. It rejects residual listeners instead of killing them, backs up the runtime and the database plus any WAL/SHM sidecars, introduces the candidate into the bundle, swaps the runtime atomically, reseals, and starts through the same wrapper.
6. Goal creation/reuse uses authenticated loopback `POST/GET/PUT`; it never writes SQLite directly. A response-loss branch may recover only by reading back the same fixed Goal identity. The transaction then performs a second wrapper Stop/Start and proves restart persistence, one Goal row, and one `goal.created` event.
7. Any failure after mutation restores the R115 runtime and the stopped database/WAL/SHM snapshot, reseals, starts through the wrapper, and requires loopback/public health plus R115 tree identity.

## Truthful audit fields

Production runtime backup/swap, database backup, reseal, and Goal API persistence are production writes and are reported as such. They are not mislabeled as zero writes. Separate fields report:

- `production_writes_performed` and `production_write_categories`
- `managed_service_actions`
- `hpc_remote_writes=0`
- `direct_signals_sent=0`
- `other_processes_modified=false`
- `hpc_accessed=false` and `gpu_touched=false`

## Attempt 1 event-window failure and closure

Attempt 1 stopped at Goal event acceptance and then reported `rollback_verified`, loopback/public health `3/3`, and the fixed Run restored as `cancelled` with event sequence `3565`. The Goal row itself was not the root cause. The runner queried the public projection at `/v1/runs/{run_id}/events?after=0`; thousands of historical events filled the bounded first page, so the newly appended `goal.created` event was outside the returned window.

The corrected contract is deliberately narrower:

- Capture `runBefore.last_event_seq` while the fixed Run is still in the read-only preflight.
- Pass that same sequence to both the first Goal bootstrap check and the post-restart reuse check.
- Read raw immutable events from `/v1/sessions/{run_id}/events?after=<cursor>`, not the public alias projection.
- Page by the last raw sequence until the post-bootstrap `last_event_seq` upper bound is reached; reject gaps, duplicate/non-increasing sequences, a stalled cursor, or an excessive page count.
- Within that anchored window require exactly one raw `goal.created`, zero `goal.updated`, and bind its goal ID/key/spec/board hashes to the persisted Goal.
- Require the first and post-restart checks to observe the same `goal.created` sequence.

The directed regression fixture creates more than 1,200 historical events before the anchor, creates/replays/PUTs the fixed Goal, restarts the Runtime, and proves the anchored raw window still contains exactly one `goal.created` and no `goal.updated`.

## Current stopping condition

Candidate ZIP SHA `37929a1c9baf682e4f946450f834f10fe3b7a78323084ae96948bd0e92f8de8d` is permanently `NOT_DEPLOYABLE` because its tree-hash path convention differed from production and its strict Goal validator had P1 evidence gaps. The deployer and verifier hard-reject that SHA and bind rejection evidence SHA `8900c32d41a47561715dc215ec2e560e338c2d164d6fa410554b7a2b83b5fb83`.

The latest post-attempt read-only evidence supplied to this implementation reports the fixed Run as `cancelled` at sequence `3565` after verified rollback. Any later attempt must sample the Run again; a future candidate or approval never overrides the terminal/effective-active/pending preflight.
