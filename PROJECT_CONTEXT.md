# EvoMind Project Context

> This is the compact project-local working memory. Repository code, tests, current runtime evidence, and authoritative artifacts always take precedence. Historical notes are navigation aids, not proof of current state.

## Metadata

- project_id: `evomind`
- context_status: `historical`
- captured_at: `2026-08-22T14:10:23+08:00`
- last_verified: `2026-08-22T14:10:23+08:00`
- verification_scope: repository structure, Git metadata, existing instruction/document entrypoints only
- excluded_from_verification: current UI/API health, running processes, HPC allocation/container, GPU state, training jobs, Kaggle authorization, release readiness

## Current objective

Maintain an auditable EvoMind / XCIENTIST research workstation while preserving strict evidence, credential, external-system, and human-approval boundaries. Use this file to recover task context quickly; use live evidence to decide whether a requested operation is currently safe or complete.

## Architecture and key entrypoints

- Root `README.md`: product overview, CLI entrypoints, setup, and verification commands.
- `src/evomind_runtime/`: EvoMind runtime, policies, tools, models, and HTTP surface.
- `src/research_agent_workstation/`: workstation application and backend services.
- `src/research_os/`: research workflow, evolution, benchmark, GPU/HPC, and evidence logic.
- `src/xsci/`: terminal agent, assistant, project, memory, and workflow interfaces.
- `web/research-agent-workstation/`: Next.js workstation UI and API routes.
- `scripts/`: bounded setup, lifecycle, verification, training, evidence, and remote-operation helpers.

These are structural pointers only. Their current behavior must be checked in the relevant code and tests before use.

## Verified facts

- On 2026-08-22, the Git root resolved to `D:\桌面\codex\科研港科技` on branch `master` tracking `origin/master`.
- Before DevVault integration, the tracked working tree had 306 status entries. The complete bounded status and pre-change metadata are stored under `D:\AI-Outputs\Codex\DevVault\_system\audit\evomind\`.
- Before DevVault integration, `AGENTS.md` SHA-256 was `46d284887c5b2347dcb6d9bf2484ec834e35e2eb2e921f3a4948a295bff30f41`.
- No UI, API, process, port, GPU, HPC, training, Kaggle, artifact, or release claim was re-verified while creating this context file.

## Dirty worktree boundary

- The repository already contained extensive user changes before this integration.
- Do not reset, checkout, clean, bulk-format, commit, push, or otherwise rewrite unrelated changes without explicit authorization.
- DevVault integration is limited to appending the marked project-memory block in `AGENTS.md` and adding this `PROJECT_CONTEXT.md`.
- Always compare against the preflight audit before attributing an existing change to DevVault work.

## Security and external-system boundaries

- Read and follow `docs/HPC_CONNECTION_MEMORY_CORE.md` and `configs/hpc_connection_memory_core.json` before any GPU/HPC action.
- A proxy, gateway, reachable host, or historical allocation is not proof of a ready HPC job container.
- Credentials must stay in approved protected stores and must not be copied into this file, DevVault, prompts, logs, or source.
- Kaggle download/submission, remote writes, training, deployment, release publication, and commits retain their existing human and authorization gates.
- Historical memory, README text, configuration, health-only checks, or a prior successful run cannot substitute for current task-specific runtime evidence.

## Current blockers

- Current external service, HPC allocation, GPU/container, Kaggle authorization, and production readiness are `unverified` in this context snapshot.
- The heavily dirty worktree makes broad automated edits and attribution unsafe without a scoped baseline and explicit file allowlist.
- Any task claiming completion must define and execute its own current acceptance evidence.

## Recent decision index

- DevVault architecture decision: repository and live evidence remain authoritative; Obsidian stores curated, source-bound summaries.
- Long-term project memory updates require the explicit phrase “收尾并更新项目记忆”.
- DevVault project snapshots use bounded Git metadata and do not imply runtime verification.

Detailed ADRs belong in `D:\AI-Outputs\Codex\DevVault\20-Decisions\` when a future task makes an architecture decision that needs durable review.

## Next three actions

1. At the start of each task, state the requested outcome and identify which facts can drift; verify those facts from current code/runtime evidence.
2. Search DevVault for relevant decisions and worklogs, then open only the source documents needed for the task.
3. When the user explicitly requests memory closeout, update verified facts, blockers, next actions, a dated worklog, and any required ADR; show the diff before commit or push.

## Detailed source documents

- `AGENTS.md`
- `README.md`
- `docs/HPC_CONNECTION_MEMORY_CORE.md`
- `configs/hpc_connection_memory_core.json`
- `docs/RELEASE_CHECKLIST.md`
- `docs/verified_workstation_launch_audit.md`
- `docs/production_readiness_audit_20260805.md`

Do not read every document by default. Select sources according to the current task, and re-verify drift-prone claims.
