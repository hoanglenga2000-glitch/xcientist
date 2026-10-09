# EvoMind Agent UX Release Source Scope

- status: needs_release_source_review
- generated_at: 2026-08-18T02:57:27+0800
- claim_boundary: Source-scope audit only; no training, Kaggle submit, private grader, commit, tag, or push.

## Blockers

- missing: none
- untracked: src/evomind_runtime/assistant_runs.py, src/evomind_runtime/hpc_runtime_overlay.py
- dirty_tracked: scripts/verify_local_evomind_release_gate.py, scripts/verify_live_assistant_demo_smoke.py, scripts/build_agent_ux_release_source_manifest.py, tests/test_verify_local_evomind_release_gate.py, tests/test_verify_live_assistant_demo_smoke.py, configs/evaluation/assistant_novice_v1.json, src/xsci/assistant_behavior_distillation.py, src/xsci/assistant_context.py, src/xsci/assistant_stream.py, src/xsci/config.py, src/xsci/kaggle_conversation.py, src/xsci/kaggle_intent.py, src/xsci/kaggle_session.py, src/xsci/terminal_tools.py, src/xsci/user_request.py, tests/test_assistant_context.py, tests/test_assistant_quality_evaluation.py, tests/test_assistant_stream.py, tests/test_kaggle_conversation_tool_loop.py, src/evomind_runtime/http_server.py, src/evomind_runtime/models.py, src/evomind_runtime/policy.py, src/evomind_runtime/runtime.py, src/evomind_runtime/store.py, src/evomind_runtime/tools.py, tests/test_evomind_runtime.py, tests/test_evomind_runtime_http_server.py, scripts/verify_new_user_release_readiness.py, scripts/verify_backend_resource_status.py, tests/test_backend_resource_status.py, scripts/manage_local_gateway.py, scripts/verify_openai_gateway.py, scripts/manage_hpc_proxy_bridge.ps1, scripts/manage_workstation_dashboard.py, tests/test_dashboard_build_transaction.py, web/research-agent-workstation/src/proxy.ts, web/research-agent-workstation/src/app/api/session/status/route.ts, web/research-agent-workstation/src/app/api/assistant/stream/route.ts, web/research-agent-workstation/src/components/workstation/LocalSessionBootstrap.tsx, web/research-agent-workstation/src/components/workstation/screens/AssistantScreen.tsx, web/research-agent-workstation/src/lib/server/scientific-report.ts, web/research-agent-workstation/src/lib/server/reviewed-existing-report.ts, web/research-agent-workstation/src/lib/server/reviewed-existing-report.test.ts, web/research-agent-workstation/src/lib/server/local-session.ts, web/research-agent-workstation/src/lib/server/local-session-security.test.ts, web/research-agent-workstation/scripts/verify-localhost-security-contract.mjs, web/research-agent-workstation/package.json, web/research-agent-workstation/package-lock.json
- secret_scan: passed

## Files

| Path | Tracked | Dirty | Required Action | Purpose |
|---|---:|---:|---|---|
| scripts/verify_local_evomind_release_gate.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Aggregates local demo and formal source gates for Agent UX release readiness. |
| scripts/verify_live_assistant_demo_smoke.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Runs authenticated four-turn browser-equivalent Agent smoke, including job90673 connection self-check. |
| scripts/build_agent_ux_release_source_manifest.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| tests/test_verify_local_evomind_release_gate.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Prevents release gate source-tracking regressions. |
| tests/test_verify_live_assistant_demo_smoke.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Prevents job90673 connection smoke and side-effect-boundary regressions. |
| tests/test_build_agent_ux_release_source_manifest.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| configs/evaluation/assistant_behavior_board_v1.json | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| configs/evaluation/assistant_novice_v1.json | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/xsci/assistant_behavior_distillation.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Defines novice-facing behavior contracts, visible response audit, and deterministic repairs. |
| src/xsci/assistant_context.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/xsci/assistant_quality_evaluation.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/xsci/assistant_stream.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/xsci/config.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/xsci/kaggle_conversation.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Implements browser/terminal Agent routing, LLM-first tool loop, safe context, and fallback behavior. |
| src/xsci/kaggle_intent.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/xsci/kaggle_session.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/xsci/terminal_agent.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/xsci/terminal_tools.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/xsci/user_request.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/research_os/agent/messaging.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/research_os/llm_client.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| tests/test_agent_messaging.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| tests/test_assistant_behavior_distillation.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| tests/test_assistant_context.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| tests/test_assistant_quality_evaluation.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| tests/test_assistant_stream.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| tests/test_kaggle_conversation_tool_loop.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| tests/test_user_request_protocol.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/evomind_runtime/__init__.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/evomind_runtime/assistant_runs.py | no | no | review_and_git_add_before_release_commit | Release-critical Agent UX source file. |
| src/evomind_runtime/benchmark.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/evomind_runtime/cli.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/evomind_runtime/client.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/evomind_runtime/evolution.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/evomind_runtime/hpc_runtime_overlay.py | no | no | review_and_git_add_before_release_commit | Release-critical Agent UX source file. |
| src/evomind_runtime/http_server.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/evomind_runtime/mcp.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/evomind_runtime/model_router.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| src/evomind_runtime/models.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/evomind_runtime/policy.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/evomind_runtime/runtime.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/evomind_runtime/store.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| src/evomind_runtime/tools.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| tests/test_evomind_runtime.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| tests/test_evomind_runtime_http_server.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| scripts/verify_new_user_release_readiness.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Verifies new-user local gateway readiness and novice quality gate freshness. |
| scripts/verify_backend_resource_status.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Checks the canonical Connector Registry state and raw connector provenance. |
| tests/test_backend_resource_status.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Prevents canonical connector-state and authoritative GPU-gate regressions. |
| scripts/run_authenticated_web_contract.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| scripts/verify_web_isolated_build.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| tests/test_verify_web_isolated_build.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| scripts/manage_local_gateway.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| scripts/verify_openai_gateway.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| scripts/manage_hpc_proxy_bridge.ps1 | yes | yes | review_and_commit_or_revert_before_strict_gate | Starts the local HPC SOCKS bridge through the designated upstream proxy by default. |
| scripts/hpc_socks_bridge.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| configs/hpc_connection_memory_core.json | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| docs/HPC_CONNECTION_MEMORY_CORE.md | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| tests/test_hpc_connection_memory_core.py | yes | no | already_tracked_clean | Prevents direct-route defaults and HPC proxy/container identity contract regressions. |
| tests/test_hpc_socks_bridge.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| tests/test_benchmark_openai_gateway_profiles.py | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| scripts/manage_workstation_dashboard.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Binds the novice-facing dashboard to the installed interactive OpenAI performance profile. |
| tests/test_dashboard_build_transaction.py | yes | yes | review_and_commit_or_revert_before_strict_gate | Prevents dashboard lifecycle, build transaction, and interactive gateway profile regressions. |
| web/research-agent-workstation/src/proxy.ts | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| web/research-agent-workstation/src/app/layout.tsx | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| web/research-agent-workstation/src/app/api/session/bootstrap/route.ts | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| web/research-agent-workstation/src/app/api/session/status/route.ts | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| web/research-agent-workstation/src/app/api/assistant/stream/route.ts | yes | yes | review_and_commit_or_revert_before_strict_gate | Authenticates and streams browser assistant requests through the Python Agent subprocess. |
| web/research-agent-workstation/src/components/workstation/LocalSessionBootstrap.tsx | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| web/research-agent-workstation/src/components/workstation/screens/AssistantScreen.tsx | yes | yes | review_and_commit_or_revert_before_strict_gate | Browser-facing chat UI, stream error handling, and visible Agent status. |
| web/research-agent-workstation/src/lib/server/scientific-report.ts | yes | yes | review_and_commit_or_revert_before_strict_gate | Projects reviewed Run evidence into the Report Studio without bypassing review or hash gates. |
| web/research-agent-workstation/src/lib/server/reviewed-existing-report.ts | yes | yes | review_and_commit_or_revert_before_strict_gate | Fails closed unless a completed Run has task binding, Reviewer, Claim Audit, and artifact provenance. |
| web/research-agent-workstation/src/lib/server/reviewed-existing-report.test.ts | yes | yes | review_and_commit_or_revert_before_strict_gate | Prevents reviewed-report task, gate, and manifest binding regressions. |
| web/research-agent-workstation/src/lib/server/local-session.ts | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| web/research-agent-workstation/src/lib/server/paths.ts | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| web/research-agent-workstation/src/lib/server/local-session-security.test.ts | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| web/research-agent-workstation/src/lib/server/loopback-session.test.ts | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| web/research-agent-workstation/scripts/lib/loopback-session.mjs | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| web/research-agent-workstation/scripts/verify-localhost-security-contract.mjs | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| web/research-agent-workstation/scripts/verify-report-studio-contract.mjs | yes | no | already_tracked_clean | Release-critical Agent UX source file. |
| web/research-agent-workstation/package.json | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
| web/research-agent-workstation/package-lock.json | yes | yes | review_and_commit_or_revert_before_strict_gate | Release-critical Agent UX source file. |
