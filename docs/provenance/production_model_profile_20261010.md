# Production model profile: temporary Claude default (2026-10-10)

Status: ACTIVE, TEMPORARY. Operator decision 2026-10-10 01:20 America/New_York
("可以先用Claude模型做，等以后pe那个恢复了再换，目前默认就是用Claude做好了").

## Why

The Pezayo `deepseek-flash` route (`https://api.pezayo.com/v1`) returned HTTP 402
"Insufficient Balance" on 2026-10-10. The novice assistant quality gate therefore
could not be measured on the committed suite route, and the production assistant
could not answer on that route.

## Recorded exception: novice quality gate

- Committed suite `configs/evaluation/assistant_novice_v1.json` still pins
  `openai` / `deepseek-flash` / Pezayo. It is unchanged.
- The 5-case gate (g3) was run with a LOCAL, UNCOMMITTED override suite pinning
  `anthropic` / `claude-sonnet-5` / `https://api.lt4net.org/v1`
  (`wire_protocol=anthropic_messages`) on source commit `21ce737` plus the
  anthropic-route gate support later committed as `095c879`. Result: 5/5 pass,
  verified by `verify_evomind_assistant_quality.py --suite <override>`.
- Staging acceptance at `095c879`: 17/18 checks pass; the only failure is
  `assistant:novice_quality_gate`, because the g3 report (Claude) does not match
  the committed suite (deepseek-flash). This is the intended identity protection,
  not a code failure. The operator accepted this as a recorded exception.
- When Pezayo balance returns, rerun the gate on the committed suite and replace
  this exception with a clean result.

## Production change (Shanghai node)

- Bundle `scripts/Start-Node.ps1` model block `EVOMIND_GPT55_PROFILE_V1` now carries
  `EVOMIND_MODEL_PROFILE_V2`: an operator-selectable child-process profile.
  - `claude_lt4net_temp` (default): `EVOLUTION_PRIMARY_PROVIDER=anthropic`,
    `LLM_PROVIDER=anthropic`, `ANTHROPIC_BASE_URL=https://api.lt4net.org`
    (transport appends `/v1/messages`), `CLAUDE_CODE_MODEL=ANTHROPIC_MODEL=claude-sonnet-5`,
    strict provider, route file `state/ev-claude-lt4net-route-20261010.json`.
  - `pezayo_deepseek_flash`: the previous profile, byte-for-byte the same env values.
  - The Pezayo `OPENAI_*` / `DEEPSEEK_*` values remain published in both profiles
    (connector projection and rollback).
- Credential: CurrentUser DPAPI file `newapi_lt4net_claude_temp.xml`
  (user `__NEWAPI_LT4NET_CLAUDE_TEMP__`) in the EvoMindSvc secrets root, next to
  `pezayo_ev_deepseek_v4_pro.xml`. The key is not in git, logs, or this document.
- Bundle resealed with `Seal-Release.ps1`; node restarted through the service account.
- Verified: a real runtime turn on `127.0.0.1:8765` recorded
  `model.response provider=anthropic model=claude-sonnet-5`.

## Known limitations while the temporary profile is active

- Anthropic clients are not wrapped by the governed OpenAI transport; new runtime
  sessions are admitted as uncontracted. Runs already bound to a `deepseek-flash`
  execution contract cannot resume under Claude (`model_contract_missing_on_resume`).
- The advisory `model.routed` event may still name `openai:deepseek-flash`; the
  executing provider is reported by `model.response` / `model_execution`.
- Paths that hard-code the OpenAI provider (for example the local-GPU tabular
  workflow and the cliproxy gateway health check) still depend on Pezayo.

## Restore Pezayo deepseek-flash (no reseal required)

1. Top up Pezayo; confirm `/v1/chat/completions` with `deepseek-flash` returns 200.
2. Set `"model_profile": "pezayo_deepseek_flash"` in
   `C:\ProgramData\EvoMind\config\node-config.json`.
3. Restart the node through the service account
   (`operations\restart_evomind_node.ps1`), check `/api/healthz`.
4. Rerun the novice gate on the committed suite.
5. Revoke/rotate the temporary lt4net key and delete `newapi_lt4net_claude_temp.xml`.
