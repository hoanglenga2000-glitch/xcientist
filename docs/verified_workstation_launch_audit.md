# Verified Workstation Launch Audit

- Generated at: 2026-07-25T01:28:38
- Status: passed
- Dashboard: http://127.0.0.1:8088
- Active LLM: openai / gpt-5.6-sol
- OpenAI base URL: http://127.0.0.1:65068/v1
- OpenAI streaming enabled: True
- OpenAI tool calling enabled: True
- OpenAI DPAPI: True
- DeepSeek DPAPI: True
- Claude DPAPI: True
- Kaggle DPAPI: True
- HPC SSH DPAPI: True
- Real external calls: False
- Resource blockers allowed: True

## Check Results

- backend_resource_status: exit 0, ok True, sha256 46c8b77cf4f7d78ec6b6c56115029a8c23e0eda523dfd1b936092f0500deed69
- openai_gateway_smoke: exit 0, ok True, sha256 3f4908ba659527689842786b9553c4ed10fa372243ba1c0f4751dd05320a811a
- external_gateway_smoke: exit 0, ok True, sha256 d61fe34726c660fffd71ae85c4c98ea0648214fb585844f8465cd92e13a1b98e
- kaggle_secret_smoke: exit 0, ok True, sha256 51a0e0af8dabc3a7d9a042efa387aef63972ca34cc74220000d55479f8677990
- plaintext_secret_scan: exit 0, ok True, sha256 90dadd9c3ea73432fc50b9573b62c791191a9b5a53a884cc39cb75d6398e5ba3

## Remaining External Conditions


## Security Note

No secret values are written to this audit report; only DPAPI presence booleans, command labels, exit codes, hashes and short verifier excerpts are recorded.
