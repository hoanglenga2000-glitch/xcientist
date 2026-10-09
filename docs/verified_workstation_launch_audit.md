# Verified Workstation Launch Audit

- Generated at: 2026-08-09T16:14:59.7585067Z
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
- Resource blockers allowed: False

## Check Results

- kaggle_dpapi_readiness: exit 0, ok True, sha256 0ad710aa32001509a3dd45244274124fb19bd49c5309a5364bf9a2ab7d7c53ac
- backend_resource_status: exit 0, ok True, sha256 b6d2762cf7e393d2633a0f858400b2a50b283669b5b8df31adbc692c233f4053
- openai_gateway_smoke: exit 0, ok True, sha256 3359994b6cfa6059faa9fa029af93d6f48a2aa1c9a30a97c9ea39001c4f54e60
- external_gateway_smoke: exit 0, ok True, sha256 4fcd4ac560f1b821991a77198501d10643504b45143dfd33afdce13c0a407ace
- kaggle_secret_smoke: exit 0, ok True, sha256 fbc581adc5ab605eb12fc8b47b674d101cfefc426ae431d13796e387d73577a9
- plaintext_secret_scan: exit 0, ok True, sha256 6e054a5c85caccf11a65a24a3988e8ece26c83887eeb3117c7decc7312ffffd2

## Remaining External Conditions


## Security Note

No secret values or raw command output are written to this audit report; only DPAPI presence booleans, command labels, exit codes, hashes and bounded verifier signals are recorded.
