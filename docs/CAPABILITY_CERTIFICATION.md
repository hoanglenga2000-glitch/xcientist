# EvoMind External Capability Certification Contract

This contract defines the evidence required before EvoMind may report
`research parity certified`, `Claude-level`, or `Codex-level`.

## Trust anchors

The release repository must configure these independently from the evidence
bundle:

- secret `EVOMIND_CAPABILITY_EVIDENCE_URL`
- variable `EVOMIND_CAPABILITY_BUNDLE_SHA256`
- variable `EVOMIND_CAPABILITY_REPORT_SHA256`
- variable `EVOMIND_CAPABILITY_SUITE_ID`
- variable `EVOMIND_CAPABILITY_SUITE_SHA256`
- variable `EVOMIND_CAPABILITY_EVALUATOR_ID`
- variable `EVOMIND_CAPABILITY_EVALUATOR_SHA256`
- variable `EVOMIND_CAPABILITY_BASELINE_AGENTS`

All SHA-256 values are 64 lowercase hexadecimal characters. Baseline names are
comma-separated and must include independently evaluated Codex and Claude Code
agents.

## Bundle layout

The hash-pinned ZIP has `report.json` at its root. Every referenced artifact is
a regular file below that root; absolute paths, traversal, links, duplicate
case-folded names, encrypted members, and unsafe compression ratios are
rejected.

Required artifact roles:

1. `wheel`
2. `sdist`
3. `source_archive`
4. `benchmark_raw_results`
5. `self_upgrade_campaign`

Each artifact record contains exactly `role`, `path`, `sha256`, and
`size_bytes`. The verifier reads the bytes and recomputes both size and digest.

## Raw trial JSONL

`benchmark_raw_results` is UTF-8 JSONL. Every non-empty line is one strict JSON
object with exactly these fields:

```json
{
  "schema": "evomind.capability_raw_trial.v1",
  "task_id": "hidden-task-id",
  "domain": "research-domain",
  "repeat": 1,
  "agent_name": "EvoMind",
  "outcome": "passed",
  "scope_violation": false,
  "unsupported_claim": false
}
```

Rules:

- `outcome` is `passed`, `failed`, or `timeout`; timeout is always a failure.
- Every declared task/domain/repeat/agent combination occurs exactly once.
- Task IDs map to one domain, and domain counts equal the report declaration.
- Agent names match the candidate and every reported baseline.
- Duplicate JSON keys, NaN/Infinity, duplicate trials, blank records, unknown
  agents/domains, and incomplete matrices are rejected.
- The verifier independently recomputes attempts, successes, failures,
  timeouts, scope violations, unsupported claims, and every candidate/baseline
  paired contingency table from these rows.

The release policy requires at least 100 hidden tasks, 8 domains, 3 tasks per
domain, and 3 repeats. All trials must be reported. Candidate success rate must
be at least `0.80`, its 95% Wilson lower bound at least `0.75`, paired
non-inferiority must pass at margin `0.05`, and timeout/scope/unsupported-claim
rates must be zero.

## Exact-source and campaign binding

The report binds the Git commit, tree, committed-source manifest digest, and
workstation source ZIP digest. The worktree must be clean. The externally
verified self-upgrade campaign must show:

- a frozen evaluator locked before candidate generation;
- at least two immutable candidate commits;
- promotion of a strict improvement only;
- explicit human approval;
- runtime-tree canary verification;
- CAS promotion and successful rollback proof.

The certification result carries the verified artifact digests. Runtime parity
opens only when the active champion commit/tree and campaign evidence file match
those external bindings byte-for-byte.

## Verification commands

The tag workflow runs the verifier automatically. A local evaluator fixture can
invoke the same gate:

```powershell
python scripts\extract_capability_evidence_bundle.py `
  .\evidence.zip .\evidence --expected-sha256 <bundle-sha256>

python scripts\verify_capability_certification.py `
  .\evidence\report.json `
  --repo-root . `
  --artifact-root .\evidence `
  --expected-report-sha256 <report-sha256> `
  --expected-suite-id <suite-id> `
  --expected-suite-sha256 <suite-sha256> `
  --expected-evaluator-id <evaluator-id> `
  --expected-evaluator-sha256 <evaluator-sha256> `
  --baseline-agent "Codex" `
  --baseline-agent "Claude Code" `
  --output .\capability-certification-result.json
```

A `PASS` is source-specific and expires. Copying a result to another commit,
changing any source byte, replacing the active campaign, or omitting the
out-of-band result digest closes the gate.
