# MLEvolve Gap Analysis: Why Our Medal Rate Is Poor

## MLEvolve (65.3% medal rate) vs Our Workstation (0% confirmed medals)

| Capability | MLEvolve | Our Workstation | Gap |
|-----------|----------|----------------|-----|
| **Search Engine** | Progressive MCGS with UCT | Simple ensemble blending | CRITICAL |
| **Node Selection** | UCT: exploitation + C*sqrt(ln(N)/n) | Random/grid search | CRITICAL |
| **Exploration Control** | Piecewise decay C: 1.414→0.5 | None | HIGH |
| **Branches** | Multiple concurrent (3+ drafts) | Single solution path | CRITICAL |
| **Stagnation Detection** | Branch + Global level | None | HIGH |
| **Cross-Branch Fusion** | Top-K nodes across branches | None | HIGH |
| **Memory** | BGE embeddings + BM25 + FAISS | None | CRITICAL |
| **Cold-Start KB** | 100+ competition patterns, model guidance | 10 basic entries | MEDIUM |
| **Coding Modes** | Base / Diff / Stepwise | Full rewrite only | HIGH |
| **Hierarchical Planning** | Planner→Plan→Coder (3-step) | One-shot generation | HIGH |
| **Parallel Execution** | ThreadPool 3-5 concurrent | Sequential | MEDIUM |
| **Submission Validation** | Format check + data leakage agent | None | HIGH |
| **Search Budget** | 500 steps / 12 hours | 1-5 runs | CRITICAL |
| **Error Recovery** | Debug agent with error trace analysis | None | HIGH |
| **Code Review** | Pre-execution code review agent | None | MEDIUM |

## Root Causes

1. **We train, they search.** MLEvolve performs 500-step search with feedback loops. We run 1 ensemble and submit.

2. **We have no memory.** MLEvolve remembers every attempt and reuses successful patterns. We start fresh each time.

3. **We have no branches.** MLEvolve explores 3+ parallel solution paths simultaneously. We follow one path.

4. **We don't detect stagnation.** When MLEvolve stalls, it triggers cross-branch fusion or aggregation. We just stop.

5. **They use LLM planning.** MLEvolve separates "what to change" from "how to change". Our code is hand-written.

## Immediate Fixes Required

1. Enable iterative search: replace one-shot training with multi-step feedback loop
2. Add branch management: track multiple solution paths
3. Add memory: accumulate experience across runs
4. Add stagnation detection: trigger fusion when stuck
5. Use DeepSeek for planning: separate plan generation from code execution
