# HPC-only test contract migration (2026-08-06)

The release test suite previously assumed that local training could bypass an unavailable GPU/HPC manifest and that mocked LLM credentials were enough to enter `_run_agent`. That behavior conflicts with the current release invariant: training is allowed only after the named HPC job profile reaches and verifies the intended job container.

The affected tests were restored as active assertions and updated to prove the stricter behavior:

- local compute, resume, auto mode, and switch-then-train remain fail-closed when HPC identity is absent;
- checkpoint, decision, action-queue, loop, and execution-contract artifacts report the blocker without invoking training;
- offline benchmark summaries are untrusted unless their current implementation/source identity and external certification anchors validate;
- AI Scientist parity claims require both the hidden-suite certification and an active immutable upgrade campaign;
- generic chat remains a direct LLM conversation and does not create Scientist planning artifacts.

No case is skipped or marked xfail. The migration preserves read-only diagnostic and memory actions while preventing local or unverified remote execution.
