# EvoMind Engineering Guidance

## Source Of Truth

- Inspect the current checkout, tests, runtime logs, and generated evidence before acting.
- Treat historical reports and prompts as snapshots, never as proof of current readiness.
- Do not claim provider, Kaggle, GPU, or release readiness without a current live check.

## Change Discipline

- Keep changes scoped and preserve existing public contracts unless a verified security or correctness issue requires a migration.
- Run the relevant focused tests first, then the complete release gates for shared behavior.
- Never bypass a failed gate, reinterpret `SKIP` as `PASS`, or use local compute when the configured policy requires HPC/GPU.

## Credentials And External Resources

- Store secrets with `evomind setup`, the Windows DPAPI managers, or protected `*_FILE` paths outside the repository.
- Never create a project secret `.env`, print secret values, or pass a secret as a command-line argument.
- HPC operations require the explicit `EVOMIND_HPC_*` contract and strict host-key verification.
- External submissions and patch application remain behind their Human/Manual Gates.

## Release Evidence

- A build is not a release. Verify clean installation, production startup, live API/UI behavior, audits, SBOMs, artifact hashes, and the published assets.
- Default release configuration must remain `Not Configured` or `Auth Pending` until the receiving user's runtime proves otherwise.
