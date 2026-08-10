# EvoMind v0.3.0 Formal Release Authorization

- Date: 2026-08-10 (Asia/Shanghai)
- Status: **AUTHORIZED**
- Operator: 景浩伟
- Repository: `hoanglenga2000-glitch/xcientist`
- Base branch: `master`
- Release branch: `codex/evomind-production-closure-20260810`
- Release tag: `v0.3.0`
- Remote: `origin`

## Authorization statement

The operator explicitly authorized the release operation with the instruction:

> 我授权你操作确保发布正式上线

This authorization covers the formal release commit, the dedicated release branch, a clean checkout from that commit, the protected `v0.3.0` tag, and pushing the branch and tag to `origin`.

## Release invariants

- Preserve the existing mixed `master` worktree and its 35-file staged set.
- Build the release commit from the reviewed alternate-index candidate rather than staging the mixed worktree.
- Keep the release commit parent bound to the preflight-verified `origin/master` baseline.
- Run clean-checkout Python, Web, dependency, runtime, secret, and integrity gates before publishing the protected tag.
- Do not connect to real HPC, start external training, approve a production Gate, or submit to Kaggle as part of release verification.
- Verify the remote release branch and peeled annotated tag resolve to the exact formal commit after push.
- Treat tag-triggered GitHub Actions and protected-environment gates as authoritative online publication evidence.

## Rollback boundary

The release branch and tag are isolated from `master`. If a pre-push verification fails, neither ref is published. If an online protected gate fails after the tag is pushed, preserve the immutable failure evidence and issue a new corrective release rather than rewriting the published tag.
