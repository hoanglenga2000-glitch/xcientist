# `@evomind-ai/cli`

Windows x64 bootstrapper for a local-first EvoMind installation. The package
downloads a versioned Windows bundle, verifies its Ed25519 release manifest and
SHA-256, runs the signed bundle installer against the centralized user-data
layout, starts and verifies both dashboard and runtime health, and only then
switches `%LOCALAPPDATA%\EvoMind\app\current.json` atomically. A failed install
or upgrade restores the SQLite snapshot and previous version pointer.

```powershell
npm install -g @evomind-ai/cli
evomind install
evomind open
evomind doctor --json
```

`EVOMIND_MANIFEST_URL` or `--manifest` can select a different signed release
channel. The Ed25519 trust root remains pinned inside the npm package; release
`key_id` and `min_bootstrap_version` are authenticated manifest fields. User
workspaces, logs, backups and credentials remain under the local user profile.
