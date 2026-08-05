# Security Policy

## Supported Versions

Only the latest `0.2.x` release receives security fixes.

## Reporting a Vulnerability

Report vulnerabilities through the repository's private security advisory form:

https://github.com/hoanglenga2000-glitch/xcientist/security/advisories/new

Do not include credentials, private datasets, or exploit details in a public issue.
Include the affected version, reproduction steps, impact, and any suggested fix.

## Network Boundary

The workstation control plane is a local-only service. The supplied npm scripts
and Docker Compose configuration publish it on `127.0.0.1` only. Do not expose
ports 8088 or 3090 to a LAN or the public internet.

Remote deployment is not supported by the default configuration. It requires a
separate authenticated reverse proxy, TLS, authorization for sensitive actions,
and cross-origin request protection before any port is exposed beyond loopback.

## Release Verification

Release assets are accompanied by SHA256 checksums and GitHub build provenance.
Verify both before installation:

```powershell
gh attestation verify <asset> --repo hoanglenga2000-glitch/xcientist
Get-FileHash -Algorithm SHA256 <asset>
```
