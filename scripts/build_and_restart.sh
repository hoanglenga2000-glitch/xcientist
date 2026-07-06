#!/usr/bin/env bash
# build_and_restart.sh - Safe Next.js build → restart flow
# Prevents the "Cannot find module './8948.js'" stale-module bug
# by always restarting the dev server after a build.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$ROOT/web/research-agent-workstation"

echo "=== Build Next.js ==="
cd "$WEB"
npm run build

echo ""
echo "=== Restart frontend on port 8088 ==="
cd "$ROOT"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restart_workstation_frontend.ps1 -Port 8088

echo ""
echo "=== Done. Frontend should be live at http://127.0.0.1:8088 ==="
