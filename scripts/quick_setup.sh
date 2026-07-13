#!/usr/bin/env bash
# quick_setup.sh — AI Research Workstation one-command setup
# Run this from the project root. Works on Git Bash (Windows), macOS, Linux.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() { echo -e "${CYAN}============================================================${NC}"; }
ok()     { echo -e "  ${GREEN}[✓]${NC} $1"; }
warn()   { echo -e "  ${YELLOW}[!]${NC} $1"; }
fail()   { echo -e "  ${RED}[✗]${NC} $1"; }
step()   { echo -e "\n${CYAN}>>> $1${NC}"; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$ROOT/web/research-agent-workstation"

banner
echo "AI Research Workstation — Quick Setup"
echo "Root: $ROOT"
banner

# ── 1. Python ──────────────────────────────────────────────
step "Step 1/6: Python environment"
PYTHON=$(command -v python3 || command -v python || echo "")
if [ -z "$PYTHON" ]; then
    fail "Python not found. Install Python 3.10+ from https://python.org"
    exit 1
fi
PYVER=$("$PYTHON" --version 2>&1)
ok "Found $PYVER at $PYTHON"

step "   Installing Python dependencies..."
"$PYTHON" -m pip install -e "$ROOT" --quiet 2>&1 && ok "pip install -e ." || warn "pip install -e . had warnings (may be fine)"

# ── 2. Node ────────────────────────────────────────────────
step "Step 2/6: Node.js frontend"
NODE=$(command -v node || echo "")
if [ -z "$NODE" ]; then
    fail "Node.js not found. Install Node 18+ from https://nodejs.org"
    exit 1
fi
ok "Found Node $(node --version)"

if [ -d "$WEB/node_modules" ]; then
    ok "node_modules exists, skipping npm install"
else
    echo "   Running npm install (this may take a minute)..."
    (cd "$WEB" && npm install --silent 2>&1) && ok "npm install" || fail "npm install failed"
fi

step "   Initializing local workstation database..."
WEB_ENV="$WEB/.env"
if [ ! -f "$WEB_ENV" ]; then
    NORMALIZED_ROOT=${ROOT//\\//}
    printf 'DATABASE_URL="file:./workstation.db"\nWORKSTATION_ROOT="%s"\nNEXT_TELEMETRY_DISABLED=1\n' "$NORMALIZED_ROOT" > "$WEB_ENV"
    ok "Created non-secret web .env"
fi

DATABASE_URL_VALUE=${DATABASE_URL:-}
if [ -z "$DATABASE_URL_VALUE" ]; then
    DATABASE_URL_VALUE=$(sed -n 's/^[[:space:]]*DATABASE_URL[[:space:]]*=[[:space:]]*//p' "$WEB_ENV" | head -n 1)
    DATABASE_URL_VALUE=${DATABASE_URL_VALUE#\"}
    DATABASE_URL_VALUE=${DATABASE_URL_VALUE%\"}
    DATABASE_URL_VALUE=${DATABASE_URL_VALUE#\'}
    DATABASE_URL_VALUE=${DATABASE_URL_VALUE%\'}
fi
if [ -z "$DATABASE_URL_VALUE" ]; then
    fail "DATABASE_URL is missing from web .env"
    exit 1
fi
if [ "$DATABASE_URL_VALUE" = "file:./prisma/workstation.db" ]; then
    DATABASE_URL_VALUE="file:./workstation.db"
    warn "Normalized legacy SQLite path"
fi
export DATABASE_URL="$DATABASE_URL_VALUE"
case "$DATABASE_URL_VALUE" in
    file:*)
        (cd "$WEB" && npm run db:push -- --skip-generate 2>&1) && ok "Local SQLite schema is ready" || {
            fail "Local SQLite schema initialization failed"
            exit 1
        }
        ;;
    *)
        warn "External database configured; automatic schema push skipped"
        ;;
esac

step "   Generating Prisma client..."
(cd "$WEB" && npm run db:generate 2>&1) && ok "Prisma client generated" || fail "Prisma client generation failed"

# ── 3. Frontend build ──────────────────────────────────────
step "Step 3/6: Build frontend"
(cd "$WEB" && npm run build 2>&1) && ok "npm run build" || fail "Build failed"

# ── 4. CLI wrappers ────────────────────────────────────────
step "Step 4/6: Install CLI commands (kaggle/autokaggle)"
SHIM="$HOME/.xsci/bin"
mkdir -p "$SHIM"

# Bash wrappers (Git Bash / macOS / Linux)
for name in kaggle autokaggle kaggle-official; do
    cat > "$SHIM/$name" << 'SCRIPT'
#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
case "$(basename "$0")" in
    kaggle-official) exec python -X utf8 -m xsci.kaggle official "$@" ;;
    *)              exec python -X utf8 -m xsci.kaggle "$@" ;;
esac
SCRIPT
    chmod +x "$SHIM/$name" 2>/dev/null || true
done
ok "Created bash wrappers in $SHIM"

# .cmd wrappers (Windows PowerShell / cmd.exe)
for name in autokaggle kaggle; do
    cat > "$SHIM/$name.cmd" << 'CMD'
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle %*
CMD
done
cat > "$SHIM/kaggle-official.cmd" << 'CMD'
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle official %*
CMD
ok "Created .cmd wrappers in $SHIM"

# Ensure SHIM is in PATH for this session
case ":$PATH:" in
    *:"$SHIM":*) ;;
    *) export PATH="$SHIM:$PATH" ;;
esac

# ── 5. .env file ───────────────────────────────────────────
step "Step 5/6: Configuration"
if [ -f "$ROOT/.env" ]; then
    warn "Legacy project .env detected; migrate secrets with 'xsci login' and remove secret entries"
else
    ok "No project secret file created"
fi

# ── 6. Verification ────────────────────────────────────────
step "Step 6/6: Verify installation"
"$PYTHON" -c "import xsci; print('xsci OK')" 2>&1 && ok "Python: xsci module" || fail "xsci module"
"$PYTHON" -m py_compile "$ROOT/src/xsci/kaggle.py" "$ROOT/src/xsci/config.py" 2>&1 && ok "Python: compile check" || fail "compile"

if command -v kaggle &>/dev/null; then
    ok "CLI: kaggle command found"
else
    warn "CLI: kaggle not in PATH yet. Restart terminal or run: export PATH=\"$SHIM:\$PATH\""
fi

# ── Summary ────────────────────────────────────────────────
banner
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Configure DeepSeek API key:"
echo "     xsci login"
echo "     (hidden prompt; Windows uses DPAPI, other platforms use a 0600 user file)"
echo ""
echo "  2. Start the workstation:"
echo "     powershell -File scripts/start_verified_workstation.ps1 restart"
echo "     or: cd web/research-agent-workstation && npm run dev"
echo ""
echo "  3. Open dashboard:"
echo "     http://127.0.0.1:8088/?page=control"
echo ""
echo "  4. Run first training:"
echo "     kaggle run titanic"
echo ""
echo "Full guide: docs/NEW_USER_ONBOARDING_GUIDE.md"
banner
