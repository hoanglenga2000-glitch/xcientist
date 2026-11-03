#!/usr/bin/env bash
# EvoMind GPU memory broker dead-man guard (t238 C7c-B). Uploaded + launched by
# src/evomind_runtime/gpu_memory_broker.py; operator doc: docs/GPU_MEMORY_BROKER_20261101.md.
# When the EvoMind host stops heart-beating (or max_down is hit) it kills ONLY EvoMind trainer
# processes under ROOT, restarts the owner's services with their original start scripts in
# start order, waits for health and records the result in lease.json + guard.log.
# Touches nothing else. Holds no secrets. Never retries in a loop.
set -u
ROOT=/hpc2hdd/home/aimslab/jinghw
DIR="$ROOT/evomind/gpu_broker"
LEASE="$DIR/lease.json"
LOCK="$DIR/restore.lock"
LOG="$DIR/guard.log"
PIDF="$DIR/guard.pid"
POLL=15

now() { date +%s; }
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) guard[$$] $*" >> "$LOG"; }
num() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; return 0; }

field() {
  sed -n "s/^  \"$1\": \"\{0,1\}\([A-Za-z0-9_]*\)\"\{0,1\},\{0,1\}\$/\1/p" "$LEASE" 2>/dev/null | head -n 1
}

items() {
  awk -v k="$1" '
    $0 == "  \"" k "\": [" { on = 1; next }
    on && /^  \]/ { exit }
    on { s = $0; sub(/^ *"/, "", s); sub(/",?$/, "", s); gsub(/\\"/, "\"", s); print s }
  ' "$LEASE" 2>/dev/null
}

set_state() {
  sed "s/^  \"state\": \"[a-z_]*\"/  \"state\": \"$1\"/" "$LEASE" > "$LEASE.g$$" && mv -f "$LEASE.g$$" "$LEASE"
}

safe_script() {
  case "$1" in "$ROOT"/*.sh) ;; *) return 1 ;; esac
  case "$1" in *..*|*//*|*[!A-Za-z0-9._/-]*) return 1 ;; esac
  return 0
}

run_script() {
  s=${1#bash }
  if [ "bash $s" != "$1" ] || ! safe_script "$s"; then log "refused command: $1"; return 1; fi
  if [ -n "${2:-}" ] && [ "$(sha256sum -- "$s" 2>/dev/null | cut -d' ' -f1)" != "$2" ]; then
    log "refused changed script: $s"; return 1
  fi
  bash "$s" </dev/null >/dev/null 2>&1
}

healthy() {
  url=${1%% *}; want=${1#* }
  if [ -z "$want" ] || [ "$want" = "$1" ]; then return 1; fi
  case "$url" in http://127.0.0.1:[0-9]*) ;; *) return 1 ;; esac
  case "$url" in *..*|*[!A-Za-z0-9.:/_-]*) return 1 ;; esac
  case "$want" in
    *[!0-9]*)
      body=$(curl -s -m 5 "$url" 2>/dev/null | tr -d ' \t\r\n')
      case "$body" in *"$want"*) return 0 ;; esac
      return 1 ;;
    *) [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)" = "$want" ] ;;
  esac
}

trainer_pids() {
  me=$(id -u)
  items kill_patterns | while IFS= read -r pat; do
    case "$pat" in *managed_tensor_train*|*evomind_gpu_trainer*) ;; *) continue ;; esac
    case "$pat" in *..*|*[!A-Za-z0-9._/-]*) continue ;; esac
    pgrep -u "$me" -f -- "$pat" 2>/dev/null
  done | sort -u | while IFS= read -r p; do
    [ "$p" = "$$" ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in *"$ROOT/"*) echo "$p" ;; esac
  done
}

kill_trainers() {
  for p in $(trainer_pids); do log "TERM trainer pid $p"; kill -TERM "$p" 2>/dev/null; done
  sleep 10
  for p in $(trainer_pids); do log "KILL trainer pid $p"; kill -KILL "$p" 2>/dev/null; done
}

take_lock() {
  mkdir "$LOCK" 2>/dev/null && return 0
  m=$(stat -c %Y "$LOCK" 2>/dev/null) || m=0
  if [ $(( $1 - m )) -gt "$2" ]; then
    log "stale restore lock (age $(( $1 - m ))s), taking over"
    rmdir "$LOCK" 2>/dev/null
    mkdir "$LOCK" 2>/dev/null && return 0
  fi
  return 1
}

fire() {
  set_state restoring
  log "deadline passed (state=$1): forced restore"
  kill_trainers
  ht=$(field health_timeout_s); num "$ht" || ht=900
  ok=1; i=0
  while IFS= read -r cmd <&3; do
    [ -n "$cmd" ] || continue
    log "pre-stop: $cmd"; run_script "$cmd"
  done 3<<EOF
$(items pre_stop)
EOF
  while IFS= read -r cmd <&3; do
    [ -n "$cmd" ] || continue
    i=$((i + 1)); h=$(items health | sed -n "${i}p")
    if healthy "$h"; then log "already healthy: $cmd"; continue; fi
    log "start: $cmd"
    run_script "$cmd" "$(items restore_sha256 | sed -n "${i}p")" || { ok=0; continue; }
    waited=0
    until healthy "$h"; do
      if [ "$waited" -ge "$ht" ]; then log "unhealthy after ${ht}s: $cmd"; ok=0; break; fi
      sleep "$POLL"; waited=$((waited + POLL))
    done
  done 3<<EOF
$(items restore_order)
EOF
  if [ "$ok" = 1 ]; then
    set_state restored; log "restored"
  else
    set_state restore_failed; log "restore_failed: operator action needed (docs/GPU_MEMORY_BROKER_20261101.md)"
  fi
}

if [ -f "$PIDF" ]; then
  old=$(cat "$PIDF" 2>/dev/null)
  if num "$old" && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null; then
    case "$(tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null)" in *gpu_broker/guard.sh*) exit 0 ;; esac
  fi
fi
echo "$$" > "$PIDF"
log "start"
while :; do
  [ -f "$LEASE" ] || { log "no lease: exit"; break; }
  st=$(field state); dl=$(field deadline_epoch); hx=$(field hard_exit_epoch); bud=$(field restore_budget_s)
  case "$st" in released|restored|restore_failed) log "lease $st: exit"; break ;; esac
  if ! num "$dl" || ! num "$hx" || ! num "$bud"; then log "unreadable lease: exit"; break; fi
  t=$(now)
  if [ "$t" -gt "$hx" ]; then log "hard exit time passed: exit"; break; fi
  if [ "$t" -gt "$dl" ] && { [ "$st" = active ] || [ "$st" = broker_restoring ]; }; then
    if take_lock "$t" "$bud"; then fire "$st"; break; fi
  fi
  sleep "$POLL"
done
[ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"
exit 0
