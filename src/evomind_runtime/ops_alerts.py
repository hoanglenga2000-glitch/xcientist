"""t238 C7e O3: threshold alerts with dedupe/cooldown and pluggable sinks.

Default sinks are local only: an append-only JSONL log and an atomically replaced JSON status
file that PowerShell reads with ``Get-Content alert_status.json | ConvertFrom-Json``. A webhook
or the Windows event log is used only when the operator configures it (see
``docs/OPS_METRICS_ALERTS_C7E.md``); nothing is sent off the host by default.

Timing: ``OpsMonitor`` ticks every ``EVOMIND_OPS_TICK_SECONDS`` (default 30 s). A critical rule
uses ``for_seconds`` <= 60, so a fault that persists is alerted within 30 + 60 + 30 = 120 s.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_COOLDOWN_SECONDS = 900.0
DEFAULT_TICK_SECONDS = 30.0
SEVERITY_ORDER = {"ok": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class Rule:
    name: str
    metric: str
    op: str  # ">", "<", ">=", "<=", "==", "increase"
    threshold: float
    severity: str  # "warning" | "critical"
    for_seconds: float = 0.0
    summary: str = ""

    def breached(self, value: Any, previous: Any) -> bool:
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if self.op == "increase":
            return previous is not None and value - previous > self.threshold
        return {">": value > self.threshold, "<": value < self.threshold, ">=": value >= self.threshold,
                "<=": value <= self.threshold, "==": value == self.threshold}[self.op]


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("disk_free_low", "disk_free_pct", "<", 15.0, "warning", 0, "磁盘剩余空间低于 15%"),
    Rule("disk_free_critical", "disk_free_pct", "<", 8.0, "critical", 0, "磁盘剩余空间低于 8%"),
    Rule("hpc_probe_down", "hpc_probe_ok", "==", 0, "critical", 60, "算力集群探测连续失败"),
    Rule("llm_error_rate_high", "llm_error_rate", ">", 0.2, "warning", 60, "模型调用失败率超过 20%"),
    Rule("llm_error_rate_critical", "llm_error_rate", ">=", 0.5, "critical", 60, "模型调用失败率达到 50%"),
    Rule("api_p95_slow", "api_p95", ">", 5.0, "warning", 120, "接口 p95 延迟超过 5 秒"),
    Rule("run_stalled", "run_stalled_total", "increase", 0, "warning", 0, "有任务卡住（超过 3 分钟无心跳）"),
    Rule("queue_wait_long", "queue_wait_seconds", ">", 1800.0, "warning", 0, "排队等待超过 30 分钟"),
)


def all_default_rules() -> tuple[Rule, ...]:
    """Service rules plus the O4 capacity watermarks."""
    from .ops_capacity import CAPACITY_RULES
    return DEFAULT_RULES + CAPACITY_RULES


def load_rules(path: str | os.PathLike | None = None) -> tuple[Rule, ...]:
    """Default rules, replaced per name by entries in ``EVOMIND_ALERT_RULES_FILE`` (JSON list)."""
    path = path or os.environ.get("EVOMIND_ALERT_RULES_FILE") or ""
    rules = {rule.name: rule for rule in all_default_rules()}
    if path:
        try:
            for item in json.loads(Path(path).read_text(encoding="utf-8")):
                rule = Rule(str(item["name"]), str(item["metric"]), str(item["op"]), float(item["threshold"]),
                            str(item.get("severity", "warning")), float(item.get("for_seconds", 0)), str(item.get("summary", "")))
                if rule.op not in {">", "<", ">=", "<=", "==", "increase"} or rule.severity not in {"warning", "critical"}:
                    continue
                rules[rule.name] = rule
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a broken override never disables the defaults
    return tuple(rules.values())


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


# ---------------------------------------------------------------- sinks


DEFAULT_ROTATE_BYTES = 5 * 1024 * 1024
DEFAULT_ROTATE_KEEP = 3


class JsonlSink:
    """Append-only local alert log (one JSON object per line), rotated by size (FIX1 M5).

    When the next line would push ``alerts.jsonl`` past ``max_bytes`` (default 5 MB) it is renamed
    to ``alerts.jsonl.1`` (``.1`` → ``.2`` …), keeping ``keep`` rotated files (default 3); the oldest
    is dropped. Disk use is therefore bounded at about ``max_bytes × (keep + 1)``.
    """

    def __init__(self, path: Path, *, max_bytes: int = DEFAULT_ROTATE_BYTES, keep: int = DEFAULT_ROTATE_KEEP) -> None:
        self.path = Path(path)
        self.max_bytes = max(1024, int(max_bytes))
        self.keep = max(1, int(keep))
        self._lock = threading.Lock()

    def _rotate(self) -> None:
        oldest = self.path.with_name(f"{self.path.name}.{self.keep}")
        oldest.unlink(missing_ok=True)
        for index in range(self.keep - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                os.replace(source, self.path.with_name(f"{self.path.name}.{index + 1}"))
        if self.path.exists():
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))

    def emit(self, alert: dict) -> None:
        line = (json.dumps(alert, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = self.path.stat().st_size
            except OSError:
                size = 0
            if size and size + len(line) > self.max_bytes:
                self._rotate()
            with self.path.open("ab") as handle:
                handle.write(line)


class StatusFileSink:
    """Replace ``alert_status.json`` atomically with the current overall state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write_status(self, status: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)


class WebhookSink:
    """POST each alert as JSON to an operator-configured URL. Never constructed without one."""

    def __init__(self, url: str, *, sender: Callable[[str, bytes], None] | None = None, timeout: float = 5.0) -> None:
        if not url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("alert_webhook_url_must_be_https_or_loopback")
        self.url, self.timeout = url, timeout
        self.sender = sender or self._post

    def _post(self, url: str, body: bytes) -> None:
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout):  # noqa: S310 - operator-configured URL
            pass

    def emit(self, alert: dict) -> None:
        try:
            self.sender(self.url, json.dumps(alert, ensure_ascii=False).encode("utf-8"))
        except Exception:  # noqa: BLE001 - a dead channel must not stop local alerting
            pass


class WindowsEventLogSink:
    """Write to the Windows Application log via ``eventcreate`` (Windows only, opt-in)."""

    def __init__(self, *, runner: Callable[[list[str]], Any] | None = None, source: str = "EvoMind") -> None:
        self.source = source
        self.runner = runner or (lambda argv: subprocess.run(argv, capture_output=True, timeout=10, check=False))

    def emit(self, alert: dict) -> None:
        kind = "ERROR" if alert.get("severity") == "critical" and alert.get("state") == "firing" else "WARNING" if alert.get("state") == "firing" else "INFORMATION"
        text = f"[{alert.get('severity')}/{alert.get('state')}] {alert.get('rule')}: {alert.get('summary')} (value={alert.get('value')})"
        try:
            self.runner(["eventcreate", "/T", kind, "/ID", "1001" if kind == "ERROR" else "1000", "/L", "APPLICATION",
                         "/SO", self.source, "/D", text[:1000]])
        except Exception:  # noqa: BLE001
            pass


def configured_sinks(ops_dir: Path, env: dict | None = None) -> tuple[list, StatusFileSink]:
    """Local JSONL always; webhook / event log only when explicitly configured."""
    env = os.environ if env is None else env
    sinks: list = [JsonlSink(Path(ops_dir) / "alerts.jsonl")]
    url = str(env.get("EVOMIND_ALERT_WEBHOOK_URL") or "").strip()
    if not url and env.get("EVOMIND_ALERT_WEBHOOK_URL_FILE"):
        try:
            url = Path(env["EVOMIND_ALERT_WEBHOOK_URL_FILE"]).read_text(encoding="utf-8").strip()
        except OSError:
            url = ""
    if url:
        try:
            sinks.append(WebhookSink(url))
        except ValueError:
            pass
    if str(env.get("EVOMIND_ALERT_WINDOWS_EVENTLOG") or "") == "1" and os.name == "nt":
        sinks.append(WindowsEventLogSink())
    return sinks, StatusFileSink(Path(ops_dir) / "alert_status.json")


# ---------------------------------------------------------------- evaluator


@dataclass
class _RuleState:
    first_breach: float | None = None
    firing: bool = False
    last_sent: float = float("-inf")


class AlertEvaluator:
    def __init__(self, rules: Iterable[Rule] = DEFAULT_RULES, sinks: Iterable = (), *, status_sink: StatusFileSink | None = None,
                 clock: Callable[[], float] = time.time, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
        self.rules = tuple(rules)
        self.sinks = list(sinks)
        self.status_sink = status_sink
        self.clock = clock
        self.cooldown = float(cooldown_seconds)
        self._state: dict[str, _RuleState] = {rule.name: _RuleState() for rule in self.rules}
        self._previous: dict[str, Any] = {}
        self._values: dict[str, Any] = {}
        self._lock = threading.Lock()

    def evaluate(self, snapshot: dict[str, Any]) -> list[dict]:
        """Return (and send) the alerts this snapshot produces: new firing, re-notify after cooldown, resolved."""
        now = self.clock()
        emitted: list[dict] = []
        with self._lock:
            for rule in self.rules:
                state = self._state[rule.name]
                value = snapshot.get(rule.metric)
                breached = rule.breached(value, self._previous.get(rule.metric))
                if rule.op == "increase":
                    # Event-style rule: one alert per increase, deduped by the cooldown; shown as
                    # firing in the status file until the cooldown runs out. No "resolved".
                    if breached and now - state.last_sent >= self.cooldown:
                        emitted.append(self._alert(rule, "firing", value, now))
                        state.last_sent = now
                    state.firing = now - state.last_sent < self.cooldown
                elif breached:
                    if state.first_breach is None:
                        state.first_breach = now
                    held = now - state.first_breach >= rule.for_seconds
                    if held and (not state.firing or now - state.last_sent >= self.cooldown):
                        emitted.append(self._alert(rule, "firing", value, now, repeat=state.firing))
                        state.firing, state.last_sent = True, now
                else:
                    if state.firing:
                        emitted.append(self._alert(rule, "resolved", value, now))
                    state.firing, state.first_breach, state.last_sent = False, None, float("-inf")
            self._previous = dict(snapshot)
            self._values = dict(snapshot)
            status = self._status(now)
        for alert in emitted:
            for sink in self.sinks:
                try:
                    sink.emit(alert)
                except Exception:  # noqa: BLE001 - one broken sink never blocks the others
                    pass
        if self.status_sink is not None:
            try:
                self.status_sink.write_status(status)
            except OSError:
                pass
        return emitted

    def _alert(self, rule: Rule, state: str, value: Any, now: float, *, repeat: bool = False) -> dict:
        return {"schema": "evomind.alert.v1", "rule": rule.name, "metric": rule.metric, "severity": rule.severity,
                "state": state, "value": value, "threshold": rule.threshold, "op": rule.op, "summary": rule.summary,
                "repeat": repeat, "at": _iso(now)}

    def _status(self, now: float) -> dict:
        firing = [{"rule": rule.name, "severity": rule.severity, "summary": rule.summary}
                  for rule in self.rules if self._state[rule.name].firing]
        overall = max((item["severity"] for item in firing), key=SEVERITY_ORDER.__getitem__, default="ok")
        return {"schema": "evomind.alert_status.v1", "updated_at": _iso(now), "overall": overall, "firing": firing,
                "metrics": {key: value for key, value in self._values.items() if isinstance(value, (int, float)) or value is None}}

    def status(self) -> dict:
        with self._lock:
            return self._status(self.clock())


@dataclass
class OpsMonitor:
    """Collect → evaluate on a fixed tick. ``tick()`` is pure (tests call it with a fake clock)."""

    collect: Callable[[], dict]
    evaluator: AlertEvaluator
    tick_seconds: float = DEFAULT_TICK_SECONDS
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def tick(self) -> list[dict]:
        try:
            snapshot = self.collect()
        except Exception:  # noqa: BLE001 - keep ticking; the next snapshot may succeed
            return []
        return self.evaluator.evaluate(snapshot)

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(self.tick_seconds):
                self.tick()
        self._thread = threading.Thread(target=loop, name="evomind-ops-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def monitor_for(runtime, *, env: dict | None = None, clock: Callable[[], float] = time.time) -> OpsMonitor | None:
    """The default monitor for one runtime, or ``None`` when ``EVOMIND_OPS_MONITOR=0``."""
    from . import ops_metrics

    env = os.environ if env is None else env
    if str(env.get("EVOMIND_OPS_MONITOR", "1")).strip() == "0":
        return None
    try:
        tick = max(5.0, min(float(env.get("EVOMIND_OPS_TICK_SECONDS") or DEFAULT_TICK_SECONDS), 60.0))
    except ValueError:
        tick = DEFAULT_TICK_SECONDS
    ops_dir = Path(runtime.runtime_root) / "ops"
    sinks, status_sink = configured_sinks(ops_dir, env)
    evaluator = AlertEvaluator(load_rules(env.get("EVOMIND_ALERT_RULES_FILE")), sinks, status_sink=status_sink, clock=clock)
    from .run_supervisor import StallSupervisor, enabled as stall_enabled
    supervisor = StallSupervisor(runtime) if stall_enabled(env) else None

    def collect() -> dict:
        if supervisor is not None:  # t238 C7e A4: act on STALLED before the snapshot counts it
            try:
                supervisor.tick()
            except Exception:  # noqa: BLE001 - never stop the metrics/alert tick
                pass
        return ops_metrics.collect(runtime, capacity="sample")

    return OpsMonitor(collect, evaluator, tick)
