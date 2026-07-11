"""Tests for the evolution engine: llm_client, variation_generator, evolution_loop.

These are deterministic and offline: a FakeLLMClient replaces the network so the
generator/loop logic is tested without spending tokens. The strongest checks are
behavioral: CV improves across promotions, a failed proposal flips the loop into
Diff mode, and the promotion gate never promotes a failure.
"""
from __future__ import annotations

import json
from typing import Optional

import pytest

from research_os.evolution_loop import (
    EvolutionConfig,
    EvolutionLoop,
    LocalSubprocessRunner,
    RunResult,
    _classify_failure,
    _parse_cv_score,
)
from research_os.llm_client import LLMClient, LLMError, LLMResponse, _env
from research_os.variation_generator import (
    TaskContext,
    VariationGenerator,
    _extract_code,
    _extract_hypothesis,
)


# ── fakes ──────────────────────────────────────────────────────────────────
class FakeLLMClient:
    """Returns scripted responses so the generator is tested offline."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, user, *, system=None, max_tokens=4096, temperature=None, provider=None):
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return LLMResponse(text=text, provider="fake", model="fake-1", input_tokens=10, output_tokens=20)


def _script(cv_line: str = "print('CV_SCORE=0.9')") -> str:
    return f"hypothesis text here\n```python\nimport json,sys\n{cv_line}\n```"


def _full_script(score: Optional[float]) -> str:
    """A scripted 'solution' that writes the required artifacts, like a real one.

    If score is None, it emits no CV_SCORE and writes nothing (simulates failure).
    """
    if score is None:
        body = "print('boom, no score')"
    else:
        body = (
            "import argparse, json, os\n"
            "ap=argparse.ArgumentParser(); ap.add_argument('--data-dir'); ap.add_argument('--out-dir')\n"
            "a=ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)\n"
            "open(os.path.join(a.out_dir,'submission.csv'),'w').write('id,y\\n1,0\\n')\n"
            f"json.dump({{'cv_score':{score},'metric':'accuracy'}}, open(os.path.join(a.out_dir,'metrics.json'),'w'))\n"
            f"print('CV_SCORE={score}')"
        )
    return f"hypothesis\n```python\n{body}\n```"


# ── llm_client ───────────────────────────────────────────────────────────────
def test_env_prefers_direct_then_file(tmp_path, monkeypatch):
    monkeypatch.delenv("X_TOK", raising=False)
    monkeypatch.setenv("X_TOK", "direct")
    assert _env("X_TOK") == "direct"
    monkeypatch.delenv("X_TOK")
    secret = tmp_path / "s.txt"
    secret.write_text("fromfile", encoding="utf-8")
    monkeypatch.setenv("X_TOK_FILE", str(secret))
    assert _env("X_TOK") == "fromfile"


def test_llm_response_repr_hides_prompt():
    r = LLMResponse(text="secret content", provider="anthropic", model="m", input_tokens=3, output_tokens=4)
    assert "secret content" not in repr(r)
    assert "anthropic" in repr(r)


def test_llm_client_raises_when_no_providers(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE", "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE"):
        monkeypatch.delenv(var, raising=False)
    client = LLMClient()
    assert client.available_providers() == []
    with pytest.raises(LLMError):
        client.generate("hi")


# ── variation_generator ──────────────────────────────────────────────────────
def test_extract_code_picks_longest_block():
    text = "intro\n```python\nx=1\n```\nmid\n```python\nimport os\nprint('CV_SCORE=1.0')\n```"
    code = _extract_code(text)
    assert "CV_SCORE" in code and "import os" in code


def test_extract_hypothesis_is_text_before_code():
    text = "My hypothesis is X.\n```python\nx=1\n```"
    assert "hypothesis" in _extract_hypothesis(text).lower()


def test_generator_propose_returns_runnable_contract():
    ctx = TaskContext("t", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_script()]))
    p = gen.propose(ctx, exp_id="EXP000", mode="Base")
    assert "CV_SCORE" in p.code
    assert p.code_generation_mode == "Base"
    assert p.provider == "fake"


def test_generator_raises_without_code_block():
    ctx = TaskContext("t", "tabular", "classification", "accuracy", "maximize")
    gen = VariationGenerator(client=FakeLLMClient(["no code here"]))
    with pytest.raises(ValueError):
        gen.propose(ctx, exp_id="EXP000", mode="Base")


# ── evolution_loop helpers ───────────────────────────────────────────────────
def test_parse_cv_score_takes_last():
    assert _parse_cv_score("noise\nCV_SCORE=0.5\nmore\nCV_SCORE=0.7") == 0.7
    assert _parse_cv_score("no score here") is None


@pytest.mark.parametrize("err,expected", [
    ("MemoryError: cuda out of memory", "oom"),
    ("ValueError: pandas dtypes must be int", "dtype_encoding"),
    ("KeyError: 'no column named x'", "schema_mismatch"),
    ("process timeout after 900s", "timeout"),
    ("no CV_SCORE emitted", "contract_violation"),
    ("some other boom", "runtime_error"),
    # exit-code diagnostics from a remote kill: these must beat contract_violation
    # even though the message also ends in "...before emitting CV_SCORE".
    ("RUN_EXIT=124 TIMEOUT: process exceeded the 1800s wall budget and was killed "
     "before emitting CV_SCORE.", "timeout"),
    ("RUN_EXIT=137 OOM_OR_KILLED: process received SIGKILL before emitting CV_SCORE.", "oom"),
    ("RUN_EXIT=139 SEGFAULT: native crash (SIGSEGV) before CV_SCORE.", "segfault"),
    # a bare nonzero exit with no recognizable traceback is a generic runtime
    # failure (rc=0-with-no-score is what means contract_violation, not this).
    ("RUN_EXIT=1 NONZERO_EXIT: process failed before emitting CV_SCORE.", "runtime_error"),
    # richer, reusable buckets: the most actionable lessons get their own name
    ("ValueError: The 'liblinear' solver does not support multiclass classification", "estimator_api_misuse"),
    ("TypeError: __init__() got an unexpected keyword argument 'solverx'", "estimator_api_misuse"),
    ("ValueError: shapes (10,3) and (4,) not aligned", "shape_mismatch"),
    ("ModuleNotFoundError: No module named 'timm'", "import_error"),
    ("FileNotFoundError: No such file or directory: 'train/x.jpg'", "file_not_found"),
])
def test_classify_failure(err, expected):
    assert _classify_failure(err) == expected


@pytest.mark.parametrize("rc,needle,bucket", [
    (124, "TIMEOUT", "timeout"),
    (137, "OOM_OR_KILLED", "oom"),
    (139, "SEGFAULT", "segfault"),
    (1, "NONZERO_EXIT", "runtime_error"),
])
def test_gpu_runner_diagnose_exit_names_the_kill(rc, needle, bucket):
    """A remote kill carries NO traceback — only the exit code. _diagnose_exit
    must turn the code into an explicit reason that then classifies correctly,
    instead of the old blind 'no CV_SCORE emitted' -> contract_violation path."""
    from research_os.gpu_runner import _diagnose_exit
    msg = _diagnose_exit(rc, "loading data\n[fold 3] done", timeout_s=1800)
    assert f"RUN_EXIT={rc}" in msg and needle in msg
    # the last real stdout line is preserved for context
    assert "[fold 3] done" in msg
    # and it routes to the right reusable bucket
    assert _classify_failure(msg) == bucket


def test_gpu_runner_diagnose_exit_clean_exit_is_contract_violation():
    """rc==0 with no score is a genuine contract violation (script ran fine but
    never printed CV_SCORE), NOT a kill — it must not be tagged as timeout/oom."""
    from research_os.gpu_runner import _diagnose_exit
    msg = _diagnose_exit(0, "did everything but forgot to print", timeout_s=1800)
    assert "RUN_EXIT" not in msg
    assert _classify_failure(msg) in ("contract_violation", "runtime_error")


class _FakeChannel:
    def __init__(self, rc):
        self._rc = rc
    def recv_exit_status(self):
        return self._rc


class _FakeStream:
    def __init__(self, data=b"", rc=0):
        self._data = data
        self.channel = _FakeChannel(rc)
    def read(self):
        return self._data


class _FakeSSH:
    """Records every command and returns scripted (rc, stdout) per substring.

    Models the exact failure that fabricated a phantom score: the run process is
    killed (rc=124, no CV_SCORE on stdout) while a prior run's metrics.json still
    sits in the reused remote out dir.
    """
    def __init__(self, run_rc, run_out, stale_metrics):
        self.run_rc, self.run_out, self.stale_metrics = run_rc, run_out, stale_metrics
        self.commands = []
    def exec_command(self, command, timeout=None):
        self.commands.append(command)
        if "solution.py" in command and "timeout" in command:
            return None, _FakeStream(self.run_out.encode(), self.run_rc), _FakeStream()
        if "cat" in command and "metrics.json" in command:
            # would return the STALE score if the gate/clean-up let it be read
            return None, _FakeStream(self.stale_metrics.encode(), 0), _FakeStream()
        if command.startswith("ls"):
            return None, _FakeStream(b"metrics.json\nsubmission.csv\n", 0), _FakeStream()
        return None, _FakeStream(b"", 0), _FakeStream()  # mkdir/rm/etc
    def close(self):
        pass
    def open_sftp(self):
        class _SFTP:
            def file(self, *_a, **_k):
                import io as _io
                class _F(_io.BytesIO):
                    def __enter__(self_): return self_
                    def __exit__(self_, *a): return False
                    def write(self_, *_a, **_k): return None
                return _F()
            def close(self): pass
        return _SFTP()


def test_gpu_runner_does_not_fabricate_score_from_stale_metrics_on_kill():
    """no-fabrication invariant: a killed run (rc=124, no CV_SCORE) must NOT adopt
    the metrics.json a prior successful run left in the reused remote out dir.
    Regression for phantom cv_score=0.812 attached to a RUN_EXIT=124 failure."""
    from research_os.gpu_runner import GPURunner, GPURunnerConfig
    ssh = _FakeSSH(run_rc=124, run_out="loading data\n[fold 3] done",
                   stale_metrics='{"cv_score": 0.812769}')
    runner = GPURunner("essay", config=GPURunnerConfig(timeout=60), connect=lambda: ssh)
    res = runner.run("print('x')", data_dir="essay", out_dir="out", exp_id="EXP000")
    assert res.success is False
    assert res.cv_score is None, "failed run must carry NO score (no fabrication)"
    assert "RUN_EXIT=124" in res.error and "TIMEOUT" in res.error
    # defense in depth: the stale out dir is wiped before the run...
    assert any("rm -rf" in c and "out" in c for c in ssh.commands), "out dir not cleaned"
    # ...and the metrics.json fallback is never even reached on a non-zero exit
    assert not any("cat" in c and "metrics.json" in c for c in ssh.commands), \
        "must not read metrics.json after a kill"


def test_corrective_action_injected_for_timeout_and_oom_lessons():
    """The self-solve half: a NAMED failure pattern must produce a concrete fix
    directive in the prompt, not just echo 'FAILED: oom'. Without this the loop
    can diagnose but never actually corrects the next proposal."""
    from research_os.variation_generator import _format_lessons
    text = _format_lessons([
        {"task_type": "text", "failure_pattern": "oom", "what_failed": "OOM_OR_KILLED ..."},
        {"task_type": "text", "failure_pattern": "timeout", "what_failed": "TIMEOUT ..."},
    ])
    assert "REQUIRED CORRECTIVE ACTIONS" in text
    # timeout -> fold/feature caps; oom -> sparse / no toarray
    assert "<=3 CV folds" in text or "<=3 cv folds" in text.lower()
    assert "toarray" in text.lower() and "sparse" in text.lower()


def test_corrective_action_absent_when_no_failure_pattern():
    """No spurious directives when lessons are all successes."""
    from research_os.variation_generator import _format_lessons
    text = _format_lessons([{"task_type": "text", "what_worked": "tfidf + ridge"}])
    assert "REQUIRED CORRECTIVE ACTIONS" not in text


def test_salient_error_surfaces_exception_line_not_path_noise():
    """what_failed must open with the actionable exception, not a mid-token path
    slice. Regression for the liblinear lesson stored as 'ib\\site-packages...'."""
    from research_os.evolution_loop import _salient_error
    tb = (
        'Traceback (most recent call last):\n'
        '  File "C:\\\\tools\\\\py\\\\lib\\\\site-packages\\\\sklearn\\\\_logistic.py", line 1488, in fit\n'
        '    raise ValueError(\n'
        "ValueError: The 'liblinear' solver does not support multiclass "
        "classification (n_classes >= 3). Use another solver or OneVsRestClassifier."
    )
    out = _salient_error(tb, max_chars=300)
    assert "ValueError" in out and "liblinear" in out and "multiclass" in out
    # must not open the message half-way through a windows path token
    assert not out.split("->")[-1].strip().startswith("ib\\")
    assert not out.startswith("site-packages")


def test_failed_parent_score_does_not_suppress_successful_successor(tmp_path):
    """Regression for the aerial-cactus 0/3 anomaly: EXP000 crashed at test-time
    but flushed a high CV score. That failed node must NOT anchor the promotion
    baseline, otherwise genuinely successful successors (delta < min_delta over
    the crashed score) are silently held and nothing ever promotes."""
    from research_os.search_graph import ExperimentNode, SearchGraph

    def node(eid, parent, score, success):
        return ExperimentNode(
            exp_id=eid, parent_id=parent, branch_type="Base", task_name="t",
            hypothesis="", implementation_summary="", code_path=f"{eid}/s.py",
            artifacts=[{"path": "submission.csv"}, {"path": "metrics.json"}],
            cv_score=score, metric_direction="maximize", run_success=success,
        )

    g = SearchGraph(task_id="t", root_exp_id="EXP000", metric_direction="maximize")
    g.add_node(node("EXP000", "root", 0.99989, False))  # crashed, high score
    d0 = g.decide_promotion("EXP000", metric="cv_score", direction="maximize",
                            min_delta=1e-4, required_artifacts=["submission.csv"], run_success=False)
    assert d0["promoted"] is False and g.best_exp_id is None

    g.add_node(node("EXP001", "EXP000", 0.99991, True))  # success, +2e-5 over crash
    d1 = g.decide_promotion("EXP001", parent_exp_id=None, metric="cv_score", direction="maximize",
                            min_delta=1e-4, required_artifacts=["submission.csv"], run_success=True)
    # first *successful* scored node promotes despite the crashed parent's higher-ish score
    assert d1["promoted"] is True and g.best_exp_id == "EXP001"


# ── local runner executes a real subprocess ──────────────────────────────────
def test_local_runner_runs_and_parses(tmp_path):
    code = (
        "import argparse, json, os\n"
        "p=argparse.ArgumentParser(); p.add_argument('--data-dir'); p.add_argument('--out-dir')\n"
        "a=p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)\n"
        "open(os.path.join(a.out_dir,'submission.csv'),'w').write('id,y\\n1,0\\n')\n"
        "json.dump({'cv_score':0.83,'metric':'accuracy'}, open(os.path.join(a.out_dir,'metrics.json'),'w'))\n"
        "print('CV_SCORE=0.83')\n"
    )
    runner = LocalSubprocessRunner(tmp_path, timeout=60)
    (tmp_path / "data").mkdir()
    res = runner.run(code, data_dir=str(tmp_path / "data"), out_dir=str(tmp_path / "out"), exp_id="EXP000")
    assert res.success and res.cv_score == pytest.approx(0.83)
    assert any(a.endswith("submission.csv") for a in res.artifacts)


def test_local_runner_reports_failure_on_bad_code(tmp_path):
    runner = LocalSubprocessRunner(tmp_path, timeout=60)
    (tmp_path / "data").mkdir()
    res = runner.run("raise SystemExit('boom')", data_dir=str(tmp_path / "data"),
                     out_dir=str(tmp_path / "out"), exp_id="EXP000")
    assert not res.success and res.cv_score is None


# ── full loop behavior with fake generator + fake runner ──────────────────────
class _FakeGenLoop:
    """Drives EvolutionLoop with scripted proposals to test loop control flow."""


def test_full_loop_promotes_only_improvements_and_switches_to_diff(tmp_path, monkeypatch):
    # Scripted LLM: EXP000 ok(0.60), EXP001 ok(0.55 worse), EXP002 FAIL(no score), EXP003 ok(0.65)
    scripts = [
        _full_script(0.60),
        _full_script(0.55),
        _full_script(None),   # no CV_SCORE -> failure
        _full_script(0.65),
    ]
    ctx = TaskContext("loop-test", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient(scripts))
    runner = LocalSubprocessRunner(tmp_path / "work", timeout=60)
    (tmp_path / "data").mkdir()
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=runner, generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=4, stagnation_patience=1))
    summary = loop.run()

    modes = [it["mode"] for it in summary["iterations"]]
    proms = [it["promoted"] for it in summary["iterations"]]
    scores = [it["cv_score"] for it in summary["iterations"]]
    assert scores[0] == pytest.approx(0.60)
    assert proms[0] is True                     # baseline promoted
    assert proms[1] is False                    # 0.55 worse -> held
    assert proms[2] is False                    # failure -> held
    assert scores[3] == pytest.approx(0.65) and proms[3] is True  # improvement promoted
    assert "Diff" in modes                      # failure/stagnation flipped to Diff
    assert summary["best_cv_score"] == pytest.approx(0.65)
    assert summary["n_promotions"] == 2
    # memory recorded a failure lesson
    records = json.loads((tmp_path / "mem.json").read_text(encoding="utf-8"))
    assert any(r["failure_pattern"] for r in records)


# ── cross-task memory reuse (the essence of evolution) ───────────────────────
class PromptCapturingLLM:
    """Records the user prompt sent to the LLM, returns a fixed valid script."""

    def __init__(self, script: str) -> None:
        self.script = script
        self.prompts: list[str] = []

    def generate(self, user, *, system=None, max_tokens=4096, temperature=None, provider=None):
        self.prompts.append(user)
        return LLMResponse(text=self.script, provider="fake", model="fake-1")


def test_lessons_from_prior_task_reach_a_new_task_prompt(tmp_path):
    from research_os.retrospective_memory import MemoryRecord, RetrospectiveMemoryStore

    mem = RetrospectiveMemoryStore(tmp_path / "shared_mem.json")
    # A prior CLASSIFICATION task learned something reusable.
    mem.add_memory(MemoryRecord(
        memory_id="prior-task:EXP001", task_type="classification",
        dataset_profile={"modality": "tabular"}, method="Stepwise:target_encoding",
        what_worked="splitting the string code column into per-character features lifted AUC",
        what_failed="", metric_delta=0.01,
        reusable_strategy="per_character_features", failure_pattern="", linked_exp_ids=["EXP001"],
    ))

    # A NEW classification task (different name) should retrieve that lesson.
    ctx = TaskContext("brand-new-task", "tabular", "classification", "roc_auc", "maximize", target_column="target")
    capturing = PromptCapturingLLM(_full_script(0.80))
    gen = VariationGenerator(client=capturing)
    runner = LocalSubprocessRunner(tmp_path / "work", timeout=60)
    (tmp_path / "data").mkdir()
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=runner, generator=gen, memory=mem,
                         config=EvolutionConfig(max_iterations=1))
    loop.run()

    assert capturing.prompts, "generator was never called"
    first_prompt = capturing.prompts[0]
    # The prior task's lesson text must have been injected into the new task's prompt.
    assert "per-character features" in first_prompt or "per_character_features" in first_prompt
    assert "RETROSPECTIVE MEMORY" in first_prompt


def test_loop_emits_library_sourced_audit_artifacts(tmp_path):
    """Each experiment must write validation_contract.json + claim_audit.json
    with the research_os library schema (single source of truth, no inline .v1 fork)."""
    ctx = TaskContext("audit-task", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.77)]))
    runner = LocalSubprocessRunner(tmp_path / "work", timeout=60)
    (tmp_path / "data").mkdir()
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=runner, generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=1))
    loop.run()
    vc = tmp_path / "work" / "EXP000" / "validation_contract.json"
    ca = tmp_path / "work" / "EXP000" / "claim_audit.json"
    assert vc.exists() and ca.exists()
    vc_data = json.loads(vc.read_text(encoding="utf-8"))
    ca_data = json.loads(ca.read_text(encoding="utf-8"))
    assert vc_data["schema"] == "academic_research_os.validation_contract.v1"
    assert ca_data["schema"] == "academic_research_os.claim_audit.v1"
    assert vc_data["cv_score"] == pytest.approx(0.77)
    assert "drift_type" in ca_data  # produced by research_os.claim_audit.audit_claim


class _RaisingRunner:
    """Simulates a transient runner/SSH failure on the first call, then succeeds."""

    def __init__(self, tmp_path):
        self.calls = 0
        self._ok = LocalSubprocessRunner(tmp_path / "work", timeout=60)

    def run(self, code, *, data_dir, out_dir, exp_id):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("simulated SOCKS connect timeout")
        return self._ok.run(code, data_dir=data_dir, out_dir=out_dir, exp_id=exp_id)


def test_transient_runner_exception_does_not_crash_task(tmp_path):
    """With transient_retries=0 (opt-out), a transient runner exception is recorded
    as a failed run and the loop keeps going (recovers on the NEXT iteration)."""
    ctx = TaskContext("resilience-task", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.70), _full_script(0.72)]))
    (tmp_path / "data").mkdir()
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=_RaisingRunner(tmp_path), generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=2, transient_retries=0))
    summary = loop.run()  # must NOT raise
    assert summary["n_iterations"] == 2
    assert summary["iterations"][0]["success"] is False   # transient failure recorded
    assert summary["iterations"][1]["success"] is True    # recovered next iteration
    assert summary["best_cv_score"] == pytest.approx(0.72)


def test_transient_infra_error_is_retried_in_place_not_burned(tmp_path):
    """With transient_retries>=1 (the default), an SSH/SOCKS blip re-runs the SAME
    proposal within the iteration, so a network hiccup doesn't cost an LLM proposal
    or needlessly flip the loop into Diff mode. The first iteration should SUCCEED."""
    ctx = TaskContext("resilience-retry", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.70), _full_script(0.72)]))
    (tmp_path / "data").mkdir()
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=_RaisingRunner(tmp_path), generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=2, transient_retries=1))
    summary = loop.run()
    # iteration 0's TimeoutError was retried in-place and then succeeded (0.70).
    assert summary["iterations"][0]["success"] is True
    assert summary["iterations"][0]["cv_score"] == pytest.approx(0.70)
    assert summary["iterations"][1]["cv_score"] == pytest.approx(0.72)
    assert summary["best_cv_score"] == pytest.approx(0.72)


def test_is_transient_infra_recognizes_ssh_and_timeout():
    from research_os.evolution_loop import _is_transient_infra
    class _SSHException(Exception):
        pass
    assert _is_transient_infra(TimeoutError("timed out")) is True
    assert _is_transient_infra(ConnectionResetError("reset")) is True
    assert _is_transient_infra(_SSHException("SSH session not active")) is True
    assert _is_transient_infra(RuntimeError("SOCKS connect timeout")) is True
    # a real code bug is NOT transient -> must not be retried
    assert _is_transient_infra(ValueError("could not convert string to float")) is False
    assert _is_transient_infra(KeyError("no column named target")) is False



class _FailButScoredRunner:
    """A run that FAILS (non-zero exit / remote kill) yet still flushed a valid
    score + artifacts to disk before dying. This is exactly the GPU-timeout /
    OOM case observed on tps-may-2022. The gate must refuse to promote it.
    """

    def __init__(self, tmp_path, score: float):
        self.tmp_path = tmp_path
        self.score = score

    def run(self, code, *, data_dir, out_dir, exp_id):
        import os
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "submission.csv"), "w").write("id,y\n1,0\n")
        json.dump({"cv_score": self.score, "metric": "roc_auc"},
                  open(os.path.join(out_dir, "metrics.json"), "w"))
        artifacts = [os.path.join(out_dir, "submission.csv"), os.path.join(out_dir, "metrics.json")]
        # success=False despite a real score + artifacts (process was killed).
        return RunResult(False, self.score, error="process killed after flush", out_dir=out_dir, artifacts=artifacts)


def test_gate_refuses_to_promote_failed_run_even_with_score_and_artifacts(tmp_path):
    """Unit test at the gate: run_success=False must never promote, even when the
    candidate score would otherwise win and all required artifacts are present."""
    from research_os.search_graph import ExperimentNode, SearchGraph
    g = SearchGraph(task_id="t", root_exp_id="EXP000", metric_name="cv_score", metric_direction="maximize")
    g.add_node(ExperimentNode(
        exp_id="EXP000", parent_id=None, branch_type="Base", task_name="t",
        hypothesis="h", implementation_summary="s", code_path="EXP000/solution.py",
        cv_score=0.99, metric_name="cv_score", metric_direction="maximize",
        artifacts=[{"path": "submission.csv"}, {"path": "metrics.json"}],
    ))
    decision = g.decide_promotion(
        "EXP000", metric="cv_score", direction="maximize", min_delta=1e-4,
        required_artifacts=["metrics.json", "submission.csv"], run_success=False,
    )
    assert decision["promoted"] is False
    assert "success" in decision["reason"].lower()
    assert g.best_exp_id is None  # a failed run must not become best


def test_failed_run_with_valid_score_is_not_promoted_in_loop(tmp_path):
    """Loop-level regression for the tps-may-2022 anomaly: the runner reports a
    valid CV score + artifacts but success=False (remote kill). The engine must
    record it as not-promoted and keep best_cv_score empty."""
    ctx = TaskContext("kill-after-flush", "tabular", "classification", "roc_auc", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.99)]))
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=_FailButScoredRunner(tmp_path, 0.99), generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=1))
    (tmp_path / "data").mkdir()
    summary = loop.run()
    assert summary["iterations"][0]["success"] is False
    assert summary["iterations"][0]["promoted"] is False   # THE FIX: failed run not promoted
    assert summary["best_cv_score"] is None
    assert summary["n_promotions"] == 0


class _ProgressNoiseRunner:
    """First run fails with a REAL error buried under download-progress noise;
    second run succeeds. Mirrors the aerial-cactus EXP001 case where the captured
    'error' was a torch.hub download bar that hid the actual traceback.
    """

    def __init__(self, tmp_path, real_error: str, good_score: float):
        self.calls = 0
        self.real_error = real_error
        self._ok = LocalSubprocessRunner(tmp_path / "work", timeout=60)
        self.good_score = good_score

    def run(self, code, *, data_dir, out_dir, exp_id):
        self.calls += 1
        if self.calls == 1:
            noise = "".join(f"\r {p}%|##  | {p}.1M/44.7M [04:0{p}<15:01, 35.5kB/s]" for p in range(1, 9))
            blob = f"{noise}\n{self.real_error}\n{noise}"
            return RunResult(False, None, error=blob, out_dir=out_dir)
        return self._ok.run(code, data_dir=data_dir, out_dir=out_dir, exp_id=exp_id)


def test_failed_run_persists_clean_error_and_feeds_it_to_diff_prompt(tmp_path):
    """Piece A+B: on failure the full noise-stripped error must be written to
    EXP###/run_error.txt, and the REAL error (not the progress bar) must reach the
    next Diff-mode prompt so recovery is not blindfolded."""
    real_error = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    ctx = TaskContext("obs-recovery", "image", "classification", "roc_auc", "maximize", target_column="y")
    capturing = PromptCapturingLLM(_full_script(0.91))
    gen = VariationGenerator(client=capturing)
    (tmp_path / "data").mkdir()
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=_ProgressNoiseRunner(tmp_path, real_error, 0.91), generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=2))
    summary = loop.run()

    # A: the full error was persisted, noise-stripped.
    err_file = tmp_path / "work" / "EXP000" / "run_error.txt"
    assert err_file.exists()
    saved = err_file.read_text(encoding="utf-8")
    assert "CUDA out of memory" in saved
    assert "kB/s]" not in saved                      # progress frames stripped
    # B: the second (Diff) prompt received the REAL error, not the progress bar.
    assert len(capturing.prompts) >= 2
    diff_prompt = capturing.prompts[1]
    assert "CUDA out of memory" in diff_prompt
    assert "kB/s]" not in diff_prompt
    assert summary["iterations"][0]["success"] is False
    assert summary["iterations"][1]["success"] is True


def test_failed_baseline_recovers_via_diff_on_failed_code(tmp_path):
    """If the very first (Base) attempt fails, the next attempt must go Diff on the
    FAILED code (to debug it), not regenerate Base from scratch."""
    # EXP000 fails (no score); EXP001 must be Diff and receive the failed code as base.
    scripts = [_full_script(None), _full_script(0.88)]
    ctx = TaskContext("debug-recovery", "image", "classification", "roc_auc", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient(scripts))
    # capture the mode by inspecting prompts: wrap generator to record modes
    seen_modes = []
    orig = gen.propose
    def spy(context, *, exp_id, mode, **kw):
        seen_modes.append(mode)
        return orig(context, exp_id=exp_id, mode=mode, **kw)
    gen.propose = spy
    runner = LocalSubprocessRunner(tmp_path / "work", timeout=60)
    (tmp_path / "data").mkdir()
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=runner, generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=2))
    summary = loop.run()
    assert seen_modes[0] == "Base"          # first attempt is Base
    assert seen_modes[1] == "Diff"          # after failure, debug via Diff (not Base)
    assert summary["iterations"][0]["success"] is False
    assert summary["iterations"][1]["success"] is True
    assert summary["best_cv_score"] == pytest.approx(0.88)


# ── MCGS selector driving the loop end-to-end (offline) ──────────────────────
def test_selector_drives_loop_and_uct_visits_accumulate(tmp_path):
    """With a real MCGSSelector plugged in, the loop must run end-to-end, find the
    best score, AND accumulate UCT visit counts (proving the brain is live, not the
    dead visit_count of the old engine)."""
    from research_os.mcgs_selector import MCGSSelector
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    ctx = TaskContext("mcgs-drive", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.70), _full_script(0.72), _full_script(0.71)]))
    (tmp_path / "data").mkdir()
    selector = MCGSSelector(total_steps=3)
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=LocalSubprocessRunner(tmp_path / "work", timeout=60), generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=3, min_delta=1e-4), selector=selector)
    summary = loop.run()
    assert summary["n_iterations"] == 3
    assert summary["best_cv_score"] == pytest.approx(0.72)
    # UCT is alive: visits accumulated up the tree across iterations.
    assert sum(selector.visits.values()) >= 3
    assert selector.visits.get("EXP000", 0) >= 1


class _BoomSelector:
    """A selector that always raises. The loop must degrade to linear, not crash."""
    def select(self, graph, *, step):
        raise RuntimeError("selector boom")
    def register_child(self, *a, **k):
        pass
    def backpropagate(self, *a, **k):
        pass


def test_selector_exception_falls_back_to_linear(tmp_path):
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    ctx = TaskContext("mcgs-fallback", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.70), _full_script(0.72)]))
    (tmp_path / "data").mkdir()
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=LocalSubprocessRunner(tmp_path / "work", timeout=60), generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=2), selector=_BoomSelector())
    summary = loop.run()  # must NOT raise despite the selector blowing up every step
    assert summary["n_iterations"] == 2
    assert summary["best_cv_score"] == pytest.approx(0.72)   # linear path still works


class _AggregationSelector:
    """Forces an aggregation plan that references EXP000, to prove reference code
    (from another node) reaches the generator prompt."""
    def __init__(self):
        self.branch_of = {}
    def select(self, graph, *, step):
        from research_os.mcgs_selector import ExpansionPlan
        return ExpansionPlan(node_exp_id="EXP000", expansion_type="aggregation",
                             coding_mode="Base", reference_exp_ids=["EXP000"], branch_id="branch_1")
    def register_child(self, plan, child_id):
        self.branch_of[child_id] = plan.branch_id
    def backpropagate(self, *a, **k):
        pass


def test_selector_reference_solution_reaches_prompt(tmp_path):
    """cross/aggregation expansions must inject the referenced node's code into the
    prompt — the capability the old system entirely lacked."""
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    ctx = TaskContext("mcgs-agg", "tabular", "classification", "accuracy", "maximize", target_column="y")
    capturing = PromptCapturingLLM(_full_script(0.80))
    gen = VariationGenerator(client=capturing)
    (tmp_path / "data").mkdir()
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
                         runner=LocalSubprocessRunner(tmp_path / "work", timeout=60), generator=gen,
                         memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
                         config=EvolutionConfig(max_iterations=2), selector=_AggregationSelector())
    loop.run()
    assert len(capturing.prompts) >= 2
    agg_prompt = capturing.prompts[1]           # iteration 1 uses the aggregation plan
    assert "AGGREGATION" in agg_prompt
    assert "EXP000" in agg_prompt               # the referenced node is named
    assert "TOP SOLUTIONS TO FUSE" in agg_prompt


class _NullSelector:
    """A minimal, always-primary selector: it never fails and never expands, so the
    ONLY behavioral difference from linear mode is that a selector is present. This
    isolates the global-stagnation early-break gating (selector is None vs not)."""
    def select(self, graph, *, step):
        from research_os.mcgs_selector import ExpansionPlan
        # Always expand the root as a primary Base child (topology stays linear-ish),
        # so the run itself behaves like linear except for the break-gating branch.
        root = graph.root_exp_id or next(iter(graph.nodes))
        return ExpansionPlan(node_exp_id=root, expansion_type="primary", coding_mode="Base")
    def register_child(self, *a, **k):
        pass
    def backpropagate(self, *a, **k):
        pass


def _stagnating_loop(tmp_path, *, selector):
    """Build a loop whose every proposal scores identically (0.50), so after >4 scored
    nodes detect_global_stagnation() is True. Six iterations give it room to break."""
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    ctx = TaskContext("stall", "tabular", "classification", "accuracy", "maximize", target_column="y")
    # 6 identical scores → no improvement across the window → global stagnation.
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.50)] * 6))
    (tmp_path / "data").mkdir()
    return EvolutionLoop(
        ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
        runner=LocalSubprocessRunner(tmp_path / "work", timeout=60), generator=gen,
        memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
        config=EvolutionConfig(max_iterations=6, min_delta=1e-4), selector=selector,
    )


def test_linear_mode_early_breaks_on_global_stagnation(tmp_path):
    """Baseline behavior: with NO selector, a plateau should stop the search early
    (don't burn budget on a solved-flat problem)."""
    loop = _stagnating_loop(tmp_path, selector=None)
    summary = loop.run()
    # Stagnation detected after 5 scored nodes (window=4), so it breaks before 6.
    assert summary["n_iterations"] < 6
    assert summary["n_iterations"] >= 5   # needs >window scored nodes to detect it


def test_mcgs_mode_does_not_early_break_on_global_stagnation(tmp_path):
    """THE regression: with the MCGS brain active, global stagnation is the CUE to
    fuse branches (cross_branch/aggregation), not a reason to quit. The loop must run
    its full budget so the brain can do its most valuable work. Guards evolution_loop
    line ~313, which previously broke unconditionally and killed MCGS mid-search."""
    loop = _stagnating_loop(tmp_path, selector=_NullSelector())
    summary = loop.run()
    assert summary["n_iterations"] == 6   # full budget, NO early break under stagnation


def test_evolution_memory_records_run_promotion_and_evidence_truthfully(tmp_path):
    from research_os.retrospective_memory import RetrospectiveMemoryStore

    ctx = TaskContext(
        "memory-truth", "tabular", "classification", "accuracy", "maximize",
        target_column="y",
    )
    (tmp_path / "data").mkdir()
    store = RetrospectiveMemoryStore(tmp_path / "mem.json")
    loop = EvolutionLoop(
        ctx,
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "work",
        runner=LocalSubprocessRunner(tmp_path / "work", timeout=60),
        generator=VariationGenerator(client=FakeLLMClient([
            _full_script(0.70), _full_script(0.72), _full_script(0.71),
        ])),
        memory=store,
        config=EvolutionConfig(max_iterations=3),
    )
    loop.run(strategies=["feature_crossing", "oof_stacking"])

    records = store._load()
    first, improved, held = records
    assert first.dataset_profile == {
        "modality": "tabular",
        "n_train": 0,
        "run_success": True,
        "promoted": True,
        "evidence_level": "observed",
        "outcome_status": "promoted",
    }
    assert improved.dataset_profile["run_success"] is True
    assert improved.dataset_profile["promoted"] is True
    assert improved.dataset_profile["evidence_level"] == "validated"
    assert improved.dataset_profile["outcome_status"] == "promoted"
    assert improved.metric_delta == pytest.approx(0.02)
    assert held.dataset_profile["run_success"] is True
    assert held.dataset_profile["promoted"] is False
    assert held.dataset_profile["evidence_level"] == "failure"
    assert held.dataset_profile["outcome_status"] == "held"


def test_failed_evolution_run_is_failure_evidence(tmp_path):
    from research_os.retrospective_memory import RetrospectiveMemoryStore

    ctx = TaskContext(
        "memory-failure", "tabular", "classification", "accuracy", "maximize",
        target_column="y",
    )
    (tmp_path / "data").mkdir()
    store = RetrospectiveMemoryStore(tmp_path / "mem.json")
    loop = EvolutionLoop(
        ctx,
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "work",
        runner=LocalSubprocessRunner(tmp_path / "work", timeout=60),
        generator=VariationGenerator(client=FakeLLMClient([_full_script(None)])),
        memory=store,
        config=EvolutionConfig(max_iterations=1),
    )
    loop.run()
    profile = store._load()[0].dataset_profile
    assert profile["run_success"] is False
    assert profile["promoted"] is False
    assert profile["evidence_level"] == "failure"
    assert profile["outcome_status"] == "failed"


def test_only_explicit_multi_strategy_run_records_innovation_attempts(tmp_path):
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    from xsci.innovation_engine import InnovationEngine

    ctx = TaskContext(
        "innovation-truth", "tabular", "classification", "accuracy", "maximize",
        target_column="y",
    )
    (tmp_path / "data").mkdir()
    engine = InnovationEngine(workspace_root=tmp_path)

    normal = EvolutionLoop(
        ctx,
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "normal",
        runner=LocalSubprocessRunner(tmp_path / "normal", timeout=60),
        generator=VariationGenerator(client=FakeLLMClient([_full_script(0.70)])),
        memory=RetrospectiveMemoryStore(tmp_path / "normal-memory.json"),
        config=EvolutionConfig(max_iterations=1),
        innovation_engine=engine,
    )
    normal.run(strategies=["feature_crossing", "oof_stacking"])
    assert engine.stats()["executed_attempts"] == 0

    single = EvolutionLoop(
        ctx,
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "single",
        runner=LocalSubprocessRunner(tmp_path / "single", timeout=60),
        generator=VariationGenerator(client=FakeLLMClient([
            _full_script(0.70), _full_script(0.72),
        ])),
        memory=RetrospectiveMemoryStore(tmp_path / "single-memory.json"),
        config=EvolutionConfig(max_iterations=2),
        innovation_engine=engine,
    )
    single.run(
        strategies=["feature_crossing"],
        innovation_strategy_name="feature_crossing experiment",
    )
    assert engine.stats()["executed_attempts"] == 0

    explicit = EvolutionLoop(
        ctx,
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "explicit",
        runner=LocalSubprocessRunner(tmp_path / "explicit", timeout=60),
        generator=VariationGenerator(client=FakeLLMClient([
            _full_script(0.70), _full_script(0.72),
        ])),
        memory=RetrospectiveMemoryStore(tmp_path / "explicit-memory.json"),
        config=EvolutionConfig(max_iterations=2),
        innovation_engine=engine,
    )
    explicit.run(
        strategies=["feature_crossing", "oof_stacking"],
        innovation_strategy_name="feature_crossing + oof_stacking",
        innovation_source_memory_ids=["memory-a", "memory-b"],
    )

    stats = engine.stats()
    assert stats["executed_attempts"] == 1
    assert stats["innovations_tried"] == 1
    assert stats["successes"] == 1
    assert stats["failures"] == 0
    log = json.loads(
        (tmp_path / ".xsci" / "innovation_log.json").read_text(encoding="utf-8")
    )
    assert [item["attempt_status"] for item in log["tried"]] == ["validated_success"]
    assert log["tried"][0]["source_memory_ids"] == ["memory-a", "memory-b"]


def test_minimize_metric_records_positive_improvement_delta_for_innovation(tmp_path):
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    from xsci.innovation_engine import InnovationEngine

    ctx = TaskContext(
        "innovation-minimize", "tabular", "regression", "rmse", "minimize",
        target_column="y",
    )
    (tmp_path / "data").mkdir()
    engine = InnovationEngine(workspace_root=tmp_path)
    loop = EvolutionLoop(
        ctx,
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "minimize",
        runner=LocalSubprocessRunner(tmp_path / "minimize", timeout=60),
        generator=VariationGenerator(client=FakeLLMClient([
            _full_script(0.30), _full_script(0.25),
        ])),
        memory=RetrospectiveMemoryStore(tmp_path / "minimize-memory.json"),
        config=EvolutionConfig(max_iterations=2),
        innovation_engine=engine,
    )
    loop.run(
        strategies=["log1p_target", "oof_stacking"],
        innovation_strategy_name="log1p_target + oof_stacking",
    )

    log = json.loads(
        (tmp_path / ".xsci" / "innovation_log.json").read_text(encoding="utf-8")
    )
    assert len(log["tried"]) == 1
    assert log["tried"][0]["metric_delta"] == pytest.approx(0.05)
    assert log["tried"][0]["success"] is True



