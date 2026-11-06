"""Phase 2 knowledge base: private / anonymous global / platform experience (knowledge.sqlite3)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomind_runtime import experience_store as es
from evomind_runtime.tenant_access import AccessError, Principal

ALICE = Principal("tenant_a", "owner_a")
BOB = Principal("tenant_b", "owner_b")
PROFILE = {"rows_bucket": "1k-10k", "cols_bucket": "10-50", "target_kind": "multiclass", "balance_bucket": "imbalanced"}
CONFIG = {"hidden_sizes": [128, 64], "epochs": 40, "batch_size": 128, "learning_rate": 0.001, "data_path": "inputs/x.csv",
          "target_column": "quality", "task": "classification"}


def record(store, principal, *, outcome="success", f1=0.5, run="run_1", data="d" * 64, config=CONFIG, code=""):
    return store.record_training(principal, run_id=run, data_key=data, task="classification", profile=PROFILE, config=config,
                                 outcome=outcome, metrics={"macro_f1": f1, "accuracy": 0.6} if outcome == "success" else None,
                                 failure_code=code, elapsed_seconds=125, card="A800", project_id="proj_1")


def test_private_rows_are_owner_isolated(tmp_path: Path) -> None:
    store = es.ExperienceStore(tmp_path)
    created = record(store, ALICE)
    assert created and es.EXPERIENCE_ID.fullmatch(created)
    assert store.count_mine(ALICE) == 1 and store.count_mine(BOB) == 0
    assert store.list_mine(BOB) == []
    with pytest.raises(AccessError) as error:
        store.delete_mine(BOB, created)
    assert error.value.status == 404
    with pytest.raises(AccessError):
        store.list_mine(None)
    mine = store.list_mine(ALICE)[0]
    assert mine["config"] == {"hidden_sizes": [128, 64], "epochs": 40, "batch_size": 128, "learning_rate": 0.001}
    assert "inputs/x.csv" not in json.dumps(mine) and "quality" not in json.dumps(mine)
    assert store.delete_mine(ALICE, created)["deleted"] is True
    assert store.count_mine(ALICE) == 0
    assert store.search("多分类", scope="private", principal=ALICE) == []


def test_repeat_and_non_experience_refusals_are_not_written(tmp_path: Path) -> None:
    store = es.ExperienceStore(tmp_path)
    assert record(store, ALICE) is not None
    assert record(store, ALICE) is None  # same run / data / config / outcome
    assert record(store, ALICE, outcome="failed", code="tenant_training_queue") is None
    assert record(store, ALICE, outcome="failed", code="exact_approval_required") is None
    assert store.count_mine(ALICE) == 1


def test_success_links_the_failure_it_fixed(tmp_path: Path) -> None:
    store = es.ExperienceStore(tmp_path)
    failed = record(store, ALICE, outcome="failed", code="tenant_training_config_invalid", config={"epochs": 0})
    fixed = record(store, ALICE, run="run_2")
    rows = {row["id"]: row for row in store.list_mine(ALICE)}
    assert rows[fixed]["fixed_failure_id"] == failed
    assert "后来改成这样成功了" in rows[failed]["summary"]
    assert rows[failed]["title"].startswith("失败：训练参数不合法")


def test_recall_budget_ranking_and_credit(tmp_path: Path) -> None:
    store = es.ExperienceStore(tmp_path)
    for index in range(6):
        record(store, ALICE, run=f"run_{index}", f1=0.3 + index / 100, data=str(index) * 64)
    recalled = store.recall(ALICE, task="classification", profile=PROFILE, project_id="proj_1")
    assert 1 <= len(recalled["private_ids"]) <= es.RECALL_PRIVATE
    assert sum(len(line) + 1 for line in recalled["lines"]) <= es.RECALL_CHAR_BUDGET + 200
    assert recalled["recommended"]["epochs"] == 40
    assert store.recall(BOB, task="classification", profile=PROFILE)["lines"] == []
    best = recalled["private_ids"][0]
    store.record_training(ALICE, run_id="run_new", data_key="n" * 64, task="classification", profile=PROFILE, config=CONFIG,
                          outcome="success", metrics={"macro_f1": 0.9}, used_ids=[best])
    assert next(row for row in store.list_mine(ALICE) if row["id"] == best)["success_count"] == 1


def test_recall_char_budget_is_hard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(es, "RECALL_CHAR_BUDGET", 400)
    store = es.ExperienceStore(tmp_path)
    for index in range(3):
        record(store, ALICE, run=f"run_{index}", data=str(index) * 64)
    lines = store.recall(ALICE, task="classification", profile=PROFILE)["lines"]
    assert sum(len(line) + 1 for line in lines) <= 400 + 200  # header + suggestion stay; experience lines are cut


def test_global_needs_three_tenants_and_honours_opt_out(tmp_path: Path) -> None:
    store = es.ExperienceStore(tmp_path)
    owners = [Principal(f"tenant_{index}", f"owner_{index}") for index in range(3)]
    for owner in owners[:2]:
        record(store, owner)
    assert store.aggregate_global()["published"] == 0
    record(store, owners[2])
    assert store.aggregate_global()["published"] == 1
    with sqlite3.connect(tmp_path / es.DB_NAME) as connection:
        prior = connection.execute("SELECT * FROM global_priors").fetchone()
    text = json.dumps(prior)
    assert "tenant_" not in text and "owner_" not in text and "inputs" not in text
    lines = store.recall(Principal("tenant_x", "owner_x"), task="classification", profile=PROFILE)["lines"]
    assert any("平台匿名统计（3 个用户" in line for line in lines)
    assert store.patch_settings(owners[0], {"contribute_global": False}) == {"contribute_global": False}
    assert store.aggregate_global()["published"] == 0
    with pytest.raises(AccessError):
        store.patch_settings(owners[0], {"contribute_global": "no"})


def test_platform_import_is_idempotent_and_chinese_search_works(tmp_path: Path) -> None:
    database = tmp_path / "runtime.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE research_knowledge (id TEXT PRIMARY KEY, tenant TEXT, project TEXT, category TEXT, status TEXT, payload TEXT)")
        connection.execute("INSERT INTO research_knowledge VALUES ('k1','t','p','tabular','proven',?)",
                           (json.dumps({"lesson": "表格数据先做分层抽样再调参"}, ensure_ascii=False),))
        connection.execute("INSERT INTO research_knowledge VALUES ('k2','t','p','tabular','rejected','{\"lesson\":\"x\"}')")
    retro = tmp_path / "task1" / "outputs" / "retrospective_memory.json"
    retro.parent.mkdir(parents=True)
    retro.write_text(json.dumps({"title": "polymer", "lessons": ["学习率过大导致训练发散"]}, ensure_ascii=False), encoding="utf-8")
    first = es.housekeeping(tmp_path, (tmp_path,))
    assert first["research"] == 1 and first["retrospectives"] == 1
    again = es.housekeeping(tmp_path, (tmp_path,))
    assert again["research"] == 0 and "retrospectives" not in again
    store = es.ExperienceStore(tmp_path)
    assert store.search("训练发散", scope="platform")[0]["kind"] == "retrospective"
    assert store.search("分层抽样", scope="platform")[0]["kind"] == "research"
    assert store.platform_lines("分层抽样怎么做")[0].startswith("Platform knowledge")


def test_fts_fallback_without_trigram(tmp_path: Path) -> None:
    store = es.ExperienceStore(tmp_path)
    store.fts = False
    store.import_platform([{"source_key": "a", "title": "t", "body": "学习率过大导致训练发散"}])
    assert store.search("训练发散", scope="platform")


def test_profile_helpers(tmp_path: Path) -> None:
    assert es.data_profile("classification", 1599, 11, labels=[5] * 90 + [6] * 10, classes=6)["balance_bucket"] == "imbalanced"
    assert es.data_profile("regression", 50, 3)["target_kind"] == "regression"
    assert es.primary_metric("regression", {"r2": None, "rmse": 2.0}) == ("neg_rmse", -2.0)
    path = tmp_path / "wine.csv"
    path.write_text("a,b,quality\n" + "".join(f"{i},{i * 2},{5 + i % 3}\n" for i in range(30)), encoding="utf-8")
    task, profile = es.guess_task_profile(path)
    assert task == "classification" and profile["target_kind"] == "multiclass" and profile["rows_bucket"] == "lt1k"
    path.write_text("a,y\n" + "".join(f"{i},{i * 0.37}\n" for i in range(30)), encoding="utf-8")
    assert es.guess_task_profile(path)[0] == "regression"


def test_training_tool_deposits_one_private_experience(tmp_path: Path) -> None:
    from evomind_runtime import tools

    workspace = tmp_path / "ws"
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "inputs" / "x.csv").write_text("a,b,y\n" + "".join(f"{i},{i},{i % 2}\n" for i in range(20)), encoding="utf-8")
    sessions = {"run_1": {"metadata": {"experience_used": {"private_ids": []}}}}
    context = SimpleNamespace(
        metadata={"http_access_scope": {"tenant_id": "tenant_a", "owner_principal_id": "owner_a"}, "project_id": "p1"},
        runtime_root=tmp_path / "runtime", workspace_root=workspace, session_id="run_1",
        store=SimpleNamespace(get_session=sessions.get))
    facts = {"task": "classification", "data_key": "d" * 64, "config": CONFIG}  # failed before prepare: shape from the CSV
    failed = tools.ToolResult("", False, {}, "x", error="tenant_training_config_invalid")
    tools._deposit_training_experience(context, facts, failed, 3.0)
    ok = tools.ToolResult("", True, {}, "done")
    tools._deposit_training_experience(context, {**facts, "profile": PROFILE, "metrics": {"macro_f1": 0.7}, "trainer": CONFIG}, ok, 90.0)
    tools._deposit_training_experience(context, facts, tools.ToolResult("", False, {}, "q", error="tenant_training_queue"), 1.0)
    rows = es.ExperienceStore(context.runtime_root).list_mine(ALICE)
    assert [row["outcome"] for row in rows] == ["success", "failed"]
    assert rows[1]["rows_bucket"] == "lt1k" and rows[0]["fixed_failure_id"] == rows[1]["id"]
    operator = SimpleNamespace(**{**vars(context), "metadata": {"http_access_scope": {"tenant_id": "org_ops", "owner_principal_id": "o"}}})
    tools._deposit_training_experience(operator, {**facts, "profile": PROFILE}, ok, 1.0)
    assert es.ExperienceStore(context.runtime_root).count_mine(Principal("org_ops", "o")) == 0


def test_user_memory_ranks_pinned_then_bm25(tmp_path: Path) -> None:
    from evomind_runtime.user_memory import UserMemoryStore

    store = UserMemoryStore(tmp_path)
    store.add(ALICE, "我喜欢用深色主题")
    relevant = store.add(ALICE, "我常用葡萄酒质量数据做分类训练")["id"]
    pinned = store.add(ALICE, "我的名字是小王")["id"]
    store.patch(ALICE, pinned, {"pinned": True})
    rows = store.select_for_prompt(ALICE, "帮我用葡萄酒数据训练一个分类模型", limit=3)
    assert [row["id"] for row in rows[:2]] == [pinned, relevant]
