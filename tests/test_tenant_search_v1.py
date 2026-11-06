"""Phase 3: auto model selection (candidates, CV search, prepare wiring, receipt binding, comparison report)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evomind_runtime import managed_tensor_train as card
from evomind_runtime import tenant_search, tenant_training, tools
from evomind_runtime.tenant_training import TenantTrainingError
from evomind_runtime.training_control import digest

BASE = {"hidden_sizes": [64, 32], "epochs": 40, "batch_size": 64, "learning_rate": 0.001}


def test_candidates_start_with_the_default_and_stay_distinct_and_valid() -> None:
    chosen = tenant_search.candidates(BASE, task="classification", warm={"hidden_sizes": [128], "epochs": 30, "batch_size": 32, "learning_rate": 0.003})
    assert chosen[0] == {**BASE, "class_weight": "none"}
    assert 2 <= len(chosen) <= tenant_search.MAX_TRIALS
    assert len({json.dumps(item, sort_keys=True) for item in chosen}) == len(chosen)
    assert chosen[1]["hidden_sizes"] == [128]  # warm start right after the default
    assert any(item["class_weight"] == "balanced" for item in chosen)
    regression = tenant_search.candidates(BASE, task="regression")
    assert all(item["class_weight"] == "none" for item in regression)

    def refuse_wide(candidate):
        if max(candidate["hidden_sizes"]) > 64:
            raise ValueError("too wide")
    assert all(max(item["hidden_sizes"]) <= 64 for item in tenant_search.candidates(BASE, task="classification", validate=refuse_wide))


def test_folds_cover_every_row_once_and_keep_classes_spread() -> None:
    y = np.array([0] * 30 + [1] * 6)
    folds = card.fold_indices(y, "classification", 3, seed=1)
    joined = np.sort(np.concatenate(folds))
    assert joined.tolist() == list(range(36))
    assert all((y[fold] == 1).sum() == 2 for fold in folds)


def test_class_weights_balance_present_classes_and_zero_absent() -> None:
    weights = card.class_weights(np.array([1, 1, 1, 2]), 4)
    assert weights[0] == 0 and weights[3] == 0
    assert weights[2] == pytest.approx(3 * weights[1])


def test_run_search_picks_best_cv_and_ties_keep_the_default() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, 3)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)

    def fit_predict(candidate, x_train, y_train, x_valid):
        good = candidate["class_weight"] == "balanced"
        signal = x_valid[:, 0] if good else rng.normal(size=len(x_valid))
        return np.stack([-signal, signal], axis=1), 5

    candidates = [{"class_weight": "none"}, {"class_weight": "balanced"}]
    seen = []
    result = card.run_search(candidates, x, y, "classification", 3, 600, 0, fit_predict, report=lambda done, total: seen.append((done, total)))
    assert result["selected_index"] == 1 and result["steps"] == 30 and len(result["results"]) == 2
    assert seen == [(n, 6) for n in range(1, 7)]
    tie = card.run_search([{"class_weight": "balanced"}, {"class_weight": "balanced"}], x, y, "classification", 3, 600, 0, fit_predict)
    assert tie["selected_index"] == 0


def test_run_search_stops_at_the_budget_but_always_scores_the_default() -> None:
    now = [0.0]

    def clock():
        return now[0]

    def fit_predict(candidate, x_train, y_train, x_valid):
        now[0] += 100.0
        return np.zeros(len(x_valid)), 1

    y = np.arange(30, dtype=np.float64)
    result = card.run_search([{}] * 6, y.reshape(-1, 1), y, "regression", 3, 500, 0, fit_predict, clock=clock)
    assert len(result["results"]) == 1 and result["stopped_early"] is True and result["selected_index"] == 0


def _workspace(tmp_path: Path, rows: int = 200) -> tuple[Path, str]:
    root = tmp_path / "ws"
    (root / "inputs").mkdir(parents=True)
    rng = np.random.default_rng(3)
    lines = ["a,b,c,y"] + [f"{rng.normal():.4f},{rng.normal():.4f},{rng.normal():.4f},{i % 3}" for i in range(rows)]
    path = root / "inputs" / "data.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_auto_by_default_mlp_on_request_and_rejects_unknown_family(tmp_path: Path) -> None:
    root, protocol = _workspace(tmp_path)
    config = {"data_path": "inputs/data.csv", "task": "classification", **BASE}
    auto = tenant_training.prepare(root, config, protocol, warm_start={"hidden_sizes": [16], "epochs": 20, "batch_size": 32, "learning_rate": 0.01})
    search = auto["trainer_config"]["search"]
    assert auto["model_family"] == "auto" and search["metric"] == "macro_f1" and search["folds"] == 3
    assert search["candidates"][0] == {**BASE, "class_weight": "none"} and search["candidates"][1]["hidden_sizes"] == [16]
    assert auto["trainer_config"]["hidden_sizes"] == BASE["hidden_sizes"] and auto["trainer_config"]["class_weight"] == "none"
    plain = tenant_training.prepare(root, {**config, "model_family": "mlp"}, protocol)
    assert plain["model_family"] == "mlp" and "search" not in plain["trainer_config"]
    with pytest.raises(TenantTrainingError):
        tenant_training.prepare(root, {**config, "model_family": "gbdt"}, protocol)


def test_prepare_falls_back_to_one_setting_when_the_search_would_not_fit(tmp_path: Path, monkeypatch) -> None:
    root, protocol = _workspace(tmp_path)
    monkeypatch.setattr(tenant_search, "BUDGET_SECONDS", 0)
    prepared = tenant_training.prepare(root, {"data_path": "inputs/data.csv", "task": "classification", **BASE}, protocol)
    assert prepared["model_family"] == "mlp" and "search" not in prepared["trainer_config"]


def _card_output(tmp_path: Path, trainer: dict, index: int) -> tuple[Path, str, dict]:
    """What the card writes for a search run (model-config + receipt binding), without torch."""
    work = tmp_path / "work"
    out = tmp_path / "out"
    work.mkdir(); out.mkdir()
    (work / "training-config.json").write_text(json.dumps(trainer, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    expected = digest(work / "training-config.json")
    model_config = {key: value for key, value in trainer.items() if key != "search"}
    model_config.update({key: trainer["search"]["candidates"][index][key] for key in card.CANDIDATE_FIELDS})
    model_config.update(training_config_sha256=expected, selected_candidate=index)
    (out / "model-config.json").write_text(json.dumps(model_config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt = {"model_selection": {"selected_index": index, "results": [{"index": 0, "cv_score": 0.3}, {"index": 1, "cv_score": 0.4}], "folds": 3},
               "model_config_sha256": digest(out / "model-config.json")}
    return out, expected, receipt


def test_selected_config_is_bound_to_the_receipt_and_the_approved_candidates(tmp_path: Path) -> None:
    root, protocol = _workspace(tmp_path)
    trainer = tenant_training.prepare(root, {"data_path": "inputs/data.csv", "task": "classification", **BASE}, protocol)["trainer_config"]
    out, expected, receipt = _card_output(tmp_path, trainer, 1)
    selected = tools._tenant_selected_config(out, trainer, receipt, expected)
    assert selected == trainer["search"]["candidates"][1]
    assert tools._tenant_selected_config(out, trainer, {**receipt, "model_config_sha256": "0" * 64}, expected) is None
    tampered = json.loads((out / "model-config.json").read_text(encoding="utf-8"))
    tampered["hidden_sizes"] = [999]
    (out / "model-config.json").write_text(json.dumps(tampered), encoding="utf-8")
    assert tools._tenant_selected_config(out, trainer, {**receipt, "model_config_sha256": digest(out / "model-config.json")}, expected) is None
    plain = {key: value for key, value in trainer.items() if key != "search"}
    (out / "model-config.json").write_text("x", encoding="utf-8")
    assert tools._tenant_selected_config(out, plain, {}, expected) is None


def test_selection_report_compares_with_the_default_on_the_holdout(tmp_path: Path) -> None:
    from evomind_runtime.tenant_sizing import classification_metrics, regression_metrics

    root, protocol = _workspace(tmp_path)
    prepared = tenant_training.prepare(root, {"data_path": "inputs/data.csv", "task": "classification", **BASE}, protocol)
    out, _expected, receipt = _card_output(tmp_path, prepared["trainer_config"], 1)
    labels = prepared["evaluation_y"]
    np.savez(out / "baseline-predictions.npz", predictions=np.eye(3)[labels])
    with pytest.raises(ValueError):
        tools._tenant_selection_report(out, prepared, receipt, {}, classification_metrics, regression_metrics)
    receipt["baseline_predictions_sha256"] = digest(out / "baseline-predictions.npz")
    report = tools._tenant_selection_report(out, prepared, receipt, {"hidden_sizes": [1]}, classification_metrics, regression_metrics)
    assert report["mode"] == "auto" and report["trials"] == 2 and report["selected_is_default"] is False
    assert report["default_holdout"]["accuracy"] == 1.0 and report["cv_scores"] == [0.3, 0.4]


def test_training_plan_announces_auto_selection(tmp_path: Path) -> None:
    from evomind_runtime import training_plan

    root, _protocol = _workspace(tmp_path)
    config = root / "work" / "cfg.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"data_path": "inputs/data.csv", "task": "classification"}), encoding="utf-8")
    plan = training_plan.build(root, {"configuration_path": str(config), "configuration_sha256": digest(config)})
    assert plan["model_family"] == "auto" and plan["trials"] >= 2 and "自动比较" in plan["network"]
    config.write_text(json.dumps({"data_path": "inputs/data.csv", "task": "classification", "model_family": "mlp"}), encoding="utf-8")
    plan = training_plan.build(root, {"configuration_path": str(config), "configuration_sha256": digest(config)})
    assert plan["model_family"] == "mlp" and plan["network"] == "全连接神经网络（MLP）"
