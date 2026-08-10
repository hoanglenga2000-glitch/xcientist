from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from research_os.experience_mcgs import (
    CANONICAL_HASH_SCHEMA,
    SUMMARY_CACHE_SCHEMA,
    SUMMARY_ENTRY_SCHEMA,
    BudgetLedger,
    ExperienceBoard,
    ExperienceCardBuilder,
    ExperienceCost,
    LazySummaryCache,
    SearchOperator,
    canonical_json,
    retrieve_experience,
    route_operator,
    select_experience_parent,
    shadow_replay,
    structural_features,
)

CANONICAL_FIXTURE = Path(__file__).parent / "fixtures" / "canonical_json_f64_v1.json"


def card(
    node: str,
    *,
    parent: str = "",
    score: float | None = 0.5,
    family: str = "tree",
    status: str = "success",
    error: str = "",
    provenance=None,
    summary: str = "",
    cost: ExperienceCost | None = None,
):
    return ExperienceCardBuilder.build(
        task_id="task",
        node_id=node,
        parent_ids=[parent] if parent else [],
        operator=SearchOperator.DRAFT if not parent else SearchOperator.IMPROVE,
        method_family=family,
        code=f"def solve_{node.lower()}():\n    return {score!r}\n",
        data_hashes=["a" * 64],
        prompt="stable prompt",
        execution_id=node,
        status=status,
        public_validation_score=score,
        error=error,
        provenance=provenance or {"split": "public_validation", "changes_summary": node},
        summary=summary,
        cost=cost,
    )


def structured_summary(method: str = "Gradient boosted trees", comparison: str = "Improves its parent by adding monotonic features"):
    return {
        "method_overview": method,
        "parent_comparison_experience": comparison,
    }


def test_card_builder_is_canonical_and_rejects_private_feedback_keys_and_values():
    first = card("EXP000")
    second = card("EXP000")
    assert first == second
    assert first.card_id == second.card_id
    assert first.code_hash == second.code_hash
    assert first.card_id == f"exp_{first.content_hash[:24]}"

    with pytest.raises(ValueError, match="private evaluation field"):
        card("BADKEY", provenance={"nested": {"private_grader_score": 0.9}})
    with pytest.raises(ValueError, match="private evaluation value"):
        card("BADVALUE", provenance={"notes": [{"text": "private grader feedback: excellent"}]})
    with pytest.raises(ValueError, match="private evaluation value"):
        card("BADSPLIT", provenance={"split": "private"})
    with pytest.raises(ValueError, match="private evaluation field"):
        card("BADNESTEDKEY", provenance={"private": {"score": 0.9}})
    with pytest.raises(ValueError, match="private evaluation field"):
        card("BADCAMELKEY", provenance={"privateGraderFeedback": "hidden"})
    with pytest.raises(ValueError, match="private evaluation value"):
        card("BADLIST", provenance={"notes": ["public", "official medal result"]})
    with pytest.raises(ValueError, match="private evaluation value"):
        card("BADSUMMARY", summary="Copied from the private score response")


def test_cross_language_canonical_number_fixture_is_byte_stable():
    fixture = json.loads(CANONICAL_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["canonical_hash_schema"] == CANONICAL_HASH_SCHEMA
    observed: dict[str, str] = {}
    for case in fixture["cases"]:
        encoded = canonical_json(case["value"])
        observed[case["name"]] = encoded
        assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == case["expected_sha256"]
        if "expected_canonical" in case:
            assert encoded == case["expected_canonical"]
    for group in fixture["equivalence_groups"]:
        assert len({observed[name] for name in group}) == 1


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), 2**53, float(2**53)],
)
def test_canonical_number_contract_rejects_non_finite_and_unsafe_values(value):
    with pytest.raises(ValueError):
        canonical_json(value)


def test_canonical_contract_rejects_cycles_and_non_json_values():
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic JSON array"):
        canonical_json(cyclic)
    with pytest.raises(TypeError, match="sets are not canonical JSON values"):
        canonical_json({"not_json": {1, 2}})


def test_card_provenance_is_deeply_immutable_and_board_hash_cannot_drift():
    source = {
        "split": "public_validation",
        "nested": {"steps": [{"name": "fit"}, {"name": "validate"}]},
    }
    item = card("EXP000", provenance=source)
    reordered = card("EXP000", provenance={
        "nested": {"steps": [{"name": "fit"}, {"name": "validate"}]},
        "split": "public_validation",
    })
    assert reordered.card_id == item.card_id
    assert reordered.content_hash == item.content_hash
    board = ExperienceBoard("task")
    board.append(item)
    original_hash = board.to_dict()["board_hash"]

    # Mutating the caller-owned input cannot mutate canonical card content.
    source["nested"]["steps"][0]["name"] = "tampered"
    source["nested"]["steps"].append({"name": "extra"})
    assert item.to_dict()["provenance"]["nested"]["steps"] == [
        {"name": "fit"},
        {"name": "validate"},
    ]

    # Every nested container held by the card is immutable.
    with pytest.raises(TypeError):
        item.provenance["nested"]["new"] = "value"
    with pytest.raises(TypeError):
        item.provenance["nested"]["steps"][0]["name"] = "value"
    with pytest.raises(AttributeError):
        item.provenance["nested"]["steps"].append("value")

    # Serialization is detached too.
    detached = board.to_dict()
    detached["cards"][0]["provenance"]["nested"]["steps"][0]["name"] = "changed"
    assert board.to_dict()["board_hash"] == original_hash


def test_board_is_append_only_hash_deduplicated_and_retains_node_history(tmp_path):
    board = ExperienceBoard("task")
    first = card("EXP000", summary="draft observation")
    verified = card("EXP000", summary="verified observation")

    assert board.append(first) is True
    assert board.append(first) is False
    assert board.cards.pop(first.card_id) is first  # compatibility peek, not deletion
    assert first.card_id in board.cards
    with pytest.raises(TypeError):
        board.cards[first.card_id] = verified

    assert verified.card_id != first.card_id
    assert board.append(verified) is True
    assert tuple(board.cards) == (first.card_id, verified.card_id)
    assert board.by_node("EXP000") is verified
    assert board.to_dict()["append_order"] == [first.card_id, verified.card_id]

    with pytest.raises(ValueError, match="collision"):
        board.append(replace(first, summary="mutated with stale id"))
    with pytest.raises(ValueError, match="content hash mismatch"):
        ExperienceBoard("task").append(replace(first, summary="mutated with stale id"))

    before = board.to_dict()["board_hash"]
    path = board.write(tmp_path / "board.json")
    assert path.is_file()
    assert board.to_dict()["board_hash"] == before
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["append_only"] is True
    assert persisted["deduplication_key"] == "canonical_card_hash"
    assert persisted["hash_canonicalization"] == CANONICAL_HASH_SCHEMA
    assert persisted["append_chain_head"] != "0" * 64


def test_task_global_board_aggregation_is_rich_and_deterministic():
    board = ExperienceBoard("task")
    board.append(card("EXP000", score=0.50, cost=ExperienceCost(total_tokens=10, wall_seconds=1.5)))
    board.append(card("EXP001", parent="EXP000", score=0.70, family="linear", cost=ExperienceCost(total_tokens=20, wall_seconds=2.5)))
    board.append(card("EXP002", parent="EXP001", score=None, status="failed", error="ValueError: bad input"))

    aggregate = board.aggregate()
    assert aggregate["scope"] == "task_global"
    assert aggregate["card_count"] == 3
    assert aggregate["unique_node_count"] == 3
    assert aggregate["operator_counts"] == {"Draft": 1, "Improve": 2}
    assert aggregate["method_family_counts"] == {"linear": 1, "tree": 2}
    assert aggregate["status_counts"] == {"failed": 1, "success": 2}
    assert aggregate["lineage"]["edge_count"] == 2
    assert aggregate["public_validation"]["best_node_id"] == "EXP001"
    assert aggregate["public_validation"]["new_best_count"] == 2
    assert aggregate["cost"]["total_tokens"] == 30
    assert aggregate["cost"]["wall_seconds"] == 4.0
    assert board.to_dict()["task_global_aggregation"] == aggregate
    assert board.to_dict() == board.to_dict()


def test_deterministic_ast_shingles_distinguish_structure_not_identifier_spelling():
    left = "def solve(a, b, c):\n    result = a + b * c\n    return result\n"
    same_shape = "def model(x, y, z):\n    output = x + y * z\n    return output\n"
    different_shape = "def solve(a, b, c):\n    result = (a + b) * c\n    return result\n"

    first = structural_features(left, SearchOperator.IMPROVE, "tree")
    assert first == structural_features(left, SearchOperator.IMPROVE, "tree")
    assert first == structural_features(same_shape, SearchOperator.IMPROVE, "tree")
    other = structural_features(different_shape, SearchOperator.IMPROVE, "tree")
    assert first != other
    assert {item for item in first if item.startswith("ast_shingle:")} != {
        item for item in other if item.startswith("ast_shingle:")
    }


def test_hybrid_selection_is_deterministic_and_direction_aware():
    board = ExperienceBoard("task")
    for item in [card("EXP000", score=0.50), card("EXP001", parent="EXP000", score=0.70), card("EXP002", parent="EXP000", score=0.60, family="linear")]:
        board.append(item)
    trace = select_experience_parent(board, visits={"EXP000": 5, "EXP001": 2, "EXP002": 2})
    assert trace.selected_node_id == "EXP001"
    assert trace == select_experience_parent(board, visits={"EXP000": 5, "EXP001": 2, "EXP002": 2})
    assert trace.candidates[0].utility >= trace.candidates[1].utility


def test_operator_router_covers_all_four_operations():
    empty = ExperienceBoard("task")
    assert route_operator(empty) is SearchOperator.DRAFT
    failed = ExperienceBoard("task")
    failed.append(card("EXP000", score=None, status="failed", error="ValueError: bad column"))
    assert route_operator(failed) is SearchOperator.DEBUG
    improve = ExperienceBoard("task")
    improve.append(card("EXP000"))
    assert route_operator(improve) is SearchOperator.IMPROVE
    improve.append(card("EXP001", parent="EXP000", family="linear", score=0.6))
    assert route_operator(improve, no_new_best_expansions=3) is SearchOperator.CROSSOVER


def test_operator_conditioned_retrieval_is_bounded():
    board = ExperienceBoard("task")
    board.append(card("EXP000", family="tree"))
    board.append(card("EXP001", parent="EXP000", family="linear", score=0.6))
    board.append(card("EXP002", parent="EXP001", family="neural", score=0.7))
    bundle = retrieve_experience(board, SearchOperator.CROSSOVER, selected_node_id="EXP002", max_cards=2, max_tokens=6000)
    assert len(bundle.card_ids) == 2
    assert bundle.truncated_by == "card_limit"
    assert bundle.cache_key


def test_retrieval_budget_uses_prompt_projection_not_full_audit_card():
    board = ExperienceBoard("task")
    item = card(
        "EXP000",
        provenance={
            "split": "public_validation",
            "changes_summary": "compact tree baseline",
            # Large public audit evidence belongs in the canonical card and
            # hash chain, but it is not copied verbatim into the prompt.
            "artifact_hashes": [hashlib.sha256(str(index).encode()).hexdigest() for index in range(1_000)],
        },
    )
    board.append(item)
    assert len(json.dumps(item.to_dict(), sort_keys=True)) > 24_000

    bundle = retrieve_experience(
        board,
        SearchOperator.IMPROVE,
        selected_node_id="EXP000",
        max_cards=8,
        max_tokens=100,
    )

    assert bundle.card_ids == (item.card_id,)
    assert 0 < bundle.estimated_tokens <= 100
    assert bundle.truncated_by == "none"


def test_lazy_summary_cache_has_strict_schema_and_calls_summarizer_once(tmp_path):
    path = tmp_path / "summaries.json"
    cache = LazySummaryCache(path)
    calls = []
    item = card("EXP000")
    first, hit1 = cache.get_or_create(
        item,
        summarizer_model="deterministic",
        prompt_template_hash="b" * 64,
        summarize=lambda value: calls.append(value.card_id) or structured_summary(),
    )
    second, hit2 = cache.get_or_create(
        item,
        summarizer_model="deterministic",
        prompt_template_hash="b" * 64,
        summarize=lambda value: structured_summary("different", "different"),
    )
    assert (first, hit1) == (structured_summary(), False)
    assert (second, hit2) == (structured_summary(), True)
    assert calls == [item.card_id]

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "entry_schema", "entries"}
    assert payload["schema"] == SUMMARY_CACHE_SCHEMA
    assert payload["entry_schema"] == SUMMARY_ENTRY_SCHEMA
    key = cache.cache_key(item, "deterministic", "b" * 64)
    entry = payload["entries"][key]
    assert set(entry) == {
        "schema", "card_hash", "summarizer_model", "prompt_template_hash", "temperature",
        "method_overview", "parent_comparison_experience",
    }
    assert entry["card_hash"] == item.content_hash
    assert entry["temperature"] == 0
    assert entry["method_overview"] == first["method_overview"]
    assert entry["parent_comparison_experience"] == first["parent_comparison_experience"]
    with pytest.raises(TypeError):
        cache.entries[key]["method_overview"] = "poison"


def test_lazy_summary_cache_key_is_deterministic_and_schema_validation_is_fail_closed(tmp_path):
    item = card("EXP000")
    twin = card("EXP000")
    key = LazySummaryCache.cache_key(item, "model", "c" * 64)
    legacy_contract = hashlib.sha256(json.dumps({
        "schema": item.schema,
        "card_id": item.card_id,
        "summarizer_model": "model",
        "prompt_template_hash": "c" * 64,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert key == legacy_contract
    assert key == LazySummaryCache.cache_key(twin, "model", "c" * 64)
    assert key != LazySummaryCache.cache_key(item, "other-model", "c" * 64)
    assert key != LazySummaryCache.cache_key(item, "model", "d" * 64)

    cache = LazySummaryCache(tmp_path / "strict.json")
    with pytest.raises(ValueError, match="64-character"):
        cache.get_or_create(
            item,
            summarizer_model="model",
            prompt_template_hash="not-a-hash",
            summarize=lambda _: structured_summary(),
        )
    with pytest.raises(TypeError, match="summary must be a mapping"):
        cache.get_or_create(
            item,
            summarizer_model="model",
            prompt_template_hash="c" * 64,
            summarize=lambda _: "free-form string",
        )
    with pytest.raises(ValueError, match="exactly"):
        cache.get_or_create(
            item,
            summarizer_model="model",
            prompt_template_hash="c" * 64,
            summarize=lambda _: {**structured_summary(), "extra": "field"},
        )
    with pytest.raises(ValueError, match="non-empty"):
        cache.get_or_create(
            item,
            summarizer_model="model",
            prompt_template_hash="c" * 64,
            summarize=lambda _: structured_summary(comparison=""),
        )
    with pytest.raises(ValueError, match="private evaluation value"):
        cache.get_or_create(
            item,
            summarizer_model="model",
            prompt_template_hash="c" * 64,
            summarize=lambda _: structured_summary(comparison="Uses private grader feedback"),
        )


def test_poisoned_summary_cache_entry_is_never_returned_as_a_hit(tmp_path):
    path = tmp_path / "poisoned.json"
    item = card("EXP000")
    model = "model"
    template = "e" * 64
    key = LazySummaryCache.cache_key(item, model, template)
    path.write_text(json.dumps({
        "schema": SUMMARY_CACHE_SCHEMA,
        "entry_schema": SUMMARY_ENTRY_SCHEMA,
        "entries": {
            key: {
                "schema": SUMMARY_ENTRY_SCHEMA,
                "card_hash": item.content_hash,
                "summarizer_model": model,
                "prompt_template_hash": template,
                "temperature": 0,
                "method_overview": "Tree model",
                "parent_comparison_experience": "Copied private grader score 0.9",
            }
        },
    }), encoding="utf-8")

    cache = LazySummaryCache(path)
    calls = []
    result, hit = cache.get_or_create(
        item,
        summarizer_model=model,
        prompt_template_hash=template,
        summarize=lambda _: calls.append(True) or structured_summary(),
    )
    assert hit is False
    assert result == structured_summary()
    assert calls == [True]


def test_budget_ledger_fails_closed_at_hard_limit():
    ledger = BudgetLedger(max_nodes=2, max_total_tokens=100, max_wall_seconds=10)
    ledger.record(ExperienceCost(prompt_tokens=20, completion_tokens=10, total_tokens=30, wall_seconds=2))
    assert ledger.can_continue
    ledger.record(ExperienceCost(prompt_tokens=50, completion_tokens=20, total_tokens=70, wall_seconds=2))
    assert not ledger.can_continue
    assert ledger.terminal_reason in {"node_budget_exhausted", "token_budget_exhausted"}


def test_shadow_replay_has_no_llm_or_grader_and_is_reproducible():
    board = ExperienceBoard("task")
    board.append(card("EXP000"))
    board.append(card("EXP001", parent="EXP000", score=0.7))
    replay = shadow_replay(board, visits={"EXP000": 2, "EXP001": 1})
    assert replay["deterministic"] is True
    assert replay["private_feedback_absent"] is True
    assert replay["mode"] == "shadow_replay_no_llm_no_grader"
