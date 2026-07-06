"""Modality-generality guards: the engine must adapt its proposal guidance to
image / text / multimodal / audio, not just tabular. These lock in that the
solution contract, system prompt, modality guidance, base-mode instruction, and
strategy selector are all modality-aware (not tabular-only)."""
from research_os.variation_generator import (
    TaskContext,
    _solution_contract,
    _system_prompt,
    _modality_guidance,
    _build_user_prompt,
)
from research_os.strategy_selector import recommend_strategies, TaskProfile


def _ctx(modality: str) -> TaskContext:
    return TaskContext(
        task_name="t", modality=modality, task_type="classification",
        metric="accuracy", metric_direction="maximize",
        target_column="y", id_column="id", data_schema="", n_train=50000, n_test=10000,
    )


# ---- solution contract is modality-aware (the big one) --------------------
def test_dl_modalities_contract_allows_torch_and_gpu():
    for m in ("image", "multimodal", "audio"):
        c = _solution_contract(m).lower()
        assert "torch" in c, f"{m} contract must allow torch"
        assert "gpu" in c or "cuda" in c, f"{m} contract must mention GPU/CUDA"
        # must NOT keep the tabular CPU-only tree-only restriction
        assert "only these libraries: pandas, numpy, scikit-learn, lightgbm" not in c, \
            f"{m} contract must not restrict to tabular-only libs"


def test_tabular_contract_stays_cpu_tree_only():
    c = _solution_contract("tabular").lower()
    assert "only these libraries" in c
    assert "torch" not in c
    assert "cpu" in c


def test_text_contract_permits_optional_transformers():
    c = _solution_contract("text").lower()
    # text defaults to TF-IDF but may escalate to transformers
    assert "transformers" in c or "torch" in c


# ---- modality guidance covers multimodal + audio -------------------------
def test_modality_guidance_has_multimodal_branch():
    g = _modality_guidance("multimodal").lower()
    assert "fus" in g  # fuse / fusion
    assert "torch" in g and ("gpu" in g or "cuda" in g)


def test_modality_guidance_has_audio_branch():
    g = _modality_guidance("audio").lower()
    assert "spectrogram" in g or "mel" in g
    assert "torch" in g and ("gpu" in g or "cuda" in g)


# ---- system prompt reflects modality -------------------------------------
def test_system_prompt_is_modality_aware():
    assert "torch" in _system_prompt("image").lower()
    assert "torch" not in _system_prompt("tabular").lower()


# ---- base-mode instruction not tabular-biased for DL modalities ----------
def test_base_mode_not_tabular_biased_for_image():
    p_img = _build_user_prompt(_ctx("image"), mode="Base", cv_history=[], lessons=[],
                               strategies=[], best_code=None)
    p_tab = _build_user_prompt(_ctx("tabular"), mode="Base", cv_history=[], lessons=[],
                               strategies=[], best_code=None)
    assert "gradient-boosted model (lightgbm" not in p_img.lower(), \
        "image baseline must not be told to prefer LightGBM"
    assert "gradient-boosted" in p_tab.lower(), "tabular baseline should still prefer GBM"


# ---- strategy selector recommends modality-appropriate strategies --------
def test_selector_multimodal_gets_fusion():
    r = recommend_strategies(TaskProfile(modality="multimodal", task_type="classification",
                                         train_size=50000, n_features=10))
    assert "multimodal_fusion" in r.strategies


def test_selector_audio_gets_spectrogram():
    r = recommend_strategies(TaskProfile(modality="audio", task_type="classification",
                                         train_size=50000))
    assert "audio_spectrogram" in r.strategies


def test_selector_text_gets_text_strategy():
    r = recommend_strategies(TaskProfile(modality="text", task_type="classification",
                                         train_size=50000))
    assert "tfidf_ngrams" in r.strategies


def test_selector_image_still_gets_tta_regression_guard():
    r = recommend_strategies(TaskProfile(modality="image", task_type="classification",
                                         train_size=42000))
    assert "test_time_augmentation" in r.strategies
