from __future__ import annotations

from xsci import kaggle_intent as intent
from xsci.user_request import ComputePolicy, UserRequest, parse_user_request

TITANIC_REQUEST = (
    "请用本地已有的 Titanic 数据完成一个小型二分类模型开发任务：自动检查数据、提出并比较方案、"
    "在 HPC 上训练、独立审核、生成候选 submission 和研究报告；不要使用本地 GPU，也不要提交 Kaggle。"
)
LLM_REQUEST = (
    "请用本地 EvoMind 文档，让一个成熟的 7B 中文大模型更懂我们的科研工作流；自动准备数据，"
    "在远程 A40 上完成训练和评测，并生成可下载的适配器、模型卡和报告。"
    "不要使用本地 GPU，也不要发布模型。"
)


def test_compound_execution_is_not_stolen_by_status_or_report_words():
    request = parse_user_request(TITANIC_REQUEST)

    assert request.dataset == "titanic"
    assert request.requests_execution is True
    assert request.compute_policy.backend == "hpc"
    assert request.compute_policy.local_gpu_allowed is False
    assert request.compute_policy.remote_gpu_required is True
    assert request.submission_policy.create_candidate is True
    assert request.submission_policy.official_submission == "forbidden"
    assert {"inspect_data", "design", "compare", "train", "review", "report"}.issubset(request.actions)
    assert intent.classify(TITANIC_REQUEST).kind == intent.EXECUTION


def test_no_training_keeps_research_and_planning_open():
    text = "请检查数据、比较三个方案并设计实验，但先不要训练，给出指标和报告结构。"
    request = parse_user_request(text)

    assert request.requests_execution is False
    assert request.requests_research is True
    assert "no_training" in request.negative_constraints
    assert intent.classify(text).kind == intent.PLANNING


def test_novice_status_question_with_colloquial_negations_is_read_only():
    text = (
        "你好，我第一次用这个，不懂专业术语。"
        "先别训练，也别提交。"
        "你能简单告诉我，上次做的实验是什么，现在做完了吗？最多用六句话。"
    )
    request = parse_user_request(text)

    assert request.requests_execution is False
    assert "no_training" in request.negative_constraints
    assert "no_official_submission" in request.negative_constraints
    assert intent.classify(text).kind != intent.EXECUTION


def test_progress_query_does_not_start_training():
    assert intent.classify("训练结果怎么样").kind == intent.TOOL_QUERY
    assert intent.classify("训练好了吗").kind == intent.TOOL_QUERY


def test_training_set_nouns_do_not_become_execution_actions():
    text = "为什么训练集和验证集必须按患者分组？请结合上次结果解释。"
    request = parse_user_request(text)

    assert request.requests_execution is False
    assert "train" not in request.actions
    assert intent.classify(text).kind != intent.EXECUTION


def test_any_training_negation_is_preserved_as_read_only():
    text = "我是小白，上次实验留下了哪些文件？不要启动任何训练。"
    request = parse_user_request(text)

    assert request.requests_execution is False
    assert "train" not in request.actions
    assert "no_training" in request.negative_constraints
    assert intent.classify(text).kind != intent.EXECUTION


def test_explicit_training_action_remains_execution():
    text = "请在 A800 上训练 SIIM-ISIC 图像分类模型并生成报告，不提交 Kaggle。"
    request = parse_user_request(text)

    assert request.requests_execution is True
    assert request.dataset == "siim-isic-melanoma-classification"
    assert request.task_type == "image_classification"
    assert intent.classify(text).kind == intent.EXECUTION


def test_request_budget_and_submission_policy_are_bounded():
    request = parse_user_request(
        "用 HPC 上 2 张 GPU 并发 5 跑 9 个方案，最多 45 分钟，训练并生成 submission.csv，不要提交 Kaggle"
    )

    assert request.budget.solution_repositories == 7
    assert request.budget.max_parallel == 5
    assert request.budget.max_minutes == 45
    assert request.budget.gpu_count == 2
    assert request.submission_policy.official_submission == "forbidden"


def test_llm_finetune_request_has_explicit_model_data_and_release_policy():
    request = parse_user_request(LLM_REQUEST)

    assert request.task_type == "llm_finetune"
    assert request.data_source == "evomind_docs"
    assert request.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert request.training_method == "qlora_4bit"
    assert request.requests_execution is True
    assert request.compute_policy.backend == "hpc"
    assert request.compute_policy.local_gpu_allowed is False
    assert request.compute_policy.remote_gpu_required is True
    assert request.submission_policy.model_publication == "forbidden"
    assert "no_model_publication" in request.negative_constraints
    assert {"adapter", "model_card.md", "metrics.json", "artifact_manifest.json", "research_report"}.issubset(
        request.deliverables
    )


def test_persisted_request_round_trip_preserves_execution_budget_and_policies():
    request = parse_user_request(LLM_REQUEST + "最多 20 分钟。")

    restored = UserRequest.from_dict(request.to_dict())

    assert restored.to_dict() == request.to_dict()
    assert restored.budget.max_minutes == 20
    assert restored.compute_policy.local_gpu_allowed is False
    assert restored.submission_policy.model_publication == "forbidden"


def test_persisted_compute_policy_keeps_named_hpc_allocation_binding():
    request = parse_user_request(TITANIC_REQUEST)
    request.compute_policy = ComputePolicy(
        backend="hpc",
        local_gpu_allowed=False,
        remote_gpu_required=True,
        gpu_count=1,
        credential_profile="job90353",
        job_id=90353,
    )

    restored = UserRequest.from_dict(request.to_dict())

    assert restored.compute_policy.credential_profile == "job90353"
    assert restored.compute_policy.job_id == 90353


def test_current_finetune_status_is_read_only_not_execution():
    result = intent.classify("上次模型微调完成到哪了")

    assert result.kind == intent.TOOL_QUERY
    assert result.payload == "current_run"


def test_local_environment_routes_to_read_only_probe():
    result = intent.classify("检查本地开发环境")

    assert result.kind == intent.TOOL_QUERY
    assert result.payload == "local_environment"


def test_previous_model_refinement_routes_to_gate_plan():
    vague = intent.classify("继续优化上次模型")
    explicit = intent.classify("把学习率降低一半，继续训练 1 个周期并更新报告")
    approve = intent.classify("批准微调")

    assert (vague.kind, vague.payload) == (intent.TOOL_QUERY, "llm_refinement")
    assert (explicit.kind, explicit.payload) == (intent.TOOL_QUERY, "llm_refinement")
    assert (approve.kind, approve.payload) == (intent.TOOL_QUERY, "llm_refinement_approve")


def test_credit_card_fraud_request_uses_local_gpu_and_never_flips_negated_remote_cluster():
    request = parse_user_request(
        "请创建一个完整的信用卡交易欺诈检测研究任务。数据源使用 fraudTrain.csv 和 fraudTest.csv（约 501.59 MB、23 列）；"
        "推理与科研编排使用 GPT-5.6；计算仅使用本机 RTX 4060 8GB，不使用远程集群。"
        "自动完成数据审计、泄漏检查、时间切分、类别不平衡策略、LightGBM、CatBoost 与 XGBoost 候选方案、"
        "训练监控、PR-AUC、F1 与 Recall 评估、独立复核、证据链、研究报告和可复现产物。"
        "先执行本地资源门和人工 Gate，再开始训练；不提交公开榜单。"
    )

    assert request.dataset == "credit-card-fraud-detection"
    assert request.task_type == "tabular_classification"
    assert request.orchestration_model == "gpt-5.6-sol"
    assert request.requests_execution is True
    assert request.compute_policy.backend == "local_gpu"
    assert request.compute_policy.local_gpu_allowed is True
    assert request.compute_policy.remote_gpu_required is False
    assert "no_remote_compute" in request.negative_constraints
    assert request.submission_policy.official_submission == "forbidden"
    assert {"inspect_data", "research", "design", "train", "review", "report"}.issubset(request.actions)
