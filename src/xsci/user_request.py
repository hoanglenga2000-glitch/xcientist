"""Structured natural-language request protocol for EvoMind.

The terminal and the workstation API both use this parser. It extracts
independent actions before selecting a route, so a deliverable word such as
``report`` cannot hide an execution request in the same sentence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

_TRAIN_ACTIONS = (
    "训练",
    "微调",
    "建模",
    "拟合",
    "跑模型",
    "做实验",
    "开始实验",
    "运行实验",
    "执行实验",
    "完成实验",
    "train",
    "training",
    "fine-tune",
    "fine tune",
    "finetune",
    "fit model",
    "run experiment",
    "execute experiment",
)
_PLAN_ACTIONS = (
    "研究",
    "分析",
    "规划",
    "计划",
    "设计",
    "方案",
    "假设",
    "比较",
    "research",
    "analyze",
    "analyse",
    "plan",
    "design",
    "hypothesis",
    "compare",
)
_DATA_ACTIONS = (
    "检查数据",
    "数据检查",
    "数据审计",
    "数据质量",
    "缺失值",
    "泄漏",
    "标签分布",
    "inspect data",
    "check data",
    "audit data",
    "data quality",
    "missing values",
    "leakage",
)
_REVIEW_ACTIONS = (
    "独立审核",
    "独立审查",
    "审核",
    "审查",
    "复核",
    "review",
    "audit result",
    "judge",
)
_REPORT_ACTIONS = (
    "研究报告",
    "生成报告",
    "输出报告",
    "报告",
    "report",
    "write-up",
    "writeup",
)
_CANDIDATE_SUBMISSION_ACTIONS = (
    "候选 submission",
    "候选submission",
    "submission 候选",
    "submission文件",
    "submission.csv",
    "candidate submission",
    "submission candidate",
)
_OFFICIAL_SUBMIT_ACTIONS = (
    "正式提交",
    "提交 kaggle",
    "提交kaggle",
    "kaggle submit",
    "official submit",
    "submit to kaggle",
)
_LITERATURE_ACTIONS = (
    "检索论文",
    "搜索论文",
    "找论文",
    "查论文",
    "文献检索",
    "参考文献",
    "literature search",
    "search papers",
    "find papers",
)

_NO_TRAIN_PATTERNS = (
    r"(?:先|暂时|目前|现在)?\s*(?:不要|别|不需要|无需|禁止)\s*(?:开始|启动|进行|执行)?\s*(?:任何|任意|新的?|新一轮)?\s*(?:训练|建模|拟合|运行实验|执行实验)",
    r"(?:先|暂时|目前|现在)?\s*(?:不要|别|不需要|无需|禁止)\s*(?:进入|做)?\s*(?:任何|任意)?\s*(?:实际)?\s*(?:训练|建模|拟合|实验执行)",
    r"(?:do not|don't|dont|without)\s+(?:start(?:ing)?\s+)?(?:any\s+)?(?:train(?:ing)?|fit(?:ting)?|run(?:ning)?\s+experiments?)",
)
_NO_LOCAL_GPU_PATTERNS = (
    r"(?:不要(?:使用)?|不使用|禁用|禁止使用|无需)\s*(?:我的|本机|本地)?\s*(?:gpu|显卡)",
    r"(?:no|do not use|don't use|without)\s+local\s+gpu",
)
_NO_REMOTE_COMPUTE_PATTERNS = (
    r"(?:不要(?:使用)?|不使用|禁用|禁止使用|无需)\s*(?:远程|远端)?\s*(?:hpc|集群|服务器|gpu)",
    r"(?:仅|只)\s*(?:使用|用)\s*(?:本机|本地)\s*(?:gpu|显卡|rtx)",
    r"(?:no|do not use|don't use|without)\s+(?:remote\s+)?(?:hpc|cluster|server|gpu)",
    r"(?:local|on-device)\s+(?:gpu\s+)?only",
)
_LOCAL_GPU_PATTERNS = (
    r"(?:仅|只)?\s*(?:使用|用)\s*(?:本机|本地)\s*(?:的\s*)?(?:rtx\s*\d{4}|gpu|显卡)",
    r"(?:本机|本地)\s*(?:rtx\s*\d{4}|gpu|显卡)",
    r"(?:use\s+)?(?:the\s+)?local\s+(?:rtx\s*\d{4}|gpu)",
    r"(?:rtx\s*\d{4}|gpu)\s+(?:locally|on-device)",
)
_NO_OFFICIAL_SUBMIT_PATTERNS = (
    r"(?:先|暂时|目前|现在)?\s*(?:不要|别|不需要|无需|禁止)\s*(?:正式)?\s*(?:提交|上传)(?:\s*到)?\s*(?:kaggle)?",
    r"(?:不要|不)\s*(?:提交|上传)\s*(?:公开)?\s*(?:榜单|排行榜|leaderboard)",
    r"(?:do not|don't|dont|without)\s+(?:officially\s+)?submit(?:ting)?(?:\s+to\s+kaggle)?",
)
_NO_MODEL_PUBLISH_PATTERNS = (
    r"(?:不要|不需要|无需|禁止)\s*(?:发布|上传|推送)(?:\s*模型)?",
    r"(?:do not|don't|dont|without)\s+(?:publish|upload|push)(?:ing)?(?:\s+the)?\s+model",
)
_LLM_TASK_TERMS = (
    "大模型",
    "语言模型",
    "基座模型",
    "领域微调",
    "qlora",
    "lora",
    "peft",
    "llm",
    "language model",
    "foundation model",
    "fine-tune",
    "fine tune",
    "finetune",
)
_EVOMIND_DOC_TERMS = (
    "evomind 文档",
    "evomind文档",
    "系统文档",
    "项目文档",
    "本地文档",
    "evomind docs",
    "project docs",
    "local docs",
)
_SIIM_DATASET = "siim-isic-melanoma-classification"
_SIIM_IMAGE_TERMS = (
    "siim-isic melanoma classification",
    "siim-isic",
    "siim isic",
    "黑色素瘤",
    "皮肤镜",
    "皮肤镜图像",
    "melanoma",
    "dermoscopy",
    "dermoscopic",
)
_SIIM_DELIVERABLES = (
    "evomind-siim-isic-report.pdf",
    "evomind-siim-isic-results.csv",
    "evomind-siim-isic-code.zip",
    "evomind-siim-isic-evidence.zip",
)


def _contains(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _contains_training_action(text: str) -> bool:
    """Distinguish the verb "train" from nouns such as "training set".

    Novice research questions often ask why a training/validation split is
    needed.  A substring match on ``训练`` used to turn those explanations into
    execution requests.  Mask common noun phrases before applying the existing
    bilingual action vocabulary; explicit verbs such as ``训练模型`` remain.
    """

    masked = re.sub(
        r"训练(?:集|数据|样本|标签|结果|指标|历史|记录|过程|日志|证据|文件)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    masked = re.sub(
        r"\btraining\s+(?:set|data|samples?|labels?|results?|metrics?|history|records?|logs?|evidence|files?)\b",
        " ",
        masked,
        flags=re.IGNORECASE,
    )
    if re.search(r"(?:怎么|如何|怎样).{0,12}(?:让|用|叫|请|操作).{0,28}(?:实验|训练)", masked):
        return False
    if re.search(r"(?:做|开始|跑|执行|运行)\s*(?:一次|一轮|个|1\s*次)?\s*(?:机器学习|模型)?\s*实验", masked):
        return True
    return _contains(masked, _TRAIN_ACTIONS)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


@dataclass(frozen=True)
class ComputePolicy:
    backend: str = "auto"
    local_gpu_allowed: bool = False
    remote_gpu_required: bool = False
    remote_root: str = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
    gpu_count: int = 1
    credential_profile: str | None = None
    job_id: int | None = None
    resource_profile: str | None = None


@dataclass(frozen=True)
class SubmissionPolicy:
    create_candidate: bool = False
    official_submission: str = "human_gate"
    model_publication: str = "human_gate"


@dataclass(frozen=True)
class RequestBudget:
    solution_repositories: int = 3
    max_minutes: int = 20
    max_parallel: int = 3
    gpu_count: int = 1


@dataclass
class UserRequest:
    objective: str
    task_type: str = "general"
    dataset: str | None = None
    data_source: str | None = None
    base_model: str | None = None
    training_method: str | None = None
    orchestration_model: str | None = None
    actions: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)
    negative_constraints: list[str] = field(default_factory=list)
    submission_policy: SubmissionPolicy = field(default_factory=SubmissionPolicy)
    budget: RequestBudget = field(default_factory=RequestBudget)
    source_text: str = ""

    @property
    def requests_execution(self) -> bool:
        return "train" in self.actions

    @property
    def requests_research(self) -> bool:
        research_actions = {"research", "inspect_data", "literature", "design", "compare", "review"}
        return bool(research_actions.intersection(self.actions))

    @property
    def route(self) -> str:
        if self.requests_execution:
            return "execution"
        if self.requests_research:
            return "planning"
        return "chat"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserRequest":
        """Restore the exact persisted request contract without reparsing prose."""
        compute = dict(payload.get("compute_policy") or {})
        submission = dict(payload.get("submission_policy") or {})
        budget = dict(payload.get("budget") or {})
        return cls(
            objective=str(payload.get("objective") or payload.get("source_text") or ""),
            task_type=str(payload.get("task_type") or "general"),
            dataset=payload.get("dataset"),
            data_source=payload.get("data_source"),
            base_model=payload.get("base_model"),
            training_method=payload.get("training_method"),
            orchestration_model=payload.get("orchestration_model"),
            actions=[str(item) for item in payload.get("actions") or []],
            deliverables=[str(item) for item in payload.get("deliverables") or []],
            compute_policy=ComputePolicy(
                backend=str(compute.get("backend") or "auto"),
                local_gpu_allowed=bool(compute.get("local_gpu_allowed", False)),
                remote_gpu_required=bool(compute.get("remote_gpu_required", False)),
                remote_root=str(compute.get("remote_root") or ComputePolicy.remote_root),
                gpu_count=int(compute.get("gpu_count") or 1),
                credential_profile=(
                    str(compute.get("credential_profile")) if compute.get("credential_profile") else None
                ),
                job_id=int(compute["job_id"]) if compute.get("job_id") is not None else None,
                resource_profile=(
                    str(compute.get("resource_profile")) if compute.get("resource_profile") else None
                ),
            ),
            negative_constraints=[str(item) for item in payload.get("negative_constraints") or []],
            submission_policy=SubmissionPolicy(
                create_candidate=bool(submission.get("create_candidate", False)),
                official_submission=str(submission.get("official_submission") or "human_gate"),
                model_publication=str(submission.get("model_publication") or "human_gate"),
            ),
            budget=RequestBudget(
                solution_repositories=int(budget.get("solution_repositories") or 3),
                max_minutes=int(budget.get("max_minutes") or 20),
                max_parallel=int(budget.get("max_parallel") or 3),
                gpu_count=int(budget.get("gpu_count") or 1),
            ),
            source_text=str(payload.get("source_text") or payload.get("objective") or ""),
        )


def _extract_dataset(raw: str, low: str) -> str | None:
    if re.search(r"\b(?:siim|isic)\b", low):
        return _SIIM_DATASET
    known = (
        ("siim-isic melanoma classification", _SIIM_DATASET),
        ("siim-isic", _SIIM_DATASET),
        ("siim isic", _SIIM_DATASET),
        ("黑色素瘤", _SIIM_DATASET),
        ("皮肤镜", _SIIM_DATASET),
        ("melanoma", _SIIM_DATASET),
        ("dermoscopy", _SIIM_DATASET),
        ("dermoscopic", _SIIM_DATASET),
        ("fraudtrain.csv", "credit-card-fraud-detection"),
        ("fraudtest.csv", "credit-card-fraud-detection"),
        ("信用卡交易欺诈", "credit-card-fraud-detection"),
        ("信用卡欺诈", "credit-card-fraud-detection"),
        ("credit card fraud", "credit-card-fraud-detection"),
        ("fraud detection", "credit-card-fraud-detection"),
        ("spaceship titanic", "spaceship-titanic"),
        ("spaceship-titanic", "spaceship-titanic"),
        ("泰坦尼克", "titanic"),
        ("titanic", "titanic"),
        ("house prices", "house-prices"),
        ("house-prices", "house-prices"),
        ("房价", "house-prices"),
    )
    for needle, value in known:
        if needle in low:
            return value
    path_match = re.search(r"(?P<path>(?:[A-Za-z]:[\\/]|\.?\.?[\\/])[^\s\"']+\.(?:csv|parquet|jsonl?))", raw)
    if path_match:
        return path_match.group("path")
    local_match = re.search(r"(?:本地已有的?|local)\s*([\w.-]+)\s*(?:数据|dataset)", low, flags=re.IGNORECASE)
    return local_match.group(1) if local_match else None


def _extract_image_task_type(low: str) -> str | None:
    if any(term in low for term in _SIIM_IMAGE_TERMS) or re.search(r"\b(?:siim|isic)\b", low):
        return "image_classification"
    return None


def _extract_tabular_task_type(low: str) -> str | None:
    classification_terms = (
        "fraudtrain.csv",
        "fraudtest.csv",
        "信用卡交易欺诈",
        "信用卡欺诈",
        "credit card fraud",
        "fraud detection",
        "二分类",
        "binary classification",
    )
    if any(term in low for term in classification_terms):
        return "tabular_classification"
    return None


def _extract_orchestration_model(low: str) -> str | None:
    if any(term in low for term in ("gpt-5.6", "gpt 5.6", "gpt5.6", "gpt‑5.6")):
        return "gpt-5.6-sol"
    return None


def _extract_llm_fields(low: str) -> tuple[str, str | None, str | None, str | None]:
    if not _contains(low, _LLM_TASK_TERMS):
        return "general", None, None, None
    if "qwen2.5-7b-instruct" in low or "qwen 2.5 7b" in low or "qwen2.5 7b" in low:
        base_model = "Qwen/Qwen2.5-7B-Instruct"
    elif "qwen" in low or "7b" in low or "70 亿" in low or "70亿" in low:
        base_model = "Qwen/Qwen2.5-7B-Instruct"
    else:
        base_model = "Qwen/Qwen2.5-7B-Instruct"
    method = "qlora_4bit" if any(term in low for term in ("qlora", "4-bit", "4bit", "4 位", "4位")) else "qlora_4bit"
    data_source = "evomind_docs" if _contains(low, _EVOMIND_DOC_TERMS) else None
    return "llm_finetune", data_source, base_model, method


def _parse_positive_int(low: str, patterns: tuple[str, ...], default: int, *, maximum: int) -> int:
    for pattern in patterns:
        match = re.search(pattern, low, flags=re.IGNORECASE)
        if match:
            return max(1, min(int(match.group(1)), maximum))
    return default


def parse_user_request(text: str) -> UserRequest:
    """Parse one user turn without calling a model or executing any action."""
    raw = (text or "").strip()
    low = raw.lower()
    actions: list[str] = []
    deliverables: list[str] = []
    negatives: list[str] = []

    no_train = _matches(low, _NO_TRAIN_PATTERNS)
    no_local_gpu = _matches(low, _NO_LOCAL_GPU_PATTERNS)
    no_remote_compute = _matches(low, _NO_REMOTE_COMPUTE_PATTERNS)
    local_gpu_requested = _matches(low, _LOCAL_GPU_PATTERNS)
    no_submit = _matches(low, _NO_OFFICIAL_SUBMIT_PATTERNS)
    no_publish = _matches(low, _NO_MODEL_PUBLISH_PATTERNS)
    task_type, data_source, base_model, training_method = _extract_llm_fields(low)
    if task_type == "general":
        task_type = _extract_image_task_type(low) or _extract_tabular_task_type(low) or task_type
    dataset = _extract_dataset(raw, low)
    is_siim = task_type == "image_classification" and dataset == _SIIM_DATASET
    orchestration_model = _extract_orchestration_model(low)

    if _contains(low, _DATA_ACTIONS):
        _append_unique(actions, "inspect_data")
    if _contains(low, _LITERATURE_ACTIONS):
        _append_unique(actions, "literature")
    if _contains(low, _PLAN_ACTIONS):
        _append_unique(actions, "research")
        _append_unique(actions, "design")
    if any(term in low for term in ("比较", "对比", "compare", "benchmark", "多个方案")):
        _append_unique(actions, "compare")
    if _contains_training_action(low) and not no_train:
        _append_unique(actions, "train")
    if _contains(low, _REVIEW_ACTIONS):
        _append_unique(actions, "review")
    if _contains(low, _REPORT_ACTIONS):
        _append_unique(actions, "report")
        _append_unique(deliverables, "research_report")
    if _contains(low, _CANDIDATE_SUBMISSION_ACTIONS):
        _append_unique(actions, "candidate_submission")
        _append_unique(deliverables, "submission.csv")
    if "指标" in low or "metrics" in low:
        _append_unique(deliverables, "metrics.json")
    if "证据" in low or "artifact" in low or "产物" in low:
        _append_unique(deliverables, "artifact_manifest.json")
    if task_type == "llm_finetune":
        if any(term in low for term in ("适配器", "adapter", "lora")):
            _append_unique(deliverables, "adapter")
        if any(term in low for term in ("模型卡", "model card", "model_card")):
            _append_unique(deliverables, "model_card.md")
        if "metrics.json" not in deliverables:
            _append_unique(deliverables, "metrics.json")
        if "artifact_manifest.json" not in deliverables:
            _append_unique(deliverables, "artifact_manifest.json")
    if is_siim:
        for deliverable in _SIIM_DELIVERABLES:
            _append_unique(deliverables, deliverable)
        _append_unique(deliverables, "submission.csv")
        _append_unique(deliverables, "artifact_manifest.json")
    if _contains(low, _OFFICIAL_SUBMIT_ACTIONS) and not no_submit:
        _append_unique(actions, "official_submit")

    if no_train:
        negatives.append("no_training")
    if no_local_gpu:
        negatives.append("no_local_gpu")
    if no_remote_compute:
        negatives.append("no_remote_compute")
    if no_submit:
        negatives.append("no_official_submission")
    if no_publish:
        negatives.append("no_model_publication")

    mentions_hpc = any(term in low for term in ("hpc", "服务器", "远端", "远程", "remote gpu", "cluster", "集群"))
    remote_requested = mentions_hpc and not no_remote_compute
    if is_siim:
        backend = "hpc"
    elif local_gpu_requested and not no_local_gpu:
        backend = "local_gpu"
    elif remote_requested or ("train" in actions and no_local_gpu):
        backend = "hpc"
    else:
        backend = "auto"
    compute = ComputePolicy(
        backend=backend,
        local_gpu_allowed=(backend == "local_gpu" and not is_siim),
        remote_gpu_required=(backend == "hpc" and ("train" in actions or is_siim)),
        gpu_count=_parse_positive_int(low, (r"(\d+)\s*张\s*gpu", r"(\d+)\s*gpus?"), 1, maximum=8),
    )
    budget = RequestBudget(
        solution_repositories=_parse_positive_int(
            low,
            (r"(\d+)\s*(?:个|路)?\s*(?:solution\s*)?(?:repositories|repository|仓库|方案)",),
            3,
            maximum=7,
        ),
        max_minutes=_parse_positive_int(
            low,
            (r"(\d+)\s*(?:分钟|minutes?|mins?)",),
            1440 if is_siim else (45 if task_type == "llm_finetune" else 20),
            maximum=1440,
        ),
        max_parallel=_parse_positive_int(low, (r"(?:并发|parallel)\s*(\d+)",), 3, maximum=8),
        gpu_count=compute.gpu_count,
    )
    submission = SubmissionPolicy(
        create_candidate="candidate_submission" in actions or is_siim,
        official_submission="forbidden" if no_submit or is_siim else "human_gate",
        model_publication="forbidden" if no_publish else "human_gate",
    )
    if is_siim and "no_official_submission" not in negatives:
        negatives.append("no_official_submission")
    if is_siim and "official_submit" in actions:
        actions.remove("official_submit")
    return UserRequest(
        objective=raw,
        task_type=task_type,
        dataset=dataset,
        data_source=data_source,
        base_model=base_model,
        training_method=training_method,
        orchestration_model=orchestration_model,
        actions=actions,
        deliverables=deliverables,
        compute_policy=compute,
        negative_constraints=negatives,
        submission_policy=submission,
        budget=budget,
        source_text=raw,
    )


__all__ = [
    "ComputePolicy",
    "RequestBudget",
    "SubmissionPolicy",
    "UserRequest",
    "parse_user_request",
]
