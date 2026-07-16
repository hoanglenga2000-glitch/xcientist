from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "web" / "dashboard"


PLAN_REQUIREMENTS = [
    {
        "id": "task_understanding",
        "title": "任务理解",
        "description": "读取 overview、目标列、评价指标、文件结构和提交格式。",
        "evidence": ["task_scaffold.json", "workflow_stage_audit.json"],
    },
    {
        "id": "eda_quality",
        "title": "EDA 与数据质量",
        "description": "检查数据规模、缺失值、目标分布、重复值和训练测试差异。",
        "evidence": ["data_quality.json"],
    },
    {
        "id": "feature_engineering",
        "title": "特征工程",
        "description": "使用可解释、可回滚的表格特征处理和编码策略。",
        "evidence": ["task_scaffold.json", "model_results.json"],
    },
    {
        "id": "model_validation",
        "title": "建模与验证",
        "description": "模型指标与比赛指标一致，并通过本地阈值 gate。",
        "evidence": ["model_results.json", "validation_gate.json"],
    },
    {
        "id": "submission_check",
        "title": "Submission 检查",
        "description": "submission 行数、列名、缺失预测和预测合法性通过检查。",
        "evidence": ["submission.csv", "validation_gate.json"],
    },
    {
        "id": "logging_report",
        "title": "实验记录与报告",
        "description": "保存实验日志、报告、证据文件和可复盘结论。",
        "evidence": ["experiment_log.json", "local_report.md / titanic_local_report.md"],
    },
    {
        "id": "scaffold_post",
        "title": "Scaffold 与 Post-Scaffold",
        "description": "先生成解题脚手架，再记录下一轮改进建议。",
        "evidence": ["task_scaffold.json", "post_scaffold_improvement.json"],
    },
    {
        "id": "transferability",
        "title": "多任务迁移与业务验证",
        "description": "同一套工作流跑通 Titanic 分类、House Prices 回归和 Telco Churn 业务分类。",
        "evidence": ["experiments/titanic", "experiments/house_prices", "experiments/telco_churn"],
    },
]


PLAN_REVISION = {
    "source_doc": "docs/科研Agent工作站任务规划书.docx",
    "teacher_focus": [
        "表格 Kaggle 任务只能作为第一阶段入口，不能把项目长期限定在表格建模。",
        "工作站要体现资料/文献调研、方案设计、代码生成、训练执行、结果检查、报告复盘。",
        "关键结论、数据删除、长时间训练、官方提交和最终结论需要人工确认。",
        "当前成果要说成轻量可运行底座，图像、文本、图结构、多模态和论文生成属于下一阶段扩展。",
    ],
    "positioning": "轻量但真实可运行的 AI Data Scientist / AutoResearch 工作站",
    "current_stage": "第一阶段：用 Kaggle 或老师指定数据任务跑通可检查、可复盘、可复现的数据科学闭环。",
    "acceptance_focus": [
        "能跑通：任务理解、数据分析、baseline、验证、submission 检查、报告输出齐全。",
        "能检查：所有指标、文件格式、阶段 gate 和研究完整性 gate 有证据。",
        "能复现：保留配置、日志、运行命令、数据来源和实验目录。",
        "能汇报：页面和文档能自然说明已完成、受控未完成和下一步计划。",
    ],
}


RESEARCH_WORKFLOW = [
    {
        "id": "understand",
        "title": "任务理解",
        "role": "Orchestrator",
        "status": "implemented",
        "description": "读取任务说明、字段、评价指标、输出格式，先标记约束和风险。",
    },
    {
        "id": "survey",
        "title": "资料/文献调研",
        "role": "Evidence",
        "status": "planned_lightweight",
        "description": "围绕任务背景和已有方法整理依据，第一阶段保持轻量，不伪造引用。",
    },
    {
        "id": "design",
        "title": "方案设计",
        "role": "Planner",
        "status": "implemented",
        "description": "生成 scaffold：数据处理、验证方式、baseline、改进方向和输出文件。",
    },
    {
        "id": "quality",
        "title": "数据分析与处理",
        "role": "Analyst",
        "status": "implemented",
        "description": "检查规模、类型、缺失、异常、训练测试差异和目标变量分布。",
    },
    {
        "id": "coding",
        "title": "代码生成与运行",
        "role": "Developer",
        "status": "implemented",
        "description": "生成并运行本地 pipeline；失败时保留错误原因和修复记录。",
    },
    {
        "id": "training",
        "title": "模型训练与验证",
        "role": "Developer/Gate",
        "status": "implemented",
        "description": "先跑 baseline，再记录模型、参数、指标、耗时和输出文件。",
    },
    {
        "id": "check",
        "title": "结果检查",
        "role": "Reviewer",
        "status": "implemented",
        "description": "检查 submission 行数、列名、缺失、预测范围、指标阈值和格式。",
    },
    {
        "id": "report",
        "title": "报告与复盘",
        "role": "Summarizer",
        "status": "implemented",
        "description": "生成 Markdown/Word 报告，结论必须对应真实日志、代码或指标。",
    },
    {
        "id": "human",
        "title": "人工审核边界",
        "role": "Human Reviewer",
        "status": "controlled",
        "description": "Kaggle token、GPU、服务器、长训练、官方提交和最终结论保持人工确认。",
    },
]


WORKSTATION_BLUEPRINT = {
    "source_reference": "D:/下载/AI_Workstation_Agent_Interview_A4_Compact.html",
    "product_modules": [
        {
            "name": "Agent 工作台",
            "status": "mapped",
            "description": "把科研任务拆给 Orchestrator、Analyst、Developer、Evidence、Reviewer 等角色。",
        },
        {
            "name": "任务中心",
            "status": "mapped",
            "description": "以 Titanic 和 House Prices 作为可追踪任务，展示阶段、负责人、证据和 gate。",
        },
        {
            "name": "工作流",
            "status": "mapped",
            "description": "把一次性实验沉淀为任务理解、方案、执行、验证、报告、复盘的可复用流程。",
        },
        {
            "name": "知识库/RAG",
            "status": "lightweight",
            "description": "当前以研究来源、计划书、报告和审计文件作为证据库；向量检索为后续扩展。",
        },
        {
            "name": "人工确认与安全门",
            "status": "mapped",
            "description": "官方 Kaggle 提交、GPU、服务器私有 Agent、最终结论都不自动执行。",
        },
        {
            "name": "上线验收",
            "status": "mapped",
            "description": "提供 dashboard summary、validation gate、research integrity gate 和浏览器验收。",
        },
    ],
    "open_source_ui_refs": [
        {
            "name": "Dify",
            "url": "https://github.com/langgenius/dify",
            "lesson": "应用/Agent/Workflow/RAG 统一入口，适合作为工作台信息架构参考。",
        },
        {
            "name": "Flowise",
            "url": "https://github.com/FlowiseAI/Flowise",
            "lesson": "低代码节点式编排，适合借鉴流程轨道与工具节点呈现。",
        },
        {
            "name": "AutoGen Studio",
            "url": "https://github.com/microsoft/autogen",
            "lesson": "多 Agent 原型和运行记录应可视化，适合借鉴团队/任务/日志分区。",
        },
        {
            "name": "NanoResearch",
            "url": "https://github.com/OpenRaiser/NanoResearch",
            "lesson": "科研自动化需要 manifest、断点、实验产物和论文/报告链路。",
        },
    ],
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, JSONDecodeError):
        return default


def read_text(path: Path, max_chars: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")[:max_chars]
    except FileNotFoundError:
        return ""


def read_yaml(path: Path, default: Any = None) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def configured_experiment_root() -> Path:
    evidence_root = os.environ.get("RESEARCH_EVIDENCE_ROOT")
    default = Path(evidence_root) / "experiments" if evidence_root else ROOT / "experiments"
    configured = Path(os.environ.get("RESEARCH_EXPERIMENT_ROOT", default))
    return (configured if configured.is_absolute() else ROOT / configured).resolve()


def latest_experiment(task_name: str) -> Path | None:
    base = configured_experiment_root() / task_name
    if not base.exists():
        return None
    candidates = [item for item in base.iterdir() if item.is_dir()]
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: item.name, reverse=True)
    for item in ordered:
        gate = read_json(item / "validation_gate.json", {})
        if gate.get("status") == "passed":
            return item
    for item in ordered:
        gate_path = item / "validation_gate.json"
        if gate_path.exists() and gate_path.stat().st_size > 0:
            return item
    for item in ordered:
        log_path = item / "experiment_log.json"
        if log_path.exists() and log_path.stat().st_size > 0:
            return item
    return ordered[0]


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    experiment_root = configured_experiment_root()
    if resolved.is_relative_to(experiment_root):
        return f"runtime/experiments/{resolved.relative_to(experiment_root).as_posix()}"
    return "runtime/external-artifact"


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
    }


def summarize_task(task_name: str, report_name: str) -> dict[str, Any]:
    exp_dir = latest_experiment(task_name)
    if exp_dir is None:
        return {
            "name": task_name,
            "status": "missing",
            "experiment_dir": None,
            "gate": {},
            "stages": [],
            "artifacts": [],
        }

    gate = read_json(exp_dir / "validation_gate.json", {})
    log = read_json(exp_dir / "experiment_log.json", {})
    audit = read_json(exp_dir / "workflow_stage_audit.json", {})
    scaffold = read_json(exp_dir / "task_scaffold.json", {})
    post_scaffold = read_json(exp_dir / "post_scaffold_improvement.json", {})
    quality = read_json(exp_dir / "data_quality.json", {})
    results = read_json(exp_dir / "model_results.json", {})

    artifact_names = [
        "validation_gate.json",
        "task_scaffold.json",
        "task_scaffold.md",
        "workflow_stage_audit.json",
        "workflow_stage_audit.md",
        "post_scaffold_improvement.json",
        "post_scaffold_improvement.md",
        "experiment_log.json",
        "data_quality.json",
        "model_results.json",
        "submission.csv",
        report_name,
        report_name.replace(".md", ".docx"),
    ]
    artifacts = [file_info(exp_dir / name) for name in artifact_names if (exp_dir / name).exists()]

    best_model = gate.get("best_model") or log.get("evaluation", {}).get("best_model")
    metric_summary = {
        key: value
        for key, value in gate.items()
        if key
        in {
            "cv_accuracy_mean",
            "holdout_accuracy",
            "cv_rmsle_mean",
            "holdout_rmsle",
            "submission_rows",
            "submission_columns",
            "workflow_version",
        }
    }

    return {
        "name": task_name,
        "status": gate.get("status", "unknown"),
        "experiment_dir": rel(exp_dir),
        "best_model": best_model,
        "metrics": metric_summary,
        "gate": gate,
        "stages": audit.get("stages", []),
        "all_stages_passed": audit.get("all_stages_passed", False),
        "scaffold": {
            "time_budget_minutes": scaffold.get("time_budget_minutes"),
            "candidate_models": scaffold.get("candidate_models", []),
            "risk_points": scaffold.get("risk_points", []),
            "feature_plan": scaffold.get("feature_plan", []),
            "agent_template_mapping": scaffold.get("agent_template_mapping", {}),
        },
        "post_scaffold": post_scaffold,
        "quality": quality,
        "model_results": results.get("model_results", {}),
        "artifacts": artifacts,
        "report_excerpt": read_text(exp_dir / report_name, 3000),
    }


def summarize_plan_completion(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_map = {task["name"]: task for task in tasks}
    titanic_passed = task_map.get("titanic", {}).get("status") == "passed"
    house_passed = task_map.get("house_prices", {}).get("status") == "passed"
    telco_passed = task_map.get("telco_churn", {}).get("status") == "passed"
    all_required_passed = titanic_passed and house_passed and telco_passed

    completion: list[dict[str, Any]] = []
    for item in PLAN_REQUIREMENTS:
        if item["id"] == "scaffold_post":
            passed = all(task.get("scaffold", {}).get("candidate_models") for task in tasks) and bool(
                task_map.get("house_prices", {}).get("post_scaffold")
            )
        elif item["id"] == "transferability":
            passed = all_required_passed
        else:
            passed = all_required_passed and all(task.get("all_stages_passed") for task in tasks)
        completion.append({**item, "status": "passed" if passed else "needs_review"})
    return completion


def summarize_environment() -> dict[str, Any]:
    kaggle_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / "AppData" / "Roaming" / "kaggle" / "kaggle.json",
        ROOT / "kaggle.json",
    ]
    return {
        "workspace": str(ROOT),
        "local_only": True,
        "docker_ready": True,
        "kaggle_token_paths": [{"path": str(path), "exists": path.exists()} for path in kaggle_paths],
        "gpu_server_connected": False,
        "server_private_agents_modified": False,
        "server_template_policy": "read_only_public_templates",
    }


def summarize_workstation_metrics(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    stages = [stage for task in tasks for stage in task.get("stages", [])]
    artifacts = [artifact for task in tasks for artifact in task.get("artifacts", [])]
    return {
        "tasks_passed": sum(1 for task in tasks if task.get("status") == "passed"),
        "tasks_total": len(tasks),
        "stages_passed": sum(1 for stage in stages if stage.get("status") == "passed"),
        "stages_total": len(stages),
        "artifact_count": len(artifacts),
        "implemented_workflow_steps": sum(1 for item in RESEARCH_WORKFLOW if item["status"] == "implemented"),
        "workflow_steps_total": len(RESEARCH_WORKFLOW),
    }


def summarize_research_sources() -> dict[str, Any]:
    config = read_yaml(ROOT / "configs" / "research_sources.yaml", {"sources": []})
    return {
        "last_verified": config.get("last_verified"),
        "verification_policy": config.get("verification_policy"),
        "sources": config.get("sources", []),
    }


def summarize_integrity_gate() -> dict[str, Any]:
    configured_path = Path(
        os.environ.get("RESEARCH_INTEGRITY_GATE_PATH", ROOT / "docs" / "research_integrity_gate.json")
    )
    gate_path = (configured_path if configured_path.is_absolute() else ROOT / configured_path).resolve()
    root = ROOT.resolve()
    gate = read_json(gate_path, {})
    return {
        "path": rel(gate_path) if gate_path.is_relative_to(root) else "runtime/research_integrity_gate.json",
        "status": gate.get("status", "not_run"),
        "dimensions": gate.get("dimensions", []),
        "tasks": gate.get("tasks", []),
        "roadmap_items": gate.get("roadmap_items", []),
    }


def summarize_long_term_roadmap() -> dict[str, Any]:
    roadmap = read_yaml(ROOT / "configs" / "long_term_roadmap.yaml", {"items": []})
    return {
        "version": roadmap.get("version"),
        "status_policy": roadmap.get("status_policy", {}),
        "items": roadmap.get("items", []),
    }


def build_summary() -> dict[str, Any]:
    tasks = [
        summarize_task("titanic", "titanic_local_report.md"),
        summarize_task("house_prices", "local_report.md"),
        summarize_task("telco_churn", "local_report.md"),
    ]
    house_templates = tasks[1].get("scaffold", {}).get("agent_template_mapping", {})
    return {
        "title": "科研 Agent 工作站",
        "subtitle": "Academic Research Agent Workstation",
        "generated_from": str(ROOT),
        "tasks": tasks,
        "workstation_metrics": summarize_workstation_metrics(tasks),
        "plan_revision": PLAN_REVISION,
        "research_workflow": RESEARCH_WORKFLOW,
        "workstation_blueprint": WORKSTATION_BLUEPRINT,
        "plan_completion": summarize_plan_completion(tasks),
        "agent_templates": house_templates,
        "agent_template_config_present": (ROOT / "configs" / "agent_templates.yaml").exists(),
        "research_sources": summarize_research_sources(),
        "research_integrity": summarize_integrity_gate(),
        "long_term_roadmap": summarize_long_term_roadmap(),
        "environment": summarize_environment(),
        "final_audit": {
            "path": rel(ROOT / "docs" / "科研Agent工作站最终完成审计.md"),
            "excerpt": read_text(ROOT / "docs" / "科研Agent工作站最终完成审计.md", 5000),
        },
        "plan_doc": {
            "path": rel(ROOT / "docs" / "科研Agent工作站详细计划_老师审查版.md"),
            "excerpt": read_text(ROOT / "docs" / "科研Agent工作站详细计划_老师审查版.md", 4000),
        },
        "notes": [
            "This dashboard is read-only and summarizes local experiment evidence.",
            "Kaggle official submission is intentionally disabled until credentials are configured.",
            "GPU/server work is intentionally disconnected to protect the existing AI workstation.",
        ],
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, content: str, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
        data = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/summary":
            self.send_json(build_summary())
            return
        if route == "/health":
            summary = build_summary()
            tasks_ok = all(task.get("status") == "passed" for task in summary["tasks"])
            self.send_json({"status": "ok" if tasks_ok else "degraded", "tasks_ok": tasks_ok})
            return
        if route == "/api/file":
            query = parse_qs(parsed.query)
            requested = unquote(query.get("path", [""])[0])
            target = (ROOT / requested).resolve()
            if not str(target).startswith(str(ROOT.resolve())):
                self.send_json({"error": "path outside workspace"}, status=403)
                return
            allowed_suffixes = {".md", ".json", ".txt", ".csv", ".docx", ".png"}
            if not target.exists() or target.suffix.lower() not in allowed_suffixes:
                self.send_json({"error": "file not readable through dashboard"}, status=404)
                return
            if target.suffix.lower() in {".docx", ".png"}:
                data = target.read_bytes()
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_text(target.read_text(encoding="utf-8-sig", errors="ignore")[:20000])
            return

        static_path = STATIC_DIR / ("index.html" if route == "/" else route.lstrip("/"))
        static_path = static_path.resolve()
        if not str(static_path).startswith(str(STATIC_DIR.resolve())) or not static_path.exists():
            self.send_text("Not found", status=404)
            return
        content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        data = static_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local academic research agent dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
