from __future__ import annotations

from research_os.agent.local_tabular_workflow import build_local_tabular_run
from xsci.user_request import parse_user_request

REQUEST = """请创建完整的信用卡交易欺诈检测研究任务，数据源使用 fraudTrain.csv 和 fraudTest.csv。
推理与科研编排使用 GPT-5.6，计算仅使用本机 RTX 4060 8GB，不使用远程集群。
执行数据审计、训练、独立复核、报告和可复现产物，不提交公开榜单。"""


def test_local_tabular_run_has_resource_review_and_submission_gates():
    request = parse_user_request(REQUEST)
    run = build_local_tabular_run(request, run_id="run-local4060-test")

    assert list(run.tasks) == [
        "setup",
        "research_context",
        "data_audit",
        "research_design",
        "local_gpu_evolution",
        "independent_review",
        "aggregate",
    ]
    assert run.tasks["local_gpu_evolution"].resource_type == "gpu"
    assert run.tasks["local_gpu_evolution"].dependencies == ("research_design",)
    assert run.gates["local_resource_gate"] == "required_before_training"
    assert run.gates["official_submission"] == "disabled"
    assert run.resource_limits["hpc_gpu"] == 0
