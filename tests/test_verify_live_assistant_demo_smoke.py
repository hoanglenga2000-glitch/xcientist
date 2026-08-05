from __future__ import annotations

from scripts import verify_live_assistant_demo_smoke as smoke


def _assistant_result(answer: str, *, tool_names: list[str] | None = None) -> dict:
    return {
        "http_status": 200,
        "completed": True,
        "errors": [],
        "answer_characters": len(answer),
        "provider": smoke.REQUIRED_PROVIDER,
        "model": smoke.REQUIRED_MODEL,
        "tool_names": tool_names or ["hpc_connection_status"],
        "tool_calls_total": 1,
        "_answer_private": answer,
    }


def test_connection_turn_passes_for_job90673_ready_boundary() -> None:
    answer = (
        "简单说：job90673 连接已就绪。"
        "证据：作业号 90673，profile_state: active，readiness_status: ready，"
        "gateway_banner_ok=True，SOCKS 和 SSH 网关均通过；"
        "job_container_verified=True，5 次只读容器身份采样全部通过。"
        "边界：training_started=False，kaggle_submissions=0，grader_calls=0，"
        "没有返回密钥。结论是连接层可作为后续工作的基础，但训练、远程命令、提交和 grader "
        "仍要等任务级 Gate 明确放行。下一步只做受控执行前检查。"
    )
    result = _assistant_result(answer)

    passed, failures = smoke.connection_turn_passed(result)

    assert passed is True
    assert failures == []
    assert result["connection_checks"]["connection_hpc_tool"] is True
    assert result["connection_checks"]["connection_no_private_grader"] is True


def test_connection_turn_accepts_concise_equivalent_no_submit_wording() -> None:
    answer = (
        "简单说：现在可以连接服务器，作业号 90673 的连接已就绪。"
        "Profile 是 job90673，状态 active；SOCKS 桥正在监听，SSH 网关检查通过，"
        "readiness_status=ready，job_container_verified=True，5 次只读检查通过，失败检查为空。"
        "本轮仅做只读检查，未启动训练、未执行 Kaggle 提交、未调用 private grader，"
        "也没有运行远程命令或返回凭证。下一步仍需先通过任务级 Gate 再执行。"
    )
    result = _assistant_result(answer)

    passed, failures = smoke.connection_turn_passed(result)

    assert len(answer) >= 180
    assert passed is True
    assert failures == []


def test_connection_turn_requires_hpc_connection_tool() -> None:
    answer = (
        "简单说：job90673 ready。证据：profile active，gateway ready。"
        "training_started=False，kaggle_submissions=0，grader_calls=0。下一步继续只读检查。"
    )
    result = _assistant_result(answer, tool_names=["verified_context"])

    passed, failures = smoke.connection_turn_passed(result)

    assert passed is False
    assert "connection_hpc_tool" in failures


def test_connection_turn_rejects_profile_and_gateway_only_false_positive() -> None:
    answer = (
        "简单说：job90673 连接已就绪。证据：profile active，readiness_status=ready，"
        "SOCKS 正在监听，gateway_banner_ok=True。未启动训练、未提交 Kaggle、"
        "未调用 private grader。下一步做执行前检查。"
    )
    result = _assistant_result(answer)

    passed, failures = smoke.connection_turn_passed(result)

    assert passed is False
    assert "connection_truthful_live_status" in failures


def test_connection_turn_accepts_truthful_container_blocked_result() -> None:
    answer = (
        "简单说：job90673 当前没有通过目标容器实时连接验证。"
        "本地 profile_state=active，readiness_status=ready，SOCKS 桥和 SSH 网关 banner 可达；"
        "但 live_status=job_container_channel_closed，job_container_verified=False，"
        "说明网关可达不等于容器可用。下一步确认 allocation 仍在运行；本轮未启动训练、"
        "未提交 Kaggle、未调用 private grader，signals_sent=0。"
    )
    result = _assistant_result(answer)

    passed, failures = smoke.connection_turn_passed(result)

    assert passed is True
    assert failures == []


def test_connection_turn_rejects_missing_side_effect_boundaries() -> None:
    answer = "简单说：job90673 ready。证据：profile active，gateway ready。下一步继续检查。"
    result = _assistant_result(answer)

    passed, failures = smoke.connection_turn_passed(result)

    assert passed is False
    assert "connection_no_training" in failures
    assert "connection_no_kaggle_submit" in failures
    assert "connection_no_private_grader" in failures


def test_ux_budget_includes_connection_turn() -> None:
    first = {"provider": smoke.REQUIRED_PROVIDER, "model": smoke.REQUIRED_MODEL, "seconds": 40, "answer_characters": 2000, "input_tokens": 5000, "output_tokens": 1000}
    followup = {"provider": smoke.REQUIRED_PROVIDER, "model": smoke.REQUIRED_MODEL, "seconds": 30, "answer_characters": 700, "input_tokens": 9000, "output_tokens": 500}
    literature = {"provider": smoke.REQUIRED_PROVIDER, "model": smoke.REQUIRED_MODEL, "seconds": 35, "answer_characters": 900, "input_tokens": 10000, "output_tokens": 600}
    connection = {"provider": smoke.REQUIRED_PROVIDER, "model": smoke.REQUIRED_MODEL, "seconds": 30, "answer_characters": 600, "input_tokens": 9000, "output_tokens": 400}

    result = smoke.ux_budget_result(first, followup, literature, connection)

    assert result["passed"] is True
    assert "connection_seconds_le_90" in result["checks"]
    assert "connection_chars_180_1400" in result["checks"]
    assert result["turns"]["connection"]["characters"] == 600
