"""Short Chinese labels for the live process timeline (t237 C6a).

Mirror of ``activityLabel`` in web/.../task-presentation.ts. A tool id, path, argument
or provider error string is never shown; an unknown tool is 「处理一步」.
"""
from __future__ import annotations

from datetime import datetime

_TOOL_LABELS: tuple[tuple[str, str], ...] = (
    ('web_search', '搜索资料'), ('search', '搜索资料'), ('fetch', '读取网页'), ('browse', '读取网页'),
    ('file_search', '查找文件'), ('file_list', '查找文件'), ('directory_list', '查找文件'), ('directory_stat', '查找文件'),
    ('file_read', '读取文件'), ('directory_read', '读取文件'), ('attachment_read', '读取文件'), ('attachment_list', '查看附件'),
    ('file_write', '写入文件'), ('file_patch', '修改文件'), ('directory_write', '写入文件'), ('directory_mkdir', '整理目录'),
    ('directory_copy', '复制文件'), ('directory_sync', '同步文件'), ('directory_delete', '删除文件'),
    ('shell_exec', '运行命令'), ('process_', '运行命令'), ('job_', '运行作业'),
    ('artifact_', '整理结果文件'), ('report_', '生成报告'), ('memory_', '记住要点'),
    ('verified_context', '核对上下文'), ('training_route', '规划训练'), ('managed_tensor_train', '训练模型'),
    ('managed_model_prepare', '准备模型'), ('hpc_', '检查算力'), ('kaggle_', '查看比赛'),
    ('cv_score', '评估结果'), ('experiment_', '整理实验'),
)
UNKNOWN_TOOL = '处理一步'

_ERROR_LABELS = {
    'permission_denied': '没有权限', 'personal_tool_boundary': '超出可用范围', 'cross_run_path': '超出本任务范围',
    'timeout': '等待超时', 'cancelled': '已取消',
}
FAILED_STEP = '这一步没有完成'

# Fixed run steps (assistant_runs plan order); index is 1-based.
STEPS = ('understand', 'execute', 'verify', 'deliver')


def tool_label(name: str) -> str:
    name = str(name or '')
    for prefix, label in _TOOL_LABELS:
        if name == prefix or (prefix.endswith('_') and name.startswith(prefix)) or name.startswith(prefix + '_'):
            return label
    return UNKNOWN_TOOL


def error_label(code: str) -> str:
    """One short Chinese reason (≤16 characters); never the raw provider or tool text."""
    return _ERROR_LABELS.get(str(code or ''), FAILED_STEP)


def step_position(step_id: str) -> dict[str, int]:
    return {'index': STEPS.index(step_id) + 1, 'total': len(STEPS)} if step_id in STEPS else {}


def duration_ms(started_at: str, completed_at: str) -> int | None:
    try:
        delta = datetime.fromisoformat(str(completed_at)) - datetime.fromisoformat(str(started_at))
    except (TypeError, ValueError):
        return None
    return max(0, round(delta.total_seconds() * 1000))
