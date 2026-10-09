from pathlib import Path

import pytest

from evomind_runtime.assistant_runs import _artifact_requirements_from_prompt
from evomind_runtime.runtime import AgentRuntime


@pytest.mark.parametrize('marker', ['完整源码如下：', '完整源码：', '完整源代码如下：', 'Full source code:', 'Complete source code follows:'])
def test_source_appendix_does_not_create_final_deliverable_requirements(marker):
    prompt = (
        '读取 inputs/raw.csv；最终交付 result.json。' + marker + '\n'
        "raw = Path('canary-input.txt').read_text()\n"
        "# publish debug.log; 最终交付 trap.json\n"
    )
    assert _artifact_requirements_from_prompt(prompt) == {'names': ['result.json'], 'groups': []}


def test_source_only_filenames_after_publish_are_not_required():
    prompt = (
        '完成后只做本地回读/发布/预览这两个实际结果文件。完整源码如下：\n'
        "assert Path('canary-input.txt').read_text() == 'fixture'\n"
        "Path('gpu-canary-receipt.json').write_text('{}')\n"
    )
    assert _artifact_requirements_from_prompt(prompt) == {'names': [], 'groups': []}


def test_explicit_input_copy_delivery_is_still_required():
    assert _artifact_requirements_from_prompt(
        '最终交付原始输入副本 canary-input.txt 和结果 result.json。'
    )['names'] == ['canary-input.txt', 'result.json']


def test_runtime_uses_only_declared_output_before_source_appendix(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(
            prompt="最终交付 result.json。完整源码如下：\nsource = 'canary-input.txt'",
            conversation_id='source_appendix_fixture', start=False,
        )
        assert runtime.assistant._missing_expected_artifacts(run) == ['result.json']
        result = Path(run['task_root']) / 'outputs/result.json'
        result.write_text('{"status":"passed"}\n', encoding='utf-8')
        runtime.assistant.publish_path(run['id'], result, source_tool_call='fixture')
        assert runtime.assistant._missing_expected_artifacts(run) == []
    finally:
        runtime.close()
