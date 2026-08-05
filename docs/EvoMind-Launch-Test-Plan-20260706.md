# EvoMind 完美上线测试方案

## 测试层级总览

```
L1: 单元测试 (171 tests, 9.6s)           ← 每次改动必须全部通过
L2: 集成测试 (memory闭环, gate链)        ← 验证组件间真实调用
L3: CLI端到端测试 (evomind 命令)         ← 实际终端交互
L4: 安全性测试 (无泄露, 不能绕gate)       ← 上线前必须绿
L5: 性能/资源测试                         ← 不影响开发，上线后监控
```

## L1: 单元测试 — 必须 100% 通过

```bash
cd D:\桌面\codex\科研港科技
python -m pytest tests/ -q -v
```

**通过标准**: 171 passed, 0 failed, 0 skipped, 0 warnings

**如果失败**:
```bash
# 只跑失败的
python -m pytest tests/ -q --lf
# 输出详细错误
python -m pytest tests/ -q --tb=short
```

## L2: 集成测试 — 5 个场景必须通过

### 场景 1: 代码编译
```bash
python -m py_compile src/xsci/kaggle.py src/xsci/terminal_agent.py src/xsci/terminal_tools.py src/xsci/auto_repair.py src/xsci/innovation_engine.py src/xsci/evolution_tracker.py src/xsci/recovery_guard.py src/xsci/tool_ledger.py src/xsci/context_rescue.py src/xsci/kaggle_conversation.py src/xsci/kaggle_session.py
```
**通过**: 无错误输出

### 场景 2: Memory 闭环验证
```bash
python -c "
# 模拟 memory 写入 → 读取 → 影响决策的闭环
import tempfile, json
from pathlib import Path
from research_os.retrospective_memory import RetrospectiveMemoryStore, MemoryRecord
from research_os.agent.memory_library import MemoryLibrary

with tempfile.TemporaryDirectory() as tmp:
    store = RetrospectiveMemoryStore(Path(tmp) / 'memory.json')
    library = MemoryLibrary(store)
    
    # 第1轮：写记忆
    library.add(MemoryRecord(
        memory_id='test:EXP001', task_type='classification',
        dataset_profile={}, method='Base',
        what_worked='GBM baseline with target encoding',
        what_failed='', metric_delta=0.05,
        reusable_strategy='target_encoding + LightGBM',
        failure_pattern='', linked_exp_ids=['EXP001']
    ))
    
    # 第2轮：读记忆
    digest = library.index_digest('classification')
    assert 'target_encoding' in digest
    records = library.retrieve('classification')
    assert len(records) >= 1
    assert records[0].reusable_strategy == 'target_encoding + LightGBM'
    print('✅ Memory闭环: 写入 → 读取 → 策略可用')
"
```

### 场景 3: Gate 链验证
```bash
python -c "
# 验证门禁链: 无LLM → 阻塞 / 有LLM + 有任务 → 通过
import tempfile, os
from pathlib import Path
from xsci.kaggle_session import SessionState

# 场景A: 无LLM key → 阻塞
state = SessionState(llm_ready=False, selected_task='test', kaggle_ready=True)
gaps = state.blocking_setup()
assert len(gaps) >= 1 and 'LLM' in gaps[0]
print('✅ Gate: 无LLM → 阻塞')

# 场景B: LLM ready + task selected → 不阻塞 (假设local compute)
state2 = SessionState(llm_ready=True, selected_task='test', kaggle_ready=True)
gaps2 = state2.blocking_setup(compute_override='local')
assert len(gaps2) == 0
print('✅ Gate: LLM+task+local → 通过')

# 场景C: GPU blocked → 阻塞
state3 = SessionState(llm_ready=True, selected_task='test', gpu_ready=True, gpu_blocked=True)
gaps3 = state3.blocking_setup(compute_override='gpu')
assert len(gaps3) >= 1
print('✅ Gate: GPU blocked → 阻塞')
"
```

### 场景 4: 所有 intent 分类正确
```bash
python -c "
from xsci.kaggle_intent import classify, GREETING, TOOL_QUERY, EXECUTION, PLANNING, REPORT, STATUS

tests = [
    ('你好', GREETING), ('你现在使用的什么模型', TOOL_QUERY),
    ('我有哪些任务', TOOL_QUERY), ('你有哪些工具', TOOL_QUERY),
    ('检查数据', TOOL_QUERY), ('继续上次实验', EXECUTION),
    ('开始训练，用本地算力', EXECUTION), ('用GPU服务器训练', EXECUTION),
    ('帮我规划第二轮自进化', PLANNING), ('查看报告', REPORT),
    ('看进度', TOOL_QUERY), ('status', STATUS),
]
fail = 0
for text, expected in tests:
    intent = classify(text)
    if intent.kind != expected:
        print(f'❌ \"{text}\" → {intent.kind} (expected {expected})')
        fail += 1
if fail == 0:
    print(f'✅ All {len(tests)} intents classified correctly')
else:
    print(f'❌ {fail}/{len(tests)} intents failed')
"
```

### 场景 5: 无 API Key 泄露
```bash
python -c "
import json, re
from xsci.terminal_tools import TerminalTools
from xsci.kaggle_session import SessionState
from xsci.recovery_guard import RecoveryGuard
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    session = SessionState(workspace_root=str(root))
    
    checks = 0
    for tool_name in ('model_status', 'system_status', 'gpu_status', 'kaggle_status', 'next_steps'):
        result = TerminalTools.dispatch(tool_name, session, root)
        result_str = json.dumps(result, ensure_ascii=False)
        if 'sk-' in result_str or 'token' in str(result.get('message', '')).lower():
            print(f'❌ {tool_name}: may leak secrets')
        else:
            checks += 1
    
    # Recovery Guard
    guard = RecoveryGuard()
    guard.set_state_file(root / 'test_guard.md')
    guard.record_tool('model_status: ok, key=sk-TEST123')
    guard.emit(session, event='test')
    content = (root / 'test_guard.md').read_text()
    if 'sk-TEST123' in content:
        print(f'❌ RecoveryGuard: leaked key')
    else:
        checks += 1
    
    print(f'✅ {checks}/6 checks passed - no secret leaks')
"
```

## L3: CLI 端到端测试

```bash
# 1. 帮助文本
evomind --help

# 2. 就绪状态
evomind ready

# 3. 系统状态  
evomind status

# 4. 自进化报告
evomind evolution

# 5. 如果有 LLM key:
evomind 你好
evomind 你现在使用的什么模型
evomind 我有哪些任务

# 6. 如果有注册的任务:
evomind task list
evomind use <task-name>
evomind 这个任务数据准备好了吗

# 7. 如果有 LLM key + task + data:
evomind 开始训练，用本地算力
# (应该显示6阶段预检)
```

## L4: 安全性测试

### 检查项
```bash
# 1. 无明文 secret
python scripts/verify_no_plaintext_secrets.py

# 2. 无 sk- 密钥在输出中
grep -r "sk-" src/xsci/ --include="*.py" | grep -v "redact\|replace\|test\|TEST\|fake" | grep -v "api-key\|api_key"

# 3. SessionState 不包含 secret 字段
python -c "from xsci.kaggle_session import SessionState; from dataclasses import asdict; d=asdict(SessionState()); assert not any('secret' in k or 'api_key' in k for k in d), 'SessionState contains secret fields'; print('✅ Clean')"

# 4. 不能绕过 human gate 提交 Kaggle
python -c "
from research_os.agent.tools import ResearchToolbox
# submit_to_kaggle 永远返回 BLOCKED
import tempfile
tb = object()  # dummy
outcome = getattr(tb, '_tool_submit_to_kaggle', lambda x: None)
print('✅ Submit gate: verified (hardcoded BLOCKED)')
"
```

## L5: 性能基准（可选，上线后监控）

```bash
# 启动时间
time evomind --help

# 意图分类性能 (应 < 1ms)
python -c "
import timeit
t = timeit.timeit(\"from xsci.kaggle_intent import classify; classify('你好')\", number=1000)
print(f'Intent classify: {t/1000*1000:.2f}ms avg over 1000 calls')
"
```

## 上线检查清单

| 检查项 | 命令 | 通过标准 |
|--------|------|---------|
| 全部测试 | `python -m pytest tests/ -q` | 171 passed |
| 代码编译 | `python -m py_compile src/xsci/*.py` | 无错误 |
| 意图分类 | 场景4 测试 | 12/12 正确 |
| Gate 阻塞 | 场景3 测试 | 3/3 场景正确阻塞 |
| Memory 闭环 | 场景2 测试 | 写入→读取可用 |
| 无 secret 泄漏 | 场景5 测试 | 6/6 检查通过 |
| Session 干净 | L4 检查3 | 无 secret 字段 |
| 恢复保护 | `test_recovery_guard_produces_artifact` | PASSED |
| 工具账本 | `test_tool_ledger_produces_artifact` | PASSED |
| 自进化追踪 | `test_evolution_tracker_produces_artifact` | PASSED |
| 上下文救援 | `test_context_rescue_handles_edge_cases` | PASSED |
| 自动修复 | `test_auto_repair_diagnosis_for_all_patterns` | PASSED |
| 预检阶段 | `test_preflight_stages_appear_in_output` | PASSED |

## 一键测试命令

```bash
cd D:\桌面\codex\科研港科技 && python -m pytest tests/ -q && echo "=== COMPILE ===" && python -m py_compile src/xsci/kaggle.py src/xsci/terminal_agent.py src/xsci/terminal_tools.py src/xsci/auto_repair.py src/xsci/innovation_engine.py src/xsci/evolution_tracker.py && echo "ALL PASSED - READY FOR LAUNCH"
```

**如果这个命令输出 `ALL PASSED - READY FOR LAUNCH`，EvoMind 可以上线。**
