# XCIENTIST 上线验证清单 — 给 Codex 的最终检查

> 复制下面整个 block 给 Codex。Codex 只需要按顺序执行每条命令，报告 PASS/FAIL。
> 这是只读验证——不要启动训练，不要提交 Kaggle，不要修改代码。

---

```markdown
你现在是本项目的上线审计工程师。只读验证，不训练，不提交，不修改代码。

项目根目录：D:\桌面\codex\科研港科技

## 1. 编译检查（3 项）

```powershell
cd D:\桌面\codex\科研港科技

# 1a: 核心 XSCI 模块
python -m py_compile src/xsci/kaggle.py src/xsci/kaggle_conversation.py src/xsci/kaggle_actions.py src/xsci/kaggle_competitions.py src/xsci/config.py src/xsci/kaggle_session.py src/xsci/kaggle_intent.py src/xsci/memory.py src/xsci/agent.py
echo "1a: %errorlevel% (0=pass)"

# 1b: 向量索引模块
python -m py_compile src/research_os/memory_vector_index.py src/research_os/agent/memory_library.py src/research_os/agent/tools.py
echo "1b: %errorlevel% (0=pass)"

# 1c: TypeScript
cd web/research-agent-workstation && npm run typecheck
echo "1c: %errorlevel% (0=pass)"
cd D:\桌面\codex\科研港科技
```

## 2. 测试（1 项）

```powershell
cd D:\桌面\codex\科研港科技
python -m pytest tests/test_kaggle_menu.py tests/test_xsci_cli.py tests/test_xsci_phase2.py tests/test_xsci_phase3.py tests/test_kaggle_stream.py -q
# 预期: 全部 PASS 或 SKIP（1 个 SKIP 是 Windows POSIX 权限，正常）
```

## 3. Secret 扫描（1 项）

```powershell
python scripts/verify_no_plaintext_secrets.py
# 预期: {"status": "passed", ...}
```

## 4. CLI 命令（4 项）

```bash
# Git Bash 中执行:
cd "D:/桌面/codex/科研港科技"

echo "=== 4a: kaggle --help ==="
kaggle --help | head -5
# 预期: 看到 "Kaggle Research Agent  XSCI self-evolving MLE workstation"

echo "=== 4b: kaggle ready ==="
kaggle ready | head -10
# 预期: 看到 Readiness 表格，GPU 显示 blocked

echo "=== 4c: kaggle official --help ==="
kaggle official --help | head -5
# 预期: 看到官方 Kaggle CLI help

echo "=== 4d: autokaggle --help ==="
autokaggle --help | head -5
# 预期: 看到 "Kaggle Research Agent" 和 "Usage:"
```

## 5. 前端与 API（5 项）

```powershell
# 先确保前端在运行
curl -s -o nul -w "%{http_code}" http://127.0.0.1:8088
# 预期: 200

# 5a
curl -s http://127.0.0.1:8088/api/workstation-summary | python -c "import json,sys; d=json.load(sys.stdin); print('PASS' if d.get('tasks') else 'FAIL')"

# 5b
curl -s http://127.0.0.1:8088/api/tasks | python -c "import json,sys; d=json.load(sys.stdin); print('PASS' if d.get('ok') else 'FAIL')"

# 5c
curl -s "http://127.0.0.1:8088/api/evolution/state?task_id=house_prices" | python -c "import json,sys; d=json.load(sys.stdin); print('PASS' if d.get('ok') else 'FAIL')"

# 5d: 文献检索（真实 API）
curl -s -X POST http://127.0.0.1:8088/api/literature/search -H "Content-Type: application/json" -d '{"query":"gradient boosting"}' | python -c "import json,sys; d=json.load(sys.stdin); print('PASS' if d.get('papers') else 'FAIL')"

# 5e: 设置 API
curl -s http://127.0.0.1:8088/api/settings | python -c "import json,sys; d=json.load(sys.stdin); print('PASS' if d.get('ok') else 'FAIL')"
```

## 6. 向量索引验证（1 项）

```python
# python -c 一行:
import sys; sys.path.insert(0, 'src')
from research_os.memory_vector_index import MemoryVectorIndex
from research_os.retrospective_memory import RetrospectiveMemoryStore
store = RetrospectiveMemoryStore('experiments/evolution/retrospective_memory.json')
idx = MemoryVectorIndex(store); idx.build()
r = idx.query('timeout on large text data', k=3)
print(f'PASS: {len(r)} results, top={r[0]["_score"]:.3f}, id={r[0]["memory_id"]}') if r else print('FAIL: no results')
```

## 7. GPU 状态检查（1 项）

```powershell
curl -s -X POST http://127.0.0.1:8088/api/gpu/connections/test
# 预期: status=blocked_resource_gateway（这是正确的，GPU 还没开放）
```

## 8. GitHub 仓库检查（2 项）

```powershell
git remote -v
# 预期: origin https://github.com/hoanglenga2000-glitch/xcientist.git

git log --oneline -1
# 预期: XCIENTIST: AI research workstation for Kaggle/MLE-Bench
```

## 9. Claim 边界检查（3 项）

```powershell
# 9a: 确认没有虚假奖牌声明
python -c "import json; d=json.load(open('workspace/benchmark_multi_task_summary_20260623.json','r',encoding='utf-8')); assert d['medal_rate']==0.0; print('PASS: medal_rate is 0.0')"

# 9b: 确认 evolution state 不允许官方提交
curl -s "http://127.0.0.1:8088/api/evolution/state?task_id=house_prices" | python -c "import json,sys; d=json.load(sys.stdin); assert d['official_submit_allowed']==False; print('PASS: official_submit_allowed=False')"

# 9c: 确认 report 不含虚假排名
python -c "import json; r=json.loads(open('src/xsci/kaggle.py','r',encoding='utf-8').read()); assert 'medal' not in r.lower() or 'medal_rate' not in r; print('PASS: no medal claims in kaggle.py')"
```

## 判定标准

| 通过 | 条件 |
|------|------|
| **Go** | 全部 PASS 或预期的 blocked 状态 |
| **No-Go** | 任何 unexpected FAIL |

预期会失败（正确的阻塞状态）：
- GPU connections test → `blocked_resource_gateway`
- 1 个 POSIX 权限 test → SKIP

如果全部 PASS（含预期 blocked），输出：`XCIENTIST GO`
```

---

## Codex 执行说明

1. 复制上面的 ` ```markdown ... ``` ` 整个 block 给 Codex
2. Codex 必须按顺序执行每项命令
3. 每项命令后 Codex 应报告 PASS 或 FAIL
4. 最后汇总：通过 X/13，Go 或 No-Go
5. 如果前端没启动（8088 返回 Connection refused），先运行：
   ```powershell
   powershell -File scripts/start_verified_workstation.ps1 restart
   ```
