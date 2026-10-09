import importlib.util,shutil
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('siim_aide_adapter',ROOT/'scripts/siim_aide_controller.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)

@pytest.fixture
def core(tmp_path):
    root=ROOT/'artifacts/siim-mlebench-calibration-20260908/aide-source/aideml-d4a77cf3ca11e0f70749052b2701003e32bc245a/aide'
    sources={'agent.py':root/'agent.py','journal.py':root/'journal.py','metric.py':root/'utils/metric.py','response.py':root/'utils/response.py','backend-utils.py':root/'backend/utils.py'}
    for name,path in sources.items():shutil.copyfile(path,tmp_path/name)
    return adapter.load_core(tmp_path,lambda **kwargs:'Plan.\n```python\ndef fit_model(a,b,c,d,e):\n    return None\n```')

def test_native_aide_search_and_generation(core):
    cfg=SimpleNamespace(agent=SimpleNamespace(search=SimpleNamespace(num_drafts=1,debug_prob=1,max_debug_depth=20),
        time_limit=7200,steps=24,code=SimpleNamespace(model='deepseek-v4-pro',temp=.3),convert_system_to_user=False,
        obfuscate=False,data_preview=False,expose_prediction=False,k_fold_validation=5),exec=SimpleNamespace(timeout=7200))
    agent=core.Agent('test',cfg,core.Journal())
    assert agent.search_policy() is None
    node=core.Node(code='x',plan='plan');node.is_buggy=False;node.metric=core.MetricValue(.7,maximize=True);node.analysis='ok'
    agent.journal.append(node)
    assert agent.search_policy() is node
    node.is_buggy=True;node.metric=core.WorstMetricValue()
    assert agent.search_policy() is node
    plan,code=agent.plan_and_code_query({})
    assert 'fit_model' in code and plan=='Plan.'

def test_native_aide_metric_and_tree(core):
    parent=core.Node(code='x');parent.is_buggy=True
    child=core.Node(code='y',parent=parent)
    assert child.stage_name=='debug' and child.debug_depth==1
    assert core.MetricValue(.8,maximize=True)>core.MetricValue(.7,maximize=True)
