from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))
from research_os.search_graph import ExperimentNode, SearchGraph

OUT_DIR=ROOT/'workspace/score_guard'
OUT_DIR.mkdir(parents=True,exist_ok=True)
metrics=json.loads((ROOT/'experiments/house_prices/wr_2026-06-23T23-17-07.433686_1209bfa4/metrics.json').read_text(encoding='utf-8'))
baseline_score=0.12899
candidate_score=float(metrics['ensemble']['best_validation_score'])
now=datetime.now().isoformat(timespec='seconds')
graph=SearchGraph(task_id='house_prices_advanced_regression_techniques',root_exp_id='EXP_BASELINE_HOUSE_20260623_225551',metric_name='rmsle',metric_direction='minimize',best_exp_id='EXP_BASELINE_HOUSE_20260623_225551')
graph.add_node(ExperimentNode(exp_id='EXP_BASELINE_HOUSE_20260623_225551',parent_id=None,branch_type='baseline',task_name='house_prices',hypothesis='Protected baseline is current best-so-far for RMSLE.',implementation_summary='Local template baseline on log1p target.',code_path='src/research_agent_workstation/tabular_pipeline.py',artifacts=[{'artifact_type':'experiment_record','path':'experiments/house_prices/20260623_225551/experiment_record.json'},{'artifact_type':'submission','path':'experiments/house_prices/20260623_225551/submission.csv'}],metrics={'rmsle':baseline_score},cv_score=baseline_score,decision='promote',created_at=now,metric_name='rmsle',metric_direction='minimize',promoted=True))
graph.add_node(ExperimentNode(exp_id='EXP_HOUSE_REGRESSION_ENSEMBLE_WR_1209BFA4',parent_id='EXP_BASELINE_HOUSE_20260623_225551',branch_type='ensemble',task_name='house_prices',hypothesis='RF+HGB+ET regression stack may improve RMSLE over baseline.',implementation_summary='Regression-aware ensemble route through workstation API.',code_path='scripts/run_local_sklearn_ensemble.py',artifacts=[{'artifact_type':'metrics','path':'experiments/house_prices/wr_2026-06-23T23-17-07.433686_1209bfa4/metrics.json'},{'artifact_type':'oof_predictions','path':'experiments/house_prices/wr_2026-06-23T23-17-07.433686_1209bfa4/oof_predictions.csv'},{'artifact_type':'submission','path':'experiments/house_prices/wr_2026-06-23T23-17-07.433686_1209bfa4/submission.csv'},{'artifact_type':'artifact_manifest','path':'experiments/house_prices/wr_2026-06-23T23-17-07.433686_1209bfa4/artifact_manifest.json'}],metrics={'rmsle':candidate_score},cv_score=candidate_score,risk_flags=['candidate_not_better_than_parent'],created_at=now,metric_name='rmsle',metric_direction='minimize'))
graph.add_edge('EXP_BASELINE_HOUSE_20260623_225551','EXP_HOUSE_REGRESSION_ENSEMBLE_WR_1209BFA4','workstation self-evolution round after regression ensemble fix')
decision=graph.decide_promotion('EXP_HOUSE_REGRESSION_ENSEMBLE_WR_1209BFA4',metric='rmsle',direction='minimize',required_artifacts=['metrics.json','submission.csv','artifact_manifest.json'])
graph_path=graph.export_json(OUT_DIR/'house_prices_score_guard_search_graph_20260623.json')
decision_path=OUT_DIR/'house_prices_promotion_decision_20260623.json'
decision_path.write_text(json.dumps({'schema':'academic_research_os.score_promotion_gate.v1','created_at':now,'decision':decision,'invariant':'best-so-far is monotonic under declared metric direction; candidate not improving is held, not promoted'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'decision':decision,'graph':str(graph_path.relative_to(ROOT)).replace('\\','/'),'decision_path':str(decision_path.relative_to(ROOT)).replace('\\','/')},ensure_ascii=False,indent=2))
