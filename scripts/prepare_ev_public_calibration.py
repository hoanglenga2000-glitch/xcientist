"""Prepare exclusive local protocol artifacts. Does not download, train or submit."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from research_os.official_calibration import freeze_protocol, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/ev-public-calibration-20260908')
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = json.loads((ROOT/'configs/benchmarks/ev_public_calibration_20260908.json').read_text(encoding='utf-8'))
    digest = freeze_protocol(protocol, out/'agent-input/protocol.json')
    planned = [{'arm':arm,'seed':seed,'status':'not_started','oof_roc_auc':None,
                'official_public_score':None,'official_private_score':None,'actual_gpu_hours':0}
               for arm in protocol['arms'] for seed in protocol['run_seeds']]
    receipt = {'schema':'evomind.official_calibration.preparation.v1','created_at':datetime.now(timezone.utc).isoformat(),
               'status':'protocol_prepared_training_not_started','protocol_sha256':digest,
               'planned_runs':planned,'gpu_hours_consumed':0,'new_paid_resources':False,
               'official_submissions':0,'isolated_runtime_dispatched':False,
               'data_manifest':None,'fold_manifest':None,
               'pending_gates':['competition_entry_and_authorized_data','current_managed_resource_identity','runtime_isolation_and_budget_enforcement'],
               'modified_source_hashes':{p:sha256_file(ROOT/p) for p in [
                    'src/research_agent_workstation/tabular_pipeline.py','scripts/onboard_kaggle_competition.py',
                    'scripts/validate_tabular_experiment.py','src/research_os/official_calibration.py']}}
    with (out/'preparation.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
    prompt = '''Use EvoMind's existing managed research workflow to solve the task described in agent-task-contract.json.
The full operator protocol is not an agent input; do not read it or the baseline configuration.
Read only the official task/rules, the authorized competition data, the frozen split manifest,
and generic library documentation. Do not inspect this competition's notebooks, team solutions,
leaderboard scores, or historical task answers. Never send raw rows, labels or predictions to an
external control model. Learn all preprocessing inside each training fold. Produce complete OOF
and test positive-class probabilities, save models, fit events, resource usage and intervention
records. Select candidates by complete OOF AUC, with actual runtime breaking ties. Freeze all
artifacts before any official score is disclosed. Do not submit to Kaggle or accept rules.
This prompt is prepared input only: execution requires verified managed resource identity,
enforced runtime access boundaries and budget caps. It is not evidence that a run has happened.
'''
    with (out/'agent-input/task.txt').open('x',encoding='utf-8') as stream:
        stream.write(prompt)
    print(json.dumps({'status':receipt['status'],'output':str(out),'protocol_sha256':digest,
                      'planned_runs':len(planned),'training_started':False,'official_submissions':0},ensure_ascii=False))


if __name__ == '__main__':
    main()
