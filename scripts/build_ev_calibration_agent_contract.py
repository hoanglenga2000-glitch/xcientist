"""Produce the allowlisted agent view without operator baseline or leaderboard information."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from research_os.official_calibration import sha256_file


def main():
    out=ROOT/'artifacts/ev-public-calibration-20260908'
    full=json.loads((out/'agent-input/protocol.json').read_text(encoding='utf-8'))
    data=json.loads((out/'private-data/manifest.json').read_text(encoding='utf-8'))
    view={k:full[k] for k in ['competition','competition_url','target','id_column','metric','fold_count','split_seed','run_seeds','budget','isolation']}
    view.update(schema='evomind.official_calibration.agent_task.v1',positive_raw_label='Yes',positive_encoded_label=1,
                target_encoding=data['quality']['target_encoding'],
                source_manifest_sha256=sha256_file(out/'private-data/manifest.json'),
                input_files={k:v for k,v in data['files'].items() if k.endswith('.csv')},
                controller_may_read_operator_protocol=False,baseline_configuration_disclosed=False,
                local_test_results_are_not_official_scores=True,official_submission_allowed=False,
                required_outputs=['complete_oof_probabilities','test_positive_probabilities','saved_models',
                                  'fit_events','resource_usage','interventions','reload_verification'],
                selection={'metric':'complete_oof_roc_auc','direction':'maximize','tie_breaker':'shorter_actual_runtime'})
    path=out/'agent-input/agent-task-contract.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(view,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'agent_view_prepared_not_dispatched','sha256':sha256_file(path),
                      'baseline_and_leaderboard_details_excluded':True}))


if __name__=='__main__':main()
