"""V2 launch support activation/rollback uses only disposable fixture files."""
import json
from pathlib import Path

import pytest

from test_syscomplete_activation_transaction import prepared, Operations, put, mutate_reference, tx


@pytest.fixture
def prepared_v2(prepared):
    value = prepared
    root = Path(value['root'])
    launcher_path = Path(value['launcher']['path'])
    launcher = json.loads(launcher_path.read_text())
    launcher.update(schema='evomind.launcher_candidate.v2', activation_claim=False)
    targets = {
        'candidate/Start-Node.ps1': root/tx.LAUNCHER_REL,
        'candidate/lib/verify_report_dependencies.py': root/'bundle/scripts/lib/verify_report_dependencies.py',
        'candidate/report-dependency-activation.json': Path(value['dependencies']['target']).parent/'report-dependency-activation.json',
    }
    rows=[]
    for name, target in targets.items():
        source=launcher_path.parent/name
        if not source.exists():
            source.parent.mkdir(parents=True,exist_ok=True)
            source.write_bytes(b'fixture_support_only')
        rows.append({'path':name,'target':str(target),'sha256':tx.sha(source)})
    launcher['deployment_files']=rows
    value['launcher']=put(launcher_path,launcher)
    value['baseline']['launcher_support_files']=[{'target':target.relative_to(root).as_posix(),'sha256':None}
        for name,target in targets.items() if name!='candidate/Start-Node.ps1']
    mutate_reference(value['delivery'],lambda d:d['external_launcher_binding'].update(
        manifest_sha256=value['launcher']['sha256'],deployment_files=rows))
    mutate_reference(value['dependencies']['launcher_binding'],lambda d:d.update(launcher_manifest_sha256=value['launcher']['sha256']))
    delivery=json.loads(Path(value['delivery']['path']).read_text())
    checks=[{'name':name,'passed':True} for name in sorted(tx.required_report_export_checks())]
    next(row for row in checks if row['name']=='report_ack_p95').update(p95_ms=50,samples=20)
    report={'schema':'evomind.report_export_http_acceptance.v1','status':'passed','fixture_only':True,
        'production_unchanged':True,'report_site_unchanged':True,'cleanup':'owned_fixture_processes_stopped',
        'build_id':delivery['build_id'],'web_sha256':delivery['web']['sha256'],
        'runtime_sha256':delivery['runtime']['sha256'],'checks':checks}
    value['report_exports']=put(Path(value['stage'])/'report-export-gate.json',report)
    return value


def test_v2_support_copied_before_start_and_bound_to_runtime(prepared_v2):
    spec=prepared_v2
    plan=tx.validate(spec)
    assert len(plan['launcher_support_files'])==2 and plan['gates']['report_exports']['checks']==133
    class CheckedOperations(Operations):
        def preflight(self, plan):
            super().preflight(plan)
            for row in plan['launcher_support_files']:
                target=self.root/row['target']
                assert target.exists() == row['dependency_binding']
        def start(self):
            for row in plan['launcher_support_files']:
                assert tx.sha(self.root/row['target']) == row['after_sha256']
            super().start()
    result=tx.activate(spec,plan,CheckedOperations(spec))
    assert result['status']=='canary_validation_pending'


@pytest.mark.parametrize('phase',['runtime_switched','launcher_support_switched','launcher_switched','ready'])
def test_v2_rollback_preserves_new_support_in_backup_not_active_bundle(prepared_v2,phase):
    spec=prepared_v2
    plan=tx.validate(spec)
    result=tx.activate(spec,plan,Operations(spec,phase))
    assert result['status']=='HOLD' and result['research_database_restored'] is False
    for row in plan['launcher_support_files']:
        assert not (Path(spec['root'])/row['target']).exists()
    assert tx.sha(Path(spec['root'])/tx.LAUNCHER_REL)==spec['baseline']['launcher_sha256']


@pytest.mark.parametrize('change',['missing','duplicate','wrong_target','wrong_hash'])
def test_v2_support_identity_rejected_before_mutation(prepared_v2,change):
    def mutate(launcher):
        rows=launcher['deployment_files']
        if change=='missing': rows.pop()
        elif change=='duplicate': rows[2]=rows[1]
        elif change=='wrong_target': rows[1]['target']=str(Path(prepared_v2['root'])/'config/outside.py')
        else: rows[1]['sha256']='0'*64
    mutate_reference(prepared_v2['launcher'],mutate)
    with pytest.raises(tx.Hold): tx.validate(prepared_v2)
    assert not (Path(prepared_v2['root'])/'backups').exists()


@pytest.mark.parametrize('change',['missing_check','duplicate_check','wrong_build','slow','false_pass'])
def test_v2_report_gate_is_not_optional_or_summary_only(prepared_v2,change):
    def mutate(report):
        if change=='missing_check': report['checks'].pop()
        elif change=='duplicate_check': report['checks'][-1]=report['checks'][0]
        elif change=='wrong_build': report['build_id']='other'
        elif change=='slow': next(r for r in report['checks'] if r['name']=='report_ack_p95')['p95_ms']=1001
        else: report['checks'][0]['passed']='true'
    mutate_reference(prepared_v2['report_exports'],mutate)
    with pytest.raises(tx.Hold): tx.validate(prepared_v2)
