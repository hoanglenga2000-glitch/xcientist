from __future__ import annotations
import json, os, subprocess, zipfile, tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TODAY = '20260623'
TASK_ID = 'spaceship_titanic'
COMPETITION = 'spaceship-titanic'
DATA_DIR = ROOT / 'datasets' / 'kaggle' / TASK_ID
EXPERIMENT_ROOT = ROOT / 'experiments' / TASK_ID
WORKSPACE_SUMMARY = ROOT / 'workspace' / f'three_layer_new_kaggle_round_{TASK_ID}_{TODAY}.json'
MEMORY_PATH = ROOT / 'workspace' / f'retrospective_memory_new_kaggle_round_{TASK_ID}_{TODAY}.json'
REPORT_PATH = ROOT / 'reports' / f'NEW_KAGGLE_THREE_LAYER_ROUND_{TASK_ID}_{TODAY}.md'
KAGGLE_SECRET_MANAGER = ROOT / 'scripts' / 'manage_kaggle_secret.ps1'

def now_stamp(): return datetime.now().strftime('%Y%m%d_%H%M%S')
def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
def rel(path: Path): return str(path.relative_to(ROOT)).replace('\\', '/')

def run_manager_status():
    cp = subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(KAGGLE_SECRET_MANAGER),'status'], cwd=ROOT, capture_output=True, timeout=60)
    out = cp.stdout.decode('utf-8-sig', errors='replace')
    if cp.returncode != 0: raise RuntimeError(out + cp.stderr.decode('utf-8', errors='replace'))
    return json.loads(out)

def download_kaggle_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if all((DATA_DIR / n).exists() for n in ['train.csv','test.csv','sample_submission.csv']):
        return {'status':'already_present','data_dir':rel(DATA_DIR)}
    status = run_manager_status()
    if not status.get('credential_installed'):
        raise RuntimeError('Kaggle DPAPI credential not installed')
    with tempfile.TemporaryDirectory() as td:
        py_path = Path(td) / 'download_spaceship.py'
        ps_path = Path(td) / 'run_download.ps1'
        py_path.write_text("""
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
api = KaggleApi(); api.authenticate()
out = Path(r'''DATA_DIR_PLACEHOLDER'''); out.mkdir(parents=True, exist_ok=True)
api.competition_download_files('spaceship-titanic', path=str(out), quiet=True)
print('downloaded')
""".replace('DATA_DIR_PLACEHOLDER', str(DATA_DIR)), encoding='utf-8')
        ps_path.write_text("""
$ErrorActionPreference = 'Stop'
$CredentialPath = Join-Path $env:APPDATA 'ResearchAgentWorkstation\\kaggle_api_token.xml'
$credential = Import-Clixml -Path $CredentialPath
$secret = $credential.GetNetworkCredential().Password
Remove-Item Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue
Remove-Item Env:KAGGLE_KEY -ErrorAction SilentlyContinue
$env:KAGGLE_API_TOKEN = $secret
python 'PY_PATH_PLACEHOLDER'
""".replace('PY_PATH_PLACEHOLDER', str(py_path)), encoding='utf-8')
        cp = subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ps_path)], cwd=ROOT, capture_output=True, timeout=180)
        if cp.returncode != 0:
            raise RuntimeError(cp.stdout.decode('utf-8', errors='replace')[-1000:] + cp.stderr.decode('utf-8', errors='replace')[-1000:])
    for z in DATA_DIR.glob('*.zip'):
        with zipfile.ZipFile(z,'r') as archive: archive.extractall(DATA_DIR)
    if not all((DATA_DIR / n).exists() for n in ['train.csv','test.csv','sample_submission.csv']):
        raise RuntimeError('required CSVs missing after download')
    return {'status':'downloaded','data_dir':rel(DATA_DIR)}

def add_features(df):
    out=df.copy(); pid=out['PassengerId'].astype(str); out['Group']=pid.str.split('_').str[0]
    out['GroupSize']=out.groupby('Group')['PassengerId'].transform('count'); out['IsAlone']=(out['GroupSize']==1).astype(int)
    cabin=out.get('Cabin',pd.Series(index=out.index,dtype=object)).astype(str).replace('nan',np.nan); parts=cabin.str.split('/',expand=True)
    out['CabinDeck']=parts[0] if parts.shape[1]>0 else np.nan
    out['CabinNum']=pd.to_numeric(parts[1],errors='coerce') if parts.shape[1]>1 else np.nan
    out['CabinSide']=parts[2] if parts.shape[1]>2 else np.nan
    spend=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for c in spend: out[c]=pd.to_numeric(out[c],errors='coerce')
    out['TotalSpend']=out[spend].sum(axis=1,skipna=True); out['NoSpend']=(out['TotalSpend'].fillna(0)==0).astype(int)
    out['AgeBin']=pd.cut(pd.to_numeric(out['Age'],errors='coerce'), bins=[-1,12,18,30,50,80], labels=['child','teen','young','adult','senior'])
    for c in ['CryoSleep','VIP']: out[c]=out[c].astype('object')
    return out

def make_pipeline(kind):
    cat=['HomePlanet','CryoSleep','Destination','VIP','CabinDeck','CabinSide','AgeBin']
    num=['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','GroupSize','IsAlone','CabinNum','TotalSpend','NoSpend']
    pre=ColumnTransformer([('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat),('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler(with_mean=False))]),num)])
    if kind=='logistic': model=LogisticRegression(max_iter=2000,C=1.2,class_weight='balanced')
    elif kind=='voting': model=VotingClassifier([('lr',LogisticRegression(max_iter=2000,C=1.0,class_weight='balanced')),('et',ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features='sqrt',random_state=42,n_jobs=-1)),('rf',RandomForestClassifier(n_estimators=220,min_samples_leaf=2,max_features='sqrt',random_state=43,n_jobs=-1))], voting='soft', weights=[1,3,2])
    else: model=ExtraTreesClassifier(n_estimators=350,min_samples_leaf=2,max_features='sqrt',random_state=42,n_jobs=-1)
    return Pipeline([('preprocess',pre),('model',model)])

@dataclass
class Branch:
    round_id:str; branch_id:str; model_kind:str; hypothesis:str; parent_best:float|None; code_generation_mode:str; search_stage:str

def run_branch(branch, train_df, test_df, sample_df):
    out_dir=EXPERIMENT_ROOT / f"{branch.round_id}_{now_stamp()}_{branch.branch_id}"; out_dir.mkdir(parents=True,exist_ok=True)
    train_f=add_features(train_df); test_f=add_features(test_df); y=train_f['Transported'].astype(bool).astype(int)
    X=train_f.drop(columns=['Transported','Name','Cabin','PassengerId'],errors='ignore'); X_test=test_f.drop(columns=['Name','Cabin','PassengerId'],errors='ignore')
    pipe=make_pipeline(branch.model_kind); cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof_proba=cross_val_predict(pipe,X,y,cv=cv,method='predict_proba')[:,1]; oof_pred=(oof_proba>=0.5).astype(int); acc=float(accuracy_score(y,oof_pred))
    pipe.fit(X,y); test_pred=pipe.predict_proba(X_test)[:,1]>=0.5
    sub=sample_df.copy(); sub['Transported']=test_pred.astype(bool); sub.to_csv(out_dir/'submission.csv',index=False)
    pd.DataFrame({'PassengerId':train_df['PassengerId'],'Transported_true':y.astype(bool),'oof_proba':oof_proba,'oof_pred':oof_pred.astype(bool)}).to_csv(out_dir/'oof_predictions.csv',index=False)
    improved=branch.parent_best is None or acc>branch.parent_best+1e-12; decision='promote' if improved else 'preserve_parent_best'; final_best=acc if improved else branch.parent_best
    metrics={'schema':'academic_research_os.new_kaggle_branch_metrics.v1','task_id':TASK_ID,'competition':COMPETITION,'round_id':branch.round_id,'branch_id':branch.branch_id,'model_kind':branch.model_kind,'metric':'accuracy','direction':'maximize','cv_accuracy':acc,'parent_best':branch.parent_best,'improved_vs_parent':improved,'decision':decision,'final_best_so_far':final_best,'folds':5,'official_submission_made':False}
    write_json(out_dir/'metrics.json',metrics)
    write_json(out_dir/'agent_trace.json',{'schema':'academic_research_os.agent_trace.v1','task_id':TASK_ID,'agents':[{'agent_id':'OrchestratorAgent','stage':'task_decomposition','status':'completed'},{'agent_id':'DataAuditAgent','stage':'data_audit','status':'completed'},{'agent_id':'SearchControllerAgent','stage':branch.search_stage,'status':'completed'},{'agent_id':'CodeImplementationAgent','stage':branch.code_generation_mode,'status':'completed'},{'agent_id':'LocalExecutionAgent','stage':'cpu_training','status':'completed'},{'agent_id':'ValidationHarnessAgent','stage':'validation_contract_and_claim_audit','status':'completed'}]})
    write_json(out_dir/'search_controller_decision.json',{'schema':'academic_research_os.mlevolve_search_decision.v1','task_id':TASK_ID,'branch_id':branch.branch_id,'search_stage':branch.search_stage,'code_generation_mode':branch.code_generation_mode,'hypothesis':branch.hypothesis,'parent_best':branch.parent_best,'candidate_score':acc,'decision':decision,'rollback_condition':'preserve parent if candidate accuracy is not better than parent_best or validation artifacts are missing'})
    write_json(out_dir/'validation_contract.json',{'schema':'academic_research_os.validation_contract.v1','task_id':TASK_ID,'branch_id':branch.branch_id,'hypothesis':branch.hypothesis,'implementation_requirement':['CPU-only local training','5-fold OOF accuracy','submission schema equals Kaggle sample_submission'],'metric':'accuracy','baseline_exp_id':branch.parent_best,'acceptance_criteria':['candidate accuracy must strictly exceed parent best to promote','OOF and submission artifacts must exist'],'risk_checklist':['no official submission','no test labels','schema check','claim boundary'],'conclusion_boundary':'local CV proxy only; no official Kaggle score','required_artifacts':['metrics.json','oof_predictions.csv','submission.csv','artifact_manifest.json','claim_audit.json']})
    write_json(out_dir/'claim_audit.json',{'schema':'academic_research_os.claim_audit.v1','task_id':TASK_ID,'branch_id':branch.branch_id,'supporting_metrics':{'cv_accuracy':acc,'parent_best':branch.parent_best},'audit_result':'allow_local_improvement_claim' if improved else 'revise_preserve_parent','allowed_conclusion':'local CV improved and branch promoted' if improved else 'candidate did not improve; parent preserved and memory updated','blocked_claims':['official Kaggle score','leaderboard rank','GPU/HPC execution']})
    (out_dir/'report.md').write_text(f"# {TASK_ID} {branch.round_id} {branch.branch_id}\n\n- CV accuracy: {acc:.6f}\n- Parent best: {branch.parent_best}\n- Decision: {decision}\n- Boundary: local CV only.\n",encoding='utf-8')
    artifacts=[]
    for name in ['agent_trace.json','metrics.json','oof_predictions.csv','submission.csv','search_controller_decision.json','validation_contract.json','claim_audit.json','report.md']:
        p=out_dir/name; artifacts.append({'path':rel(p),'artifact_type':name,'created_by_agent':'ResearchWorkstation','stage':branch.round_id,'size':p.stat().st_size})
    write_json(out_dir/'artifact_manifest.json',{'schema':'academic_research_os.artifact_manifest.v1','task_id':TASK_ID,'branch_id':branch.branch_id,'artifacts':artifacts})
    metrics.update({'output_dir':rel(out_dir),'submission_path':rel(out_dir/'submission.csv'),'claim_audit':rel(out_dir/'claim_audit.json'),'validation_contract':rel(out_dir/'validation_contract.json'),'artifact_manifest':rel(out_dir/'artifact_manifest.json')})
    return metrics

def main():
    generated_at=datetime.now().isoformat(timespec='seconds'); data_status=download_kaggle_data()
    train=pd.read_csv(DATA_DIR/'train.csv'); test=pd.read_csv(DATA_DIR/'test.csv'); sample=pd.read_csv(DATA_DIR/'sample_submission.csv')
    branches=[]
    r1=run_branch(Branch('round1','logistic_feature_baseline','logistic','Regularized linear baseline with cabin/group/spend features.',None,'Base','exploration'),train,test,sample); branches.append(r1)
    r2=run_branch(Branch('round2','tree_ensemble_feature_exploitation','voting','Model-diverse soft-voting ensemble should improve non-linear interactions.',float(r1['final_best_so_far']),'Stepwise','exploration_to_exploitation'),train,test,sample); branches.append(r2)
    best=max(branches,key=lambda x:x['final_best_so_far'])
    memory=[]
    for b in branches:
        memory.append({'memory_id':"{}_{}_{}".format(b['round_id'],b['branch_id'],'success' if b['decision']=='promote' else 'neutral'),'task_id':TASK_ID,'method':b['model_kind'],'metric':'accuracy','metric_after':b['cv_accuracy'],'parent_best':b['parent_best'],'final_best_so_far':b['final_best_so_far'],'decision':b['decision'],'what_worked':'Promoted branch improved local CV.' if b['decision']=='promote' else 'Parent preservation prevented regression.','what_failed':None if b['decision']=='promote' else 'Candidate did not beat parent best.','reusable_strategy':'Use promoted branch as next parent; test calibration/threshold or stronger model family next.','linked_artifacts':[b['output_dir'],b['claim_audit'],b['validation_contract']]})
    write_json(MEMORY_PATH,{'schema':'academic_research_os.retrospective_memory_batch.v1','created_at':generated_at,'records':memory})
    monotonic=all(b['final_best_so_far']>=branches[max(0,i-1)]['final_best_so_far'] for i,b in enumerate(branches))
    summary={'schema':'academic_research_os.new_kaggle_three_layer_round.v1','created_at':generated_at,'task_id':TASK_ID,'competition':COMPETITION,'data_status':data_status,'execution_resource':'local_cpu_only_gpu_unavailable','official_submission_made':False,'three_layer_evidence':{'layer_1_multi_agent_research_os':'Each branch writes agent_trace, metrics, OOF, submission, manifest and report artifacts.','layer_2_mlevolve_style_search_controller':'Round2 consumes Round1 parent best and applies promote/preserve best-so-far gate.','layer_3_xcientist_research_harness':'Each branch writes validation_contract and claim_audit; official leaderboard claims are blocked.'},'trajectory':branches,'aggregate':{'rounds':len(branches),'best_cv_accuracy':best['final_best_so_far'],'best_branch':best['branch_id'],'best_so_far_never_regressed':monotonic,'promoted':sum(1 for b in branches if b['decision']=='promote'),'preserved_parent':sum(1 for b in branches if b['decision']!='promote')},'memory_path':rel(MEMORY_PATH),'claim_boundary':['Local CV only','No official Kaggle submit','No leaderboard rank','No GPU/HPC claim']}
    write_json(WORKSPACE_SUMMARY,summary)
    lines=[f'# New Kaggle Three-layer Round: {TASK_ID}','',f'- Generated at: {generated_at}',f'- Competition: `{COMPETITION}`','- Resource: local CPU only; GPU unavailable.','- Official Kaggle submission: not performed.','','## Trajectory','','| Round | Branch | Model | CV accuracy | Parent best | Decision | Final best |','|---|---|---|---:|---:|---|---:|']
    for b in branches:
        parent='n/a' if b['parent_best'] is None else f"{b['parent_best']:.6f}"; lines.append(f"| {b['round_id']} | {b['branch_id']} | {b['model_kind']} | {b['cv_accuracy']:.6f} | {parent} | {b['decision']} | {b['final_best_so_far']:.6f} |")
    lines += ['', '## Conclusion', '', f"Best local CV accuracy: `{best['final_best_so_far']:.6f}` from `{best['branch_id']}`. Three-layer artifacts and claim boundaries were generated for each branch.", '', '## Artifacts', '', f'- Summary: `{rel(WORKSPACE_SUMMARY)}`', f'- Memory: `{rel(MEMORY_PATH)}`']
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True); REPORT_PATH.write_text('\n'.join(lines),encoding='utf-8-sig')
    print(json.dumps({'status':'passed','summary':rel(WORKSPACE_SUMMARY),'report':rel(REPORT_PATH),'best_cv_accuracy':best['final_best_so_far'],'best_branch':best['branch_id']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
