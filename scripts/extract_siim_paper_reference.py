"""Verify author-published grading records by pinned Git-LFS hashes before citing.

Historical records are collected verbatim, including failures. They are not
our results, not an aggregate paper score, and not same-budget comparisons.
"""
from __future__ import annotations
import csv,hashlib,io,json,re,urllib.request
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PIN='636c7d218492973710dd7069ab176800fe87fb10'
EXPERIMENT='models-o1-preview-aide'
BASE='https://raw.githubusercontent.com/openai/mle-bench/'+PIN+'/'
INDEX='https://media.githubusercontent.com/media/openai/mle-bench/'+PIN+'/runs/run_group_experiments.csv'

def fetch(url,limit=2*1024*1024):
    with urllib.request.urlopen(url,timeout=30) as response:raw=response.read(limit+1)
    if len(raw)>limit:raise ValueError('public_source_size_exceeded')
    return raw

def verify_cached_lfs(pointer:bytes,cached:bytes):
    text=pointer.decode('utf-8').strip()
    match=re.fullmatch(r'version https://git-lfs.github.com/spec/v1\noid sha256:([a-f0-9]{64})\nsize ([0-9]+)',text.replace('\r\n','\n'))
    if match is None:raise ValueError('invalid_or_conflicted_lfs_pointer')
    if len(cached)!=int(match[2]) or hashlib.sha256(cached).hexdigest()!=match[1]:raise ValueError('cached_public_source_does_not_match_upstream')
    return match[1]

def main():
    out=ROOT/'artifacts/siim-mlebench-calibration-20260908/paper-reference'
    out.mkdir(exist_ok=False)
    index_raw=fetch(INDEX)
    index=list(csv.DictReader(io.StringIO(index_raw.decode('utf-8'))))
    groups=sorted({row['run_group'] for row in index if row['experiment_id']==EXPERIMENT})
    if not groups:raise ValueError('paper_experiment_mapping_missing')
    records=[];sources=[]
    for group in groups:
        if not re.fullmatch('[A-Za-z0-9_-]+',group):raise ValueError('upstream_group_path_invalid')
        files=sorted((ROOT/'external-projects/mle-bench/runs'/group).glob('*grading_report.json'))
        if len(files)!=1:raise ValueError('published_report_file_not_unique')
        path=files[0];relative='runs/'+group+'/'+path.name;url=BASE+relative
        pointer=fetch(url);cached=path.read_bytes()
        expected=verify_cached_lfs(pointer,cached)
        report=json.loads(cached)
        source={'run_group':group,'url':url,'sha256':expected,'bytes':len(cached),'local_hash_verified_against_pinned_official_lfs':True}
        sources.append(source)
        with (out/(group+'.lfs.txt')).open('xb') as f:f.write(pointer)
        for position,row in enumerate(report['competition_reports']):
            if row['competition_id']!='siim-isic-melanoma-classification':continue
            records.append({'run_group':group,'record_index':position,'metric':'roc_auc','score':row.get('score'),
                'valid_submission':row['valid_submission'],'created_at':row.get('created_at'),
                'gold_threshold':row['gold_threshold'],'silver_threshold':row['silver_threshold'],
                'bronze_threshold':row['bronze_threshold'],'median_threshold':row['median_threshold'],
                'source_url':url,'source_sha256':expected})
    result={'schema':'evomind.siim_paper_reference.v1','checked_at':datetime.now(timezone.utc).isoformat(),
        'experiment_id':EXPERIMENT,'agent':'AIDE','model':'o1-preview','paper_repository_commit':PIN,
        'index_url':INDEX,'index_sha256':hashlib.sha256(index_raw).hexdigest(),'sources':sources,'records':records,
        'interpretation':'Author-published individual SIIM records. Includes extra historical retries and failures. Not the paper aggregate; no selective best-run claim.',
        'same_budget_as_current_experiment':False,'training_data_variant_matches_current':False,
        'current_evomind_score':None,'current_evomind_official_rank':None}
    with (out/'reference.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    with (out/'index.csv').open('xb') as f:f.write(index_raw)
    with (out/'individual-records.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps({'status':'paper_records_verified','source_reports':len(sources),'siim_records':len(records),
        'valid_records':sum(r['valid_submission'] for r in records),'scores':[r['score'] for r in records],'output':str(out)}))

if __name__=='__main__':main()
