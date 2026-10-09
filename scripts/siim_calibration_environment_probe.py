"""Small HPC-side read-only probe; no model fitting and no GPU allocation."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra')

def main():
    pythons=[ROOT/'mlebench_lite_runtime/venv-uv/bin/python',
        ROOT/'siim_job90353/runtime/abebff4efcc16309f4b48e0a5032b39cab9814193da3b67f61f608fc7814c677/venv/bin/python']
    checks=[]
    program="import json,sys,importlib.metadata; import torch,torchvision;print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda}))"
    for python in pythons:
        row={'python_path':str(python),'exists':python.exists()}
        if python.exists():
            env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            done=subprocess.run([str(python),'-c',program],env=env,capture_output=True,text=True,timeout=45)
            row['exit_code']=done.returncode
            if done.returncode==0:row['imports']=json.loads(done.stdout.splitlines()[-1])
            else:row['error_log_sha256']=hashlib.sha256((done.stdout+done.stderr).encode()).hexdigest()
        checks.append(row)
    cache=ROOT/'mlebench_model_cache/torch/hub/checkpoints'
    weights=[]
    for name in ['convnext_tiny-983f1562.pth','convnext_small-0c510722.pth','efficientnet_v2_s-dd5fe13b.pth']:
        p=cache/name
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            digest=h.hexdigest();weights.append({'name':name,'bytes':p.stat().st_size,'sha256':digest,'official_filename_prefix_match':digest.startswith(name.rsplit('-',1)[1].split('.')[0])})
    print(json.dumps({'runtime_checks':checks,'weights':weights,'kernel':os.uname().release,'effective_uid':os.geteuid(),
                      'sandbox_tools':{name:shutil.which(name) is not None for name in ['bwrap','unshare','docker']},'private_labels_read':False,'training_started':False}))

if __name__=='__main__':main()
