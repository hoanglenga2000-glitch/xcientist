"""Exercise filesystem/network guards using synthetic sentinels, not test labels."""
import json,os,tempfile
from pathlib import Path

def main():
    base=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra')
    root=Path(tempfile.mkdtemp(prefix='siim-isolation-probe-',dir=str(base)))
    public=root/'public';hidden=root/'hidden';work=root/'work'
    for p in [public,hidden,work]:p.mkdir()
    (public/'example.txt').write_text('public');(hidden/'sentinel.txt').write_text('synthetic-not-a-label')
    import socket
    network=block_internet()
    guard=restrict_filesystem([str(public)],[str(work)],[])
    checks={'public_read':(public/'example.txt').read_text()=='public'}
    for name,action in [('hidden_read',lambda:(hidden/'sentinel.txt').read_text()),
                        ('hidden_list',lambda:list(hidden.iterdir())),
                        ('outside_write',lambda:(root/'outside.txt').write_text('x'))]:
        try:action()
        except PermissionError:checks[name+'_denied']=True
        else:checks[name+'_denied']=False
    (work/'permitted.txt').write_text('allowed');checks['workspace_write']=True
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,**guard,**network,
            'test_data_used':False,'gpu_used':False,'probe_artifacts_retained':str(root)}
    print(json.dumps(result));return 0 if result['status']=='passed' else 2

if __name__=='__main__':raise SystemExit(main())
