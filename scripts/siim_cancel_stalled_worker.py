"""Cancel only the identity-checked GNU timeout supervisor of the stalled SIIM attempt."""
import hashlib,json,os,select,signal,time,ctypes
from pathlib import Path
RUN='run_131be3f4a6f3469fb73f7a0b5547fc66'
def snapshot(pid):
    p=Path('/proc')/str(pid)
    if p.stat().st_uid!=os.geteuid():raise ValueError('process_owner_differs')
    argv=(p/'cmdline').read_bytes().split(b'\0');argv=[x.decode() for x in argv if x]
    fields=(p/'stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'argv':argv,'pgid':int(fields[2]),'start':fields[19],'state':fields[0]}
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:r=snapshot(int(p.name))
 except (OSError,ValueError,UnicodeError):continue
 if not r['argv'] or Path(r['argv'][0]).name!='timeout':continue
 scripts=[x for x in r['argv'] if x.endswith('/train_gpu.py') and RUN in x]
 if len(scripts)==1 and scripts[0].startswith('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/') and r['pid']==r['pgid']:
  rows.append(r)
if len(rows)!=1:
 print(json.dumps({'status':'blocked','reason':'owned_timeout_not_unique','matches':len(rows),'signals_sent':0}));raise SystemExit(2)
target=rows[0]
if os.uname().machine!='x86_64':raise ValueError('pidfd_architecture_unverified')
libc=ctypes.CDLL(None,use_errno=True)
fd=libc.syscall(434,target['pid'],0)
if fd<0:raise ValueError('pidfd_open_failed')
try:
 if snapshot(target['pid'])!=target:raise ValueError('timeout_identity_changed')
 # GNU timeout is the existing bounded-process supervisor; SIGTERM forwards to its own group.
 if libc.syscall(424,fd,signal.SIGTERM,ctypes.c_void_p(0),0)<0:raise ValueError('pidfd_signal_failed')
 exited=bool(select.select([fd],[],[],15)[0])
 survivors=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:r=snapshot(int(p.name))
  except (OSError,ValueError,UnicodeError):continue
  if r['pgid']==target['pgid'] and r['state'] not in {'Z','X'}:survivors.append(r['pid'])
 print(json.dumps({'status':'cancelled' if exited and not survivors else 'settlement_required','pid':target['pid'],
   'start_ticks':target['start'],'argv_sha256':hashlib.sha256(json.dumps(target['argv']).encode()).hexdigest(),
   'signals_sent':1,'exit_observed':exited,'remaining_group_pids':survivors,'other_processes_modified':False}))
finally:os.close(fd)
