"""Inspect the single run-owned dependency installer; no broad process actions."""
import hashlib,json,os
from pathlib import Path
run='run_131be3f4a6f3469fb73f7a0b5547fc66'
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or p.name==str(os.getpid()):continue
 try:
  command=(p/'cmdline').read_bytes()
  if run.encode() not in command or b'managed_dependency_lock.py' not in command:continue
  parts=command.split(b'\0');s=(p/'stat').read_text().split(') ',1)[1].split()
  rows.append({'pid':int(p.name),'parent_pid':int(s[1]),'process_group':int(s[2]),'start_ticks':s[19],
               'executable':Path(parts[0].decode()).name,'cmdline_sha256':hashlib.sha256(command).hexdigest(),'state':s[0]})
 except (OSError,ValueError):pass
print(json.dumps({'run':run,'owned_installers':rows,'signals_sent':0}))
