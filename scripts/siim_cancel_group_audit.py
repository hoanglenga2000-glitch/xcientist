"""Read-only confirmation that the previously signalled SIIM process group exited."""
import json,os
from pathlib import Path
pgid=1534666
remaining=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  fields=(p/'stat').read_text().split(') ',1)[1].split()
  if int(fields[2])==pgid and fields[0] not in {'Z','X'}:
   remaining.append({'pid':int(p.name),'state':fields[0],'owner_matches':p.stat().st_uid==os.geteuid()})
 except (OSError,ValueError):continue
print(json.dumps({'process_group':pgid,'remaining':remaining,'all_owned_processes_exited':not remaining,'signals_sent':0}))
