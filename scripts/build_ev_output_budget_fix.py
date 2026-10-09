from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/ev-public-calibration-20260908/output-budget-fix'
raw=(OUT/'runtime-before.py').read_bytes();text=raw.decode()
needle='                turn = client.send(messages, system=system, tools=specs)'
replacement='                turn = client.send(messages, system=system, tools=specs,\n                                   max_tokens=8192 if "official_calibration" in (session.get("metadata") or {}) else 4096)'
assert text.count(needle)==1
candidate=text.replace(needle,replacement,1).encode()
assert candidate.decode().replace(replacement,needle,1).encode()==raw
(OUT/'runtime.py').write_bytes(candidate)
sha=lambda b:hashlib.sha256(b).hexdigest()
m={'baseline_runtime_sha256':sha(raw),'baseline_seal_sha256':sha((OUT/'seal-before.json').read_bytes()),'candidate_sha256':sha(candidate),'scope':'EV-only output budget 4096 to 8192; same model and timeout'}
(OUT/'manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps({'manifest_sha256':sha((OUT/'manifest.json').read_bytes())}))
