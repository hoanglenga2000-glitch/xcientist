"""Resolve the fixed public object locally; send only a private descriptor to the server."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evomind_runtime.managed_model_download import (
    CDN_ASSET, EXPECTED_SHARDS, REPO, REVISION, validate_cdn_descriptor,
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def resolve_descriptor() -> bytes:
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{CDN_ASSET}?download=true"
    opener = urllib.request.build_opener(NoRedirect())
    try:
        response = opener.open(urllib.request.Request(url, method="HEAD", headers={"Cache-Control": "no-cache"}), timeout=20)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        size, digest = EXPECTED_SHARDS[CDN_ASSET]
        if response.code not in (302, 303, 307, 308) or response.headers.get("X-Repo-Commit") != REVISION or response.headers.get("X-Linked-Etag", "").strip('"') != digest or response.headers.get("X-Linked-Size") != str(size):
            raise ValueError("official_redirect_identity_mismatch")
        target = response.headers.get("Location", "")
    payload = json.dumps({"schema": "evomind.gptoss_cdn_descriptor.v1", "repo": REPO, "revision": REVISION,
                          "name": CDN_ASSET, "bytes": size, "sha256": digest, "url": target}, sort_keys=True).encode()
    validate_cdn_descriptor(payload, minimum_remaining=1300)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-package-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"run_[a-f0-9]{32}", args.run_id):
            raise ValueError("run_identity_invalid")
        payload = resolve_descriptor()
        public = validate_cdn_descriptor(payload, minimum_remaining=1300)
        if args.dry_run:
            print(json.dumps({"status": "validated", "sha256": public["sha256"], "expires_at": public["expires_at"], "signed_url_emitted": False}))
            return 0
        remote = r'''
import sys,json,pathlib
envelope=json.load(sys.stdin)
package=pathlib.Path(envelope['package_root']).resolve(strict=True)
package.relative_to(pathlib.Path('C:/ProgramData/EvoMind/staging').resolve(strict=True))
sys.path.insert(0,str(package))
from evomind_runtime.model_descriptor_store import seal_descriptor
run=envelope['run_id']
import re
if not re.fullmatch(r'run_[a-f0-9]{32}',run):raise ValueError('run_identity_invalid')
workspace=pathlib.Path('C:/ProgramData/EvoMind/data/workspace/runtime/assistant_tasks')/run
result=seal_descriptor(workspace,envelope['payload'].encode())
print(json.dumps({'status':'sealed','run_id':run,**result,'signed_url_emitted':False}))
'''
        encoded = base64.b64encode(remote.encode()).decode()
        command = 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe -c "import base64;exec(base64.b64decode(\'' + encoded + '\'))"'
        envelope = {"package_root": args.runtime_package_root, "run_id": args.run_id, "payload": payload.decode()}
        result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "evomind-shanghai", command],
                                input=json.dumps(envelope).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise ValueError("private_descriptor_stage_failed")
        received = json.loads(result.stdout.decode())
        if received.get("sha256") != public["sha256"]:
            raise ValueError("private_descriptor_stage_identity_mismatch")
        print(json.dumps(received))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_class": type(error).__name__, "signed_url_emitted": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
