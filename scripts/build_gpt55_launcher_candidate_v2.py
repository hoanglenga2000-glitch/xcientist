"""Build a local-only, immutable EvoMind launcher candidate; never deploy it.

Inputs are exact already-adjudicated files, not the dirty runtime worktree.
The generated launch binding is evidence of the intended installation and
must not be mistaken for proof it exists at its permanent target.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re


REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts/system-completeness-20260907"
V1 = ARTIFACTS / "gpt55-launcher-candidate-v1"
DEFAULT_OUTPUT = ARTIFACTS / "gpt55-launcher-candidate-v2"
SITE = "C:/ProgramData/EvoMind/report-envs/gpt55-806f87b009cd/site-packages"
BASE = "C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv"
MANIFEST_TARGET = str(PureWindowsPath(SITE).parent / "report-dependency-activation.json").replace("\\", "/")
INPUTS = {
    "baseline": (V1 / "baseline/Start-Node.ps1", "0bf9f0a69f2f2f15377cb446a49b12c5811fd9f107633864b9fd6596215a5e22"),
    "v1": (V1 / "candidate/Start-Node.ps1", "4af54177257c6cb2336246cfc41863e6413e10e3e1c00afb1a82e63b99c24c6d"),
    "v1_manifest": (V1 / "candidate-manifest.json", "297cc879c53ecd0af4f65e882100fb46bb0d854684a4ca9f3ec75d6fc1eb231d"),
    "runtime": (V1 / "baseline/lib/Runtime.ps1", "667e1fbd38a4b5cbf1e63c19e04bbe17e141adcbaa38e8a91b25f1f23e66d233"),
    "descriptor": (ARTIFACTS / "report-dependency-delivery-v1.json", "806f87b009cdec68878f822f78421390e2291a6efcdcfc08c62a6ba9b1cce076"),
    "inheritance": (ARTIFACTS / "report-inheritance-v1.json", "fb462e54fe90aaec64d11282dd6457a1ca30c1c61d8c3f7a1edd2befb3373a19"),
    "installed": (ARTIFACTS / "report-delivered-v2-proof/installed-files.json", "4f88b11e811c54cbeb8a2561ad61d596b7fae96e7d9727c25974039f227fee8f"),
    "install": (ARTIFACTS / "report-delivered-v2-proof/install-receipt.json", "63a9c5819d2a459d944e8752180932ba4022df4f46349520c14b027cb3ab328f"),
    "render": (ARTIFACTS / "report-delivered-v2-proof/render-receipt.json", "c0d265702514998cc888ad4bf13cf995ae72ca932a5189178df8817741e5c7d0"),
}
OLD_MODEL_CHECK = "  if (@($models.data | Where-Object id -eq 'gpt-5.6-sol').Count -ne 1) { throw 'LLM_MODEL_ROUTE_MISSING' }\n"
NEW_MODEL_CHECK = """  # BEGIN EVOMIND_GATEWAY_CATALOG_LIVENESS_V2
  # The authenticated gateway remains a managed compatibility service. Its
  # advertised catalog proves liveness/schema only, not the direct GPT-5.5 route.
  $gatewayModelIds = @($models.data | ForEach-Object { [string]$_.id })
  if ([string]$models.object -ne 'list' -or $gatewayModelIds.Count -lt 1 -or
      @($gatewayModelIds | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0 -or
      @($gatewayModelIds | Select-Object -Unique).Count -ne $gatewayModelIds.Count) {
    throw 'LLM_GATEWAY_CATALOG_HEALTH_FAILED'
  }
  # END EVOMIND_GATEWAY_CATALOG_LIVENESS_V2
"""
PROFILE_APPEND = """  # BEGIN EVOMIND_REPORT_PYTHONPATH_V2
  # Keep the frozen runtime first; load the exact report wheel installation
  # before release/base packages. The gate above verified all 15 import origins.
  $baseEnv['PYTHONPATH'] = $byoaPythonRoot + ';' + $reportSitePackages + ';' + (Join-Path $release 'src')
  $baseEnv['PYTHONDONTWRITEBYTECODE'] = '1'
  $baseEnv['PYTHONNOUSERSITE'] = '1'
  $baseEnv['MPLCONFIGDIR'] = Join-Path $data 'cache\\report-matplotlib'
  $baseEnv['EVOMIND_REPORT_DEPENDENCY_BINDING_PATH'] = $reportBindingPath
  $baseEnv['EVOMIND_REPORT_DEPENDENCY_BINDING_SHA256'] = $reportBindingSha256
  # END EVOMIND_REPORT_PYTHONPATH_V2
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def require(value, code):
    if not value:
        raise ValueError(code)


def checked_inputs():
    result = {}
    for key, (path, expected) in INPUTS.items():
        data = path.read_bytes()
        require(sha(data) == expected, "source_hash_mismatch_" + key)
        result[key] = data if key in {"baseline", "v1", "runtime"} else json.loads(data)
    return result


def make_binding(inputs):
    for phase in ("install", "render"):
        receipt = inputs[phase]
        require(receipt["schema"] == "evomind.report_dependency_delivery_acceptance.v1"
                and receipt["phase"] == phase and receipt["status"] == "staged_not_activated"
                and receipt["activated"] is False and receipt["production_changed"] is False
                and receipt["base_venv_unchanged"] is True
                and receipt["base_tree_sha256_before"] == receipt["base_tree_sha256_after"]
                and receipt["model_requests"] == receipt["hpc_actions"] == 0,
                "delivery_receipt_not_qualified")
        for key in ("descriptor", "inheritance"):
            require(receipt[key + "_sha256"] == INPUTS[key][1], "delivery_receipt_input_mismatch")
        require(receipt["installed_manifest_sha256"] == INPUTS["installed"][1], "installed_manifest_binding_mismatch")
    require(inputs["installed"]["file_count"] == len(inputs["installed"]["files"]) == 2187
            and sum(len(row["files"]) for row in inputs["inheritance"]["packages"]) == 573,
            "dependency_file_count_mismatch")
    require(inputs["install"]["imports"] == inputs["render"]["imports"], "delivery_import_receipts_disagree")
    imports = []
    for row in inputs["install"]["imports"]:
        old_root = PureWindowsPath(inputs["install"]["target"]) if row["source"] == "wheelhouse" else PureWindowsPath(BASE)
        rel = PureWindowsPath(row["module_path"]).relative_to(old_root).as_posix()
        imports.append({key: row[key] for key in ("module", "name", "version", "source", "module_sha256")} | {"relative_path": rel})
    return {
        "schema": "evomind.report_dependency_launch_binding.v1", "frozen": True,
        "purpose": "intended_immutable_dependency_installation_not_activation_proof",
        "deployment": {"site_packages": SITE, "base_venv": BASE, "manifest_path": MANIFEST_TARGET,
                       "python_executable": BASE + "/Scripts/python.exe",
                       "pythonpath_policy": "bundle_runtime;exact_report_site_packages;release_src"},
        "abi": inputs["install"]["preflight"]["abi"],
        "provenance": {key + "_sha256": INPUTS[key][1] for key in ("descriptor", "inheritance", "installed", "install", "render")},
        "descriptor": inputs["descriptor"], "installed": inputs["installed"],
        "inheritance": inputs["inheritance"], "imports": imports,
    }


def dependency_gate(verifier_hash, binding_hash):
    return f"""# BEGIN EVOMIND_REPORT_DEPENDENCY_GATE_V2
# Read-only integrity gate before any managed service/HPC bridge starts.
$reportSitePackages = '{SITE}'
$reportBindingPath = '{MANIFEST_TARGET}'
$reportBindingSha256 = '{binding_hash}'
$reportVerifier = Join-Path $PSScriptRoot 'lib\\verify_report_dependencies.py'
if (-not (Test-Path -LiteralPath $reportVerifier -PathType Leaf) -or
    (Get-Sha256File $reportVerifier) -ne '{verifier_hash}') {{
  throw 'REPORT_DEPENDENCY_VERIFIER_INTEGRITY_FAILED'
}}
$reportProofText = & $python -I -S -B $reportVerifier --manifest $reportBindingPath --manifest-sha256 $reportBindingSha256 --runtime-root $byoaPythonRoot --release-root $release 2>$null
if ($LASTEXITCODE -ne 0) {{ throw 'REPORT_DEPENDENCY_STARTUP_GATE_FAILED' }}
$reportProof = $reportProofText | ConvertFrom-Json
if ($reportProof.schema -ne 'evomind.report_dependency_launch_proof.v1' -or
    $reportProof.status -ne 'verified' -or $reportProof.binding_sha256 -ne $reportBindingSha256 -or
    [IO.Path]::GetFullPath([string]$reportProof.site_packages) -ne [IO.Path]::GetFullPath($reportSitePackages) -or
    $reportProof.installed_file_count -ne 2187 -or $reportProof.inherited_file_count -ne 573 -or
    $reportProof.import_origin_count -ne 15 -or $reportProof.model_readiness_claim -ne $false -or
    $reportProof.release_acceptance_claim -ne $false) {{ throw 'REPORT_DEPENDENCY_STARTUP_PROOF_INVALID' }}
# END EVOMIND_REPORT_DEPENDENCY_GATE_V2

"""


def candidate_bytes(inputs, verifier_hash, binding_hash):
    source = inputs["v1"].decode("utf-8")
    gate = dependency_gate(verifier_hash, binding_hash)
    require(source.count("$records = @()\n") == source.count(OLD_MODEL_CHECK) == 1,
            "launcher_context_not_unique")
    marker = "  # END EVOMIND_GPT55_PROFILE_V1\n"
    require(source.count(marker) == 1, "profile_marker_not_unique")
    candidate = source.replace("$records = @()\n", gate + "$records = @()\n", 1)
    candidate = candidate.replace(marker, marker + PROFILE_APPEND, 1)
    candidate = candidate.replace(OLD_MODEL_CHECK, NEW_MODEL_CHECK, 1)
    reverted = candidate.replace(gate, "", 1).replace(PROFILE_APPEND, "", 1).replace(NEW_MODEL_CHECK, OLD_MODEL_CHECK, 1)
    require(reverted.encode("utf-8") == inputs["v1"], "unrelated_launcher_bytes_changed")
    return candidate.encode("utf-8")


def build(output=DEFAULT_OUTPUT):
    output = Path(output)
    require(output.resolve().is_relative_to(ARTIFACTS.resolve()), "local_output_scope_rejected")
    require(not (output / "candidate-manifest.json").exists(), "sealed_candidate_must_not_be_rewritten")
    inputs = checked_inputs()
    verifier = (DEFAULT_OUTPUT / "source/verify_report_dependencies.py").read_bytes()
    binding = canonical(make_binding(inputs))
    candidate = candidate_bytes(inputs, sha(verifier), sha(binding))
    patch = "".join(difflib.unified_diff(inputs["baseline"].decode().splitlines(True), candidate.decode().splitlines(True),
                                       fromfile="a/bundle/scripts/Start-Node.ps1", tofile="b/bundle/scripts/Start-Node.ps1")).encode()
    files = {
        "candidate/Start-Node.ps1": candidate,
        "candidate/lib/verify_report_dependencies.py": verifier,
        "candidate/report-dependency-activation.json": binding,
        "Start-Node.gpt55-report.patch": patch,
    }
    require(all(not (output / path).exists() for path in files), "candidate_file_already_exists")
    for path, content in files.items():
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(content)
    manifest = {
        "schema": "evomind.launcher_candidate.v2", "status": "local_candidate_not_activated",
        "production_changed": False, "activation_claim": False,
        "parent_candidate_sha256": INPUTS["v1"][1],
        "parent_manifest_sha256": INPUTS["v1_manifest"][1],
        "baseline": [{"path": str(path), "sha256": digest} for name, (path, digest) in INPUTS.items() if name in {"baseline", "runtime"}],
        "candidate": {"path": "candidate/Start-Node.ps1", "target": "C:/ProgramData/EvoMind/bundle/scripts/Start-Node.ps1", "sha256": sha(candidate)},
        "files": [{"path": name, "sha256": sha(data), "bytes": len(data)} for name, data in files.items()],
        "deployment_files": [
            {"path": "candidate/Start-Node.ps1", "target": "C:/ProgramData/EvoMind/bundle/scripts/Start-Node.ps1", "sha256": sha(candidate)},
            {"path": "candidate/lib/verify_report_dependencies.py", "target": "C:/ProgramData/EvoMind/bundle/scripts/lib/verify_report_dependencies.py", "sha256": sha(verifier)},
            {"path": "candidate/report-dependency-activation.json", "target": MANIFEST_TARGET, "sha256": sha(binding)}],
        "dependency_binding": {"schema": "evomind.report_dependency_launch_binding.v1", "path": "candidate/report-dependency-activation.json",
                               "target": MANIFEST_TARGET, "sha256": sha(binding), "site_packages": SITE,
                               "source_site_packages": inputs["install"]["target"], "installed_manifest_sha256": INPUTS["installed"][1],
                               "installed_file_count": 2187, "inherited_file_count": 573},
        "profile": inputs["v1_manifest"]["profile"],
        "preserved_health_checks": ["bundle_integrity", "runtime_artifact_integrity", "gateway_binary_provenance", "gateway_listener",
                                    "gateway_authenticated_catalog", "runtime_listener", "runtime_authenticated_health", "web_listener", "web_ready_exact_build_no_failures"],
        "catalog_check_scope": "authenticated gateway liveness/schema only; no Sol requirement and no model readiness claim",
        "activation_requirements": ["fresh baseline and owner proxy checks", "immutable dependency copy and full file readback",
                                    "bundle manifest/seal includes helper", "gate executed by dedicated owner with production interpreter",
                                    "qualified model endurance", "safe transactional activation", "real Chrome E2E and actual training proof"],
    }
    with (output / "candidate-manifest.json").open("xb") as handle:
        handle.write(canonical(manifest))
    return {"candidate_root": str(output), "manifest_sha256": sha(canonical(manifest)),
            "candidate_sha256": sha(candidate), "binding_sha256": sha(binding), "verifier_sha256": sha(verifier), "activated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=True))
