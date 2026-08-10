"""Machine-facing CLI for the workstation Multi-Agent run API."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import replace
from pathlib import Path

from research_os.agent.aibuild_v1 import read_current_run_pointer
from research_os.agent.llm_finetune_workflow import resume_llm_finetune, run_llm_finetune
from research_os.agent.llm_refinement_workflow import (
    create_refinement_plan,
    decide_refinement_plan,
    reserve_refinement_run,
    resume_llm_refinement,
    run_llm_refinement,
)
from research_os.agent.local_tabular_workflow import resume_local_tabular_research, run_local_tabular_research
from research_os.agent.siim_hpc_workflow import resume_siim_hpc_research, run_siim_hpc_research
from research_os.agent.titanic_workflow import resume_titanic_aibuild, run_titanic_aibuild
from research_os.siim_hpc_binding import validate_binding

from .config import active_root
from .user_request import parse_user_request


def _bind_hpc_allocation(
    request,
    *,
    job_id: int | None,
    credential_profile: str | None,
    resource_profile: str | None,
    execution_backend: str | None,
):
    """Persist the exact allocation selected by the API instead of an ambient default."""

    selected_job = str(job_id or "").strip()
    selected_profile = str(credential_profile or "").strip()
    selected_resource = str(resource_profile or "").strip()
    selected_backend = str(execution_backend or "").strip()
    values = {
        "job_id": selected_job,
        "credential_profile": selected_profile,
        "resource_profile": selected_resource,
        "execution_backend": selected_backend,
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing HPC execution contract: {', '.join(missing)}")
    if selected_backend != "hpc" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", selected_resource):
        raise ValueError("Invalid HPC execution contract")
    binding = validate_binding(selected_job, selected_profile)
    request.compute_policy = replace(
        request.compute_policy,
        backend="hpc",
        local_gpu_allowed=False,
        remote_gpu_required=True,
        gpu_count=1,
        job_id=binding.job_id,
        credential_profile=binding.credential_profile,
        resource_profile=selected_resource,
    )
    return request


def _read_events(path: Path, after_seq: int) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(event.get("seq") or 0) > after_seq:
            events.append(event)
    return events


def _run_dir(root: Path, run_id: str | None = None) -> Path:
    if run_id:
        candidate = root / "workspace" / "evomind_runs" / run_id
    else:
        pointer = read_current_run_pointer(root)
        if not pointer:
            raise FileNotFoundError("current_run.json is missing or invalid")
        candidate = root / str(pointer["run_dir"])
    resolved = candidate.resolve()
    resolved.relative_to(root.resolve())
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _task_type(run_dir: Path) -> str:
    if (run_dir / "version.json").is_file() and (run_dir / "refinement.json").is_file():
        return "llm_refinement"
    request_path = run_dir / "request.json"
    if not request_path.is_file():
        return "general"
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "general"
    return str(request.get("task_type") or "general")


def _dataset(run_dir: Path) -> str | None:
    request_path = run_dir / "request.json"
    if not request_path.is_file():
        return None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = request.get("dataset")
    return str(value) if value is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--request-file", required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--hpc-job-id", type=int)
    run_parser.add_argument("--hpc-credential-profile")
    run_parser.add_argument("--hpc-resource-profile")
    run_parser.add_argument("--execution-backend")
    run_parser.add_argument("--parent-run-id")
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--run-id")
    event_parser = subparsers.add_parser("events")
    event_parser.add_argument("--run-id")
    event_parser.add_argument("--after-seq", type=int, default=0)
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--run-id", required=True)
    refine_plan_parser = subparsers.add_parser("refine-plan")
    refine_plan_parser.add_argument("--parent-run-id", required=True)
    refine_plan_parser.add_argument("--prompt-file", required=True)
    refine_plan_parser.add_argument("--refinement-id")
    refine_plan_parser.add_argument("--task-id", default="evomind-qwen7b-finetune")
    refine_decide_parser = subparsers.add_parser("refine-decide")
    refine_decide_parser.add_argument("--refinement-id", required=True)
    refine_decide_parser.add_argument("--decision", choices=("approve", "reject"), required=True)
    refine_decide_parser.add_argument("--task-id", default="evomind-qwen7b-finetune")
    refine_reserve_parser = subparsers.add_parser("refine-reserve")
    refine_reserve_parser.add_argument("--refinement-id", required=True)
    refine_reserve_parser.add_argument("--run-id", required=True)
    refine_reserve_parser.add_argument("--task-id", default="evomind-qwen7b-finetune")
    refine_run_parser = subparsers.add_parser("refine-run")
    refine_run_parser.add_argument("--refinement-id", required=True)
    refine_run_parser.add_argument("--run-id", required=True)
    refine_run_parser.add_argument("--task-id", default="evomind-qwen7b-finetune")
    control_parser = subparsers.add_parser("control")
    control_parser.add_argument("action", choices=("pause", "resume", "cancel"))
    control_parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    root = active_root()

    if args.command == "run":
        request_path = Path(args.request_file).resolve()
        request_path.relative_to(root.resolve())
        request = parse_user_request(request_path.read_text(encoding="utf-8"))
        is_siim_request = (
            request.task_type == "image_classification"
            and request.dataset == "siim-isic-melanoma-classification"
        )
        if args.parent_run_id and not is_siim_request:
            raise ValueError("--parent-run-id is supported only for the SIIM child workflow")
        has_hpc_contract_input = any(
            value is not None
            for value in (
                args.hpc_job_id,
                args.hpc_credential_profile,
                args.hpc_resource_profile,
                args.execution_backend,
            )
        )
        requires_hpc_contract = (
            is_siim_request
            or request.compute_policy.backend == "hpc"
            or request.compute_policy.remote_gpu_required
            or has_hpc_contract_input
        )
        if requires_hpc_contract:
            request = _bind_hpc_allocation(
                request,
                job_id=args.hpc_job_id,
                credential_profile=args.hpc_credential_profile,
                resource_profile=args.hpc_resource_profile,
                execution_backend=args.execution_backend,
            )
        if request.task_type == "llm_finetune":
            result = run_llm_finetune(root, request, run_id=args.run_id)
        elif (
            request.task_type == "image_classification"
            and request.dataset == "siim-isic-melanoma-classification"
            and request.compute_policy.backend == "hpc"
            and request.submission_policy.official_submission == "forbidden"
        ):
            siim_run_kwargs = {"run_id": args.run_id}
            if args.parent_run_id:
                siim_run_kwargs["parent_run_id"] = args.parent_run_id
            result = run_siim_hpc_research(root, request, **siim_run_kwargs)
        elif (
            request.task_type == "tabular_classification"
            and request.dataset == "credit-card-fraud-detection"
            and request.compute_policy.backend == "local_gpu"
        ):
            result = run_local_tabular_research(root, request, run_id=args.run_id)
        elif request.dataset in {None, "titanic"}:
            result = run_titanic_aibuild(root, request, run_id=args.run_id)
        else:
            raise ValueError(
                f"No governed Multi-Agent workflow for task_type={request.task_type!r}, dataset={request.dataset!r}"
            )
        print(
            json.dumps(
                {
                    "ok": result.status == "completed",
                    "run_id": result.run_id,
                    "status": result.status,
                    "seq": result.seq,
                    "open_requirements": result.open_requirements,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.status == "completed" else 2
    if args.command == "refine-plan":
        prompt_path = Path(args.prompt_file).resolve()
        prompt_path.relative_to(root.resolve())
        plan = create_refinement_plan(
            root,
            parent_run_id=args.parent_run_id,
            prompt=prompt_path.read_text(encoding="utf-8"),
            refinement_id=args.refinement_id,
            task_id=args.task_id,
        )
        print(json.dumps({"ok": True, "refinement": plan}, ensure_ascii=False))
        return 0
    if args.command == "refine-decide":
        plan = decide_refinement_plan(
            root,
            args.refinement_id,
            args.decision,
            task_id=args.task_id,
        )
        print(json.dumps({"ok": True, "refinement": plan}, ensure_ascii=False))
        return 0
    if args.command == "refine-reserve":
        plan, should_start = reserve_refinement_run(
            root,
            args.refinement_id,
            args.run_id,
            task_id=args.task_id,
        )
        print(json.dumps({"ok": True, "refinement": plan, "should_start": should_start}, ensure_ascii=False))
        return 0
    if args.command == "refine-run":
        result = run_llm_refinement(
            root,
            args.refinement_id,
            run_id=args.run_id,
            task_id=args.task_id,
        )
        print(
            json.dumps(
                {
                    "ok": result.status == "completed",
                    "run_id": result.run_id,
                    "status": result.status,
                    "seq": result.seq,
                    "open_requirements": result.open_requirements,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.status == "completed" else 2
    if args.command == "resume":
        run_dir = _run_dir(root, args.run_id)
        if _task_type(run_dir) == "llm_refinement":
            result = resume_llm_refinement(root, args.run_id)
        elif _task_type(run_dir) == "llm_finetune":
            result = resume_llm_finetune(root, args.run_id)
        elif _task_type(run_dir) == "tabular_classification":
            result = resume_local_tabular_research(root, args.run_id)
        elif _task_type(run_dir) == "image_classification" and _dataset(run_dir) == "siim-isic-melanoma-classification":
            result = resume_siim_hpc_research(root, args.run_id)
        else:
            result = resume_titanic_aibuild(root, args.run_id)
        print(
            json.dumps(
                {
                    "ok": result.status == "completed",
                    "run_id": result.run_id,
                    "status": result.status,
                    "seq": result.seq,
                    "open_requirements": result.open_requirements,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.status == "completed" else 2

    run_dir = _run_dir(root, args.run_id)
    if args.command == "snapshot":
        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.command == "events":
        print(json.dumps({"events": _read_events(run_dir / "events.jsonl", args.after_seq)}, ensure_ascii=False))
        return 0
    control_path = run_dir / "control.json"
    control_path.write_text(
        json.dumps({"schema": "evomind.multi_agent.control.v1", "action": args.action}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "run_id": run_dir.name, "action": args.action}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
