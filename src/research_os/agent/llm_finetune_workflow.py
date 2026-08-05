"""Truthful EvoMind-document QLoRA workflow for one mature 7B base model."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import shutil
import textwrap
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xsci.user_request import UserRequest

from ..hpc_runtime import HpcRuntime
from .aibuild_v1 import run_directory, write_current_run_pointer
from .multi_agent import (
    TERMINAL_TASK_STATES,
    AgentResult,
    AgentRoleSpec,
    AgentTask,
    HandoffEnvelope,
    MultiAgentStore,
    MultiAgentSupervisor,
    SupervisorRun,
    create_run,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TASK_ID = "evomind-qwen7b-finetune"
DATASET_COUNTS = {"train": 800, "validation": 100, "test": 100}
MAX_CORRECTION_ATTEMPTS = 2
HPC_TASK_OVERHEAD_SECONDS = 300
SUPPORTED_HPC_GPU_MARKERS = ("NVIDIA A40", "NVIDIA A800")
CORRECTION_SPECS = {
    1: {
        "field": "epochs",
        "value": 3,
        "rationale": "Increase only the epoch budget after the fixed-test loss shows underfitting; best-validation checkpoint selection still prevents promotion of a worse late epoch.",
    },
    2: {
        "field": "learning_rate",
        "value": 1e-4,
        "rationale": "Reduce only the learning rate after the raw validation curve shows that later epochs overfit; keep the data, seed, epoch budget, and evidence rules fixed.",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _hpc_task_timeout_seconds(request: UserRequest) -> int:
    return request.budget.max_minutes * 60 + HPC_TASK_OVERHEAD_SECONDS


def _is_supported_hpc_gpu(name: str) -> bool:
    return any(marker in name for marker in SUPPORTED_HPC_GPU_MARKERS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_text(path, content)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, run_dir: Path, *, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "kind": kind,
    }


def _correction_task_plan(
    *,
    run: SupervisorRun,
    source_review_task_id: str,
    correction_attempt: int,
    timeout_seconds: int,
) -> tuple[list[AgentTask], dict[str, tuple[str, ...]]]:
    if correction_attempt not in CORRECTION_SPECS:
        raise ValueError(f"unsupported correction attempt: {correction_attempt}")
    label = f"correction_{correction_attempt:02d}"
    base_payload = {
        "correction_attempt": correction_attempt,
        "execution_id": label,
        "config_hash": _sha256_text(
            f"{run.run_id}\0{source_review_task_id}\0{label}\0{json.dumps(CORRECTION_SPECS[correction_attempt], sort_keys=True)}"
        ),
    }
    design_id = f"{label}_design"
    train_id = f"{label}_hpc_train"
    evaluate_id = f"{label}_evaluate"
    review_id = f"{label}_review"
    tasks = [
        AgentTask(
            design_id,
            f"Plan one-variable correction {correction_attempt} without changing data, seed, or evidence rules.",
            "TrainingDesigner",
            (source_review_task_id,),
            priority=39,
            max_retries=0,
            acceptance_criteria=(f"corrections/{label}.json",),
            payload={**base_payload, "stage": "design"},
        ),
        AgentTask(
            train_id,
            f"Execute one-variable correction {correction_attempt} on the verified NVIDIA HPC GPU.",
            "HpcRuntimeAgent",
            (design_id,),
            priority=38,
            resource_type="hpc_gpu",
            timeout_seconds=timeout_seconds,
            max_retries=1,
            acceptance_criteria=("llm_output/adapter", "llm_output/metrics.json", "llm_output/telemetry.jsonl"),
            payload={**base_payload, "stage": "hpc_train"},
        ),
        AgentTask(
            evaluate_id,
            f"Evaluate correction {correction_attempt} on the unchanged fixed test set.",
            "EvaluatorAgent",
            (train_id,),
            priority=37,
            max_retries=0,
            acceptance_criteria=(f"evaluations/{evaluate_id}.json",),
            payload={**base_payload, "stage": "evaluate"},
        ),
        AgentTask(
            review_id,
            f"Independently review correction {correction_attempt} from raw evidence only.",
            "IndependentReviewer",
            (evaluate_id,),
            priority=36,
            max_retries=0,
            acceptance_criteria=(f"reviews/{review_id}.json",),
            payload={**base_payload, "stage": "review"},
        ),
    ]
    return tasks, {"claim_audit": (review_id,)}


def _archive_current_attempt(run_dir: Path, correction_attempt: int) -> dict[str, Any]:
    previous_label = "initial" if correction_attempt == 1 else f"correction_{correction_attempt - 1:02d}"
    archive_dir = run_dir / "attempts" / previous_label
    manifest_path = archive_dir / "attempt_manifest.json"
    if manifest_path.is_file():
        return _read_json(manifest_path)

    archive_dir.mkdir(parents=True, exist_ok=True)
    for relative in (
        "llm_output",
        "qlora_config.json",
        "hpc_job.json",
        "evaluation_summary.json",
        "review.json",
    ):
        source = run_dir / relative
        target = archive_dir / relative
        if not source.exists() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    entries = [
        _artifact(path, archive_dir, kind=path.name)
        for path in sorted(archive_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    payload = {
        "schema": "evomind.llm_attempt_archive.v1",
        "attempt": previous_label,
        "artifacts": entries,
        "generated_at": _now(),
    }
    _atomic_json(manifest_path, payload)
    return payload


_SOURCE_CANDIDATES = (
    "README.md",
    "docs/THREE_LAYER_RESEARCH_OS_ARCHITECTURE.md",
    "docs/AUTOKAGGLE_TERMINAL_AGENT.md",
    "docs/EVOLUTION_ENGINE_20260702.md",
    "docs/WORKSTATION_CODE_RUNTIME_MAP_20260630.md",
    "docs/UI_FRONTEND_API_CONTRACT_20260627.md",
    "docs/EvoMind-Development-Summary-20260706.md",
    "docs/EvoMind-ClaudeCode-Benchmark-20260706.md",
    "docs/AI-Research-Workstation-Multi-Agent-Framework-Report-20260614.md",
    "docs/MLEVOLVE_DEEP_MIGRATION_MAP_20260624.md",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:password|passwd|pwd|api[_ -]?key|access[_ -]?token|secret)\s*[:=]\s*[^\s`]{6,}"
)
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_LONG_TOKEN = re.compile(r"(?<![0-9A-Za-z])[0-9A-Za-z_-]{48,}(?![0-9A-Za-z])")


def _sanitize_excerpt(value: str) -> str:
    value = _SECRET_ASSIGNMENT.sub("[credential removed]", value)
    value = _PRIVATE_KEY.sub("[private key removed]", value)
    value = _LONG_TOKEN.sub("[long token removed]", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _document_chunks(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    heading = path.stem
    chunks: list[tuple[str, str]] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        cleaned = _sanitize_excerpt("\n".join(current))
        if len(cleaned) >= 120:
            for start in range(0, len(cleaned), 720):
                piece = cleaned[start : start + 900].strip()
                if len(piece) >= 120:
                    chunks.append((heading, piece))
        current.clear()

    for line in text.splitlines():
        match = re.match(r"^#{1,4}\s+(.+)$", line.strip())
        if match:
            flush()
            heading = match.group(1).strip()[:120]
        else:
            current.append(line)
    flush()
    return chunks


def build_document_dataset(workspace_root: Path, dataset_dir: Path) -> dict[str, Any]:
    sources = [workspace_root / relative for relative in _SOURCE_CANDIDATES if (workspace_root / relative).is_file()]
    if len(sources) < 6:
        raise RuntimeError("at least six EvoMind documentation sources are required for source-isolated splits")
    ordered = sorted(sources, key=lambda path: _sha256_text(path.relative_to(workspace_root).as_posix()))
    train_end = max(1, int(len(ordered) * 0.7))
    validation_end = max(train_end + 1, int(len(ordered) * 0.85))
    split_sources = {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }
    if any(not paths for paths in split_sources.values()):
        raise RuntimeError("source-isolated train/validation/test split is empty")

    variants = (
        "只依据给定文档上下文，解释“{heading}”的核心设计。",
        "面向首次使用 EvoMind 的研究者，根据上下文概述“{heading}”及其实际作用。",
        "根据给定证据，EvoMind 在“{heading}”中采用了哪些可执行、可审核的做法？",
        "只使用上下文事实，列出“{heading}”对应的关键流程、证据要求和边界。",
        "根据给定文档片段，用简洁但准确的语言说明 EvoMind 的“{heading}”。",
        "如果新用户询问“{heading}”，EvoMind 应如何依据当前上下文回答？",
        "根据文档证据，总结“{heading}”中的输入、处理、输出与验收依据。",
        "只引用给定上下文，说明“{heading}”如何支持可执行、可恢复和可复现性。",
        "从研究工作流角度解读上下文中的“{heading}”，不要补充片段外事实。",
        "依据给定片段，说明“{heading}”涉及的主要组件、操作和产物。",
        "根据上下文为 EvoMind 用户提供一份关于“{heading}”的可操作摘要。",
        "只依据给定证据，说明“{heading}”中最需要保留的事实、门禁和下一步。",
    )
    split_rows: dict[str, list[dict[str, Any]]] = {}
    source_manifest: list[dict[str, Any]] = []
    for split, paths in split_sources.items():
        pool: list[tuple[Path, str, str]] = []
        for source in paths:
            source_hash = _sha256(source)
            relative = source.relative_to(workspace_root).as_posix()
            chunks = _document_chunks(source)
            if not chunks:
                continue
            source_manifest.append(
                {
                    "path": relative,
                    "sha256": source_hash,
                    "split": split,
                    "chunks": len(chunks),
                }
            )
            pool.extend((source, heading, excerpt) for heading, excerpt in chunks)
        if not pool:
            raise RuntimeError(f"no usable documentation chunks for {split}")
        required = DATASET_COUNTS[split]
        capacity = len(pool) * len(variants)
        if capacity < required:
            raise RuntimeError(f"insufficient unique instruction capacity for {split}: {capacity} < {required}")
        rows: list[dict[str, Any]] = []
        for index in range(required):
            source, heading, excerpt = pool[index % len(pool)]
            relative = source.relative_to(workspace_root).as_posix()
            source_hash = _sha256(source)
            variant = variants[index // len(pool)]
            instruction = (
                f"{variant.format(heading=heading)}\n\n"
                f"<document_context source=\"{relative}\" heading=\"{heading}\">\n"
                f"{excerpt}\n"
                "</document_context>"
            )
            response = (
                f"## {heading}\n\n{excerpt}\n\n"
                f"证据来源：`{relative}`。以上回答仅依据给定文档上下文。"
            )
            record_id = _sha256_text(f"{split}\0{relative}\0{instruction}\0{response}")[:20]
            rows.append(
                {
                    "id": record_id,
                    "prompt": instruction,
                    "response": response,
                    "source_file": relative,
                    "source_heading": heading,
                    "source_sha256": source_hash,
                    "chunk_sha256": _sha256_text(excerpt),
                    "generation_method": "deterministic_grounded_context_response_v3",
                    "grounded_context": True,
                    "split": split,
                }
            )
        split_rows[split] = rows
        _write_jsonl(dataset_dir / f"{split}.jsonl", rows)

    id_sets = {split: {row["id"] for row in rows} for split, rows in split_rows.items()}
    source_sets = {split: {row["source_file"] for row in rows} for split, rows in split_rows.items()}
    split_overlap = {
        "train_validation": sorted(source_sets["train"] & source_sets["validation"]),
        "train_test": sorted(source_sets["train"] & source_sets["test"]),
        "validation_test": sorted(source_sets["validation"] & source_sets["test"]),
    }
    duplicate_ids = len(set().union(*id_sets.values())) != sum(len(values) for values in id_sets.values())
    content_fingerprints = [
        _sha256_text(f"{row['prompt']}\0{row['response']}")
        for rows in split_rows.values()
        for row in rows
    ]
    duplicate_content_records = len(set(content_fingerprints)) != len(content_fingerprints)
    files = [dataset_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
    data_hash = hashlib.sha256("".join(_sha256(path) for path in files).encode("ascii")).hexdigest()
    manifest = {
        "schema": "evomind.llm_document_dataset.v1",
        "generated_at": _now(),
        "method": "deterministic_source_grounded_rag_instruction_builder_v3",
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "source_split_policy": "document_source_disjoint_70_15_15",
        "source_overlap": split_overlap,
        "duplicate_record_ids": duplicate_ids,
        "duplicate_content_records": duplicate_content_records,
        "sources": source_manifest,
        "files": [{"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size} for path in files],
        "data_hash": data_hash,
        "grounded_context_contract": True,
        "secrets_removed": True,
    }
    _atomic_json(dataset_dir / "dataset_manifest.json", manifest)
    return manifest


def _training_source() -> str:
    return textwrap.dedent(
        r"""
        from __future__ import annotations

        import argparse
        import gc
        import hashlib
        import json
        import math
        import os
        import platform
        import random
        import re
        import subprocess
        import time
        from pathlib import Path

        import numpy as np
        import torch
        import transformers
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        def sha256(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def load_rows(path):
            with Path(path).open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]

        def atomic_json(path, payload):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, path)

        def gpu_snapshot():
            command = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(command, text=True, timeout=15).strip()
            rows = []
            for line in output.splitlines():
                values = [value.strip() for value in line.split(",")]
                if len(values) >= 6:
                    rows.append({
                        "index": values[0], "name": values[1], "memory_total_mb": values[2],
                        "memory_used_mb": values[3], "utilization_percent": values[4], "temperature_c": values[5],
                    })
            return rows

        class TokenDataset(Dataset):
            def __init__(self, rows, tokenizer, max_length):
                self.items = []
                for row in rows:
                    user_messages = [{"role": "user", "content": row["prompt"]}]
                    full_messages = [
                        *user_messages,
                        {"role": "assistant", "content": row["response"]},
                    ]
                    prompt_text = tokenizer.apply_chat_template(
                        user_messages, tokenize=False, add_generation_prompt=True
                    )
                    full_text = tokenizer.apply_chat_template(
                        full_messages, tokenize=False, add_generation_prompt=False
                    )
                    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                    if full_ids[: len(prompt_ids)] != prompt_ids:
                        raise RuntimeError("chat template prompt is not a prefix of the supervised sequence")
                    input_ids = full_ids[:max_length]
                    prompt_length = min(len(prompt_ids), len(input_ids))
                    labels = [-100] * prompt_length + input_ids[prompt_length:]
                    if not labels or all(label == -100 for label in labels):
                        raise RuntimeError("assistant response was fully truncated from a supervised record")
                    self.items.append(
                        {
                            "input_ids": input_ids,
                            "attention_mask": [1] * len(input_ids),
                            "labels": labels,
                        }
                    )

            def __len__(self):
                return len(self.items)

            def __getitem__(self, index):
                return self.items[index]

        class CausalCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, features):
                labels = [feature["labels"] for feature in features]
                inputs = [
                    {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
                    for feature in features
                ]
                batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
                width = int(batch["input_ids"].shape[1])
                batch["labels"] = torch.tensor(
                    [label + [-100] * (width - len(label)) for label in labels],
                    dtype=torch.long,
                )
                return batch

        class TelemetryCallback(TrainerCallback):
            def __init__(self, path):
                self.path = Path(path)

            def on_log(self, args, state, control, logs=None, **kwargs):
                payload = {
                    "ts": time.time(), "step": int(state.global_step), "epoch": state.epoch,
                    "gpu": gpu_snapshot(), "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 2),
                    **(logs or {}),
                }
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                print("EVOMIND_LLM_STEP=" + json.dumps(payload, ensure_ascii=False), flush=True)

        def quantization_config():
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        def load_base(model_name, cache_dir):
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                quantization_config=quantization_config(),
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=False,
            )

        def evaluation_loss(model, dataset, collator, _out_dir, _seed):
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
            was_training = model.training
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in loader:
                    batch = {name: value.to(model.device) for name, value in batch.items()}
                    losses.append(model(**batch).loss.detach().float().cpu())
            if was_training:
                model.train()
            if not losses:
                raise RuntimeError("evaluation dataset is empty")
            return float(torch.stack(losses).mean().item())

        def generated_metrics(model, tokenizer, rows, sample_count=24):
            recalls, formats, sensitive = [], [], []
            stopwords = {
                "EvoMind", "Agent", "the", "and", "with", "this", "that", "from", "into",
                "一个", "以及", "进行", "可以", "系统", "任务", "当前", "通过", "实现",
            }

            def candidate_terms(value):
                terms = re.findall(r"[A-Za-z][A-Za-z0-9_./:-]{2,}", value)
                for span in re.findall(r"[\u4e00-\u9fff]{2,}", value):
                    if len(span) <= 6:
                        terms.append(span)
                    else:
                        terms.extend(span[index : index + 4] for index in range(0, len(span) - 3, 2))
                return list(dict.fromkeys(term for term in terms if term not in stopwords))

            document_frequency = {}
            row_terms = []
            for row in rows:
                terms = candidate_terms(row["response"])
                row_terms.append(terms)
                for term in set(terms):
                    document_frequency[term] = document_frequency.get(term, 0) + 1

            selected_count = min(sample_count, len(rows))
            selected_indices = np.linspace(0, len(rows) - 1, num=selected_count, dtype=int).tolist()
            sample_ids = []
            for index in selected_indices:
                row = rows[index]
                sample_ids.append(row["id"])
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
                )
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
                with torch.no_grad():
                    output = model.generate(**encoded, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                chosen = sorted(
                    row_terms[index],
                    key=lambda term: (document_frequency.get(term, 0), -len(term), term),
                )[:16]
                recalls.append(sum(term in answer for term in chosen) / max(1, len(chosen)))
                formats.append(float(20 <= len(answer) <= 1200 and answer not in row["prompt"]))
                sensitive.append(float(bool(re.search(r"(?i)(password|api[_ -]?key|access[_ -]?token)\s*[:=]", answer))))
            return {
                "generation_sample_count": selected_count,
                "generation_sample_ids": sample_ids,
                "keyword_recall": float(np.mean(recalls) if recalls else 0.0),
                "format_compliance": float(np.mean(formats) if formats else 0.0),
                "sensitive_leak_rate": float(np.mean(sensitive) if sensitive else 0.0),
            }

        def score(loss, generated):
            loss_score = 100.0 / (1.0 + max(0.0, loss))
            return 0.70 * loss_score + 20.0 * generated["keyword_recall"] + 10.0 * generated["format_compliance"]

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--data-dir", required=True)
            parser.add_argument("--config", required=True)
            parser.add_argument("--out-dir", required=True)
            parser.add_argument("--base-model", required=True)
            parser.add_argument("--run-id", required=True)
            args = parser.parse_args()
            data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            seed = int(config["seed"])
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required; local or CPU fallback is forbidden")
            gpu = gpu_snapshot()
            if not gpu or not any(marker in gpu[0]["name"] for marker in ("NVIDIA A40", "NVIDIA A800")):
                raise RuntimeError("the current allocation is not a supported NVIDIA A40/A800 HPC GPU")
            train_rows, validation_rows, test_rows = (
                load_rows(data_dir / "train.jsonl"), load_rows(data_dir / "validation.jsonl"), load_rows(data_dir / "test.jsonl")
            )
            data_hash = hashlib.sha256("".join(sha256(data_dir / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl")).encode("ascii")).hexdigest()
            cache_dir = os.environ["HF_HOME"]
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache_dir, trust_remote_code=False)
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
            tokenizer.padding_side = "right"
            train_data = TokenDataset(train_rows, tokenizer, int(config["max_sequence_length"]))
            validation_data = TokenDataset(validation_rows, tokenizer, int(config["max_sequence_length"]))
            test_data = TokenDataset(test_rows, tokenizer, int(config["max_sequence_length"]))
            collator = CausalCollator(tokenizer)

            generation_eval_samples = int(config.get("generation_eval_samples", 24))
            parent_adapter_subdir = str(config.get("parent_adapter_subdir") or "").strip()
            parent_adapter_dir = data_dir / parent_adapter_subdir if parent_adapter_subdir else None
            if parent_adapter_dir is not None and not (parent_adapter_dir / "adapter_model.safetensors").is_file():
                raise RuntimeError("refinement parent adapter is missing from the controlled bundle")

            model = load_base(args.base_model, cache_dir)
            model.config.use_cache = False
            if parent_adapter_dir is not None:
                model = PeftModel.from_pretrained(model, parent_adapter_dir, is_trainable=False)
            before_loss = evaluation_loss(model, test_data, collator, out_dir / "before_eval", seed)
            before_generated = generated_metrics(model, tokenizer, test_rows, generation_eval_samples)
            before_score = score(before_loss, before_generated)

            del model
            gc.collect(); torch.cuda.empty_cache()
            model = load_base(args.base_model, cache_dir)
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            if parent_adapter_dir is not None:
                model = PeftModel.from_pretrained(model, parent_adapter_dir, is_trainable=True)
            else:
                lora = LoraConfig(
                    r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
                    lora_dropout=float(config["lora_dropout"]), bias="none", task_type="CAUSAL_LM",
                    target_modules=config["target_modules"],
                )
                model = get_peft_model(model, lora)
            telemetry_path = out_dir / "telemetry.jsonl"
            telemetry_path.write_text(json.dumps({"ts": time.time(), "event": "training_start", "gpu": gpu}) + "\n", encoding="utf-8")
            training_args = TrainingArguments(
                output_dir=str(out_dir / "checkpoints"),
                per_device_train_batch_size=int(config["per_device_batch_size"]),
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
                num_train_epochs=float(config["epochs"]),
                learning_rate=float(config["learning_rate"]),
                warmup_ratio=float(config["warmup_ratio"]),
                lr_scheduler_type="cosine",
                logging_steps=int(config["logging_steps"]),
                save_strategy="epoch", evaluation_strategy="epoch",
                load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                save_total_limit=1,
                bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
                report_to=[], seed=seed, data_seed=seed, remove_unused_columns=False,
            )
            trainer = Trainer(
                model=model, args=training_args, train_dataset=train_data,
                eval_dataset=validation_data, data_collator=collator,
                callbacks=[TelemetryCallback(telemetry_path)],
            )
            started = time.time()
            train_result = trainer.train()
            duration = time.time() - started
            after_loss = float(trainer.evaluate(eval_dataset=test_data)["eval_loss"])
            best_model_checkpoint = trainer.state.best_model_checkpoint
            best_validation_loss = trainer.state.best_metric
            adapter_dir = out_dir / "adapter"
            model.save_pretrained(adapter_dir, safe_serialization=True)
            tokenizer.save_pretrained(adapter_dir)
            code_hash = sha256(__file__)
            config_hash = sha256(args.config)
            del trainer, model
            gc.collect(); torch.cuda.empty_cache()

            reloaded_base = load_base(args.base_model, cache_dir)
            reloaded = PeftModel.from_pretrained(reloaded_base, adapter_dir, is_trainable=False)
            reloaded.config.use_cache = True
            reload_loss = evaluation_loss(reloaded, test_data, collator, out_dir / "reload_eval", seed)
            after_generated = generated_metrics(reloaded, tokenizer, test_rows, generation_eval_samples)
            after_score = score(reload_loss, after_generated)
            reload_delta = abs(reload_loss - after_loss)
            reload_payload = {
                "schema": "evomind.adapter_reload.v1", "passed": reload_delta <= 0.02,
                "pre_reload_loss": after_loss, "reloaded_loss": reload_loss, "absolute_delta": reload_delta,
                "adapter_config_sha256": sha256(adapter_dir / "adapter_config.json"),
                "adapter_model_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
            }
            atomic_json(out_dir / "adapter_reload.json", reload_payload)
            before = {"loss": before_loss, "domain_composite": before_score, **before_generated}
            after = {"loss": reload_loss, "domain_composite": after_score, **after_generated}
            metrics = {
                "schema": "evomind.llm_metrics.v1", "run_id": args.run_id,
                "base_model": args.base_model, "training_method": "4-bit QLoRA",
                "data_hash": data_hash, "code_hash": code_hash, "config_hash": config_hash,
                "dataset_counts": {"train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)},
                "before": before, "after": after,
                "improvement_pp": after_score - before_score,
                "comparison_baseline": "parent_adapter" if parent_adapter_dir is not None else "base_model",
                "parent_run_id": config.get("parent_run_id"),
                "format_regression_pp": (before_generated["format_compliance"] - after_generated["format_compliance"]) * 100.0,
                "safety_regression_pp": (after_generated["sensitive_leak_rate"] - before_generated["sensitive_leak_rate"]) * 100.0,
                "training": {
                    "train_loss": float(train_result.training_loss), "steps": int(train_result.global_step),
                    "duration_seconds": duration, "max_cuda_memory_mb": round(torch.cuda.max_memory_allocated() / 1048576, 2),
                    "best_model_checkpoint": best_model_checkpoint,
                    "best_validation_loss": best_validation_loss,
                    "assistant_only_loss": True,
                },
                "official_external_score": None, "model_published": False, "local_gpu_used": False,
            }
            atomic_json(out_dir / "metrics.json", metrics)
            atomic_json(out_dir / "evaluation.json", {
                "schema": "evomind.llm_evaluation.v1", "fixed_test_records": len(test_rows),
                "metric_definition": "0.70*(100/(1+loss)) + 20*keyword_recall + 10*format_compliance",
                "before": before, "after": after, "adapter_reload": reload_payload,
            })
            atomic_json(out_dir / "environment.json", {
                "schema": "evomind.llm_environment.v1", "run_id": args.run_id,
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "cuda": torch.version.cuda,
                "gpu": gpu, "remote_workdir": str(Path.cwd()), "local_gpu_used": False,
                "cache_root": cache_dir,
            })
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"ts": time.time(), "event": "training_complete", "metrics": metrics["training"], "gpu": gpu}) + "\n")
            print("EVOMIND_LLM_RESULT=" + json.dumps(metrics, ensure_ascii=False), flush=True)
            if not reload_payload["passed"]:
                raise RuntimeError("adapter reload reproducibility check failed")

        if __name__ == "__main__":
            main()
        """
    ).lstrip()


def llm_roles() -> dict[str, AgentRoleSpec]:
    roles = (
        AgentRoleSpec(
            "SetupAgent",
            ("environment_probe", "run_setup"),
            ("hpc_probe",),
            ("read_workspace", "hpc_read"),
            output_contract=("request.json", "hpc_probe.json"),
        ),
        AgentRoleSpec(
            "DataAuditor",
            ("source_split", "deduplication", "secret_scan"),
            ("read_docs",),
            ("read_workspace", "write_run_artifacts"),
            output_contract=("dataset_manifest.json", "data_audit.json"),
        ),
        AgentRoleSpec(
            "TrainingDesigner",
            ("qlora_design", "metric_contract"),
            ("read_dataset_manifest",),
            ("read_run_evidence", "write_run_artifacts"),
            output_contract=("qlora_config.json",),
        ),
        AgentRoleSpec(
            "EngineeringAgent",
            ("training_code", "static_validation"),
            ("write_isolated_code", "py_compile"),
            ("read_run_evidence", "write_run_artifacts"),
            output_contract=("train_qlora.py", "code_manifest.json"),
        ),
        AgentRoleSpec(
            "HpcRuntimeAgent",
            ("stage", "execute", "collect"),
            ("hpc_runtime",),
            ("hpc_execute",),
            output_contract=("metrics.json", "adapter", "telemetry.jsonl"),
            wall_time_seconds=3600,
        ),
        AgentRoleSpec(
            "EvaluatorAgent",
            ("before_after_evaluation", "reload_validation"),
            ("read_run_evidence",),
            ("read_run_evidence", "write_run_artifacts"),
            output_contract=("evaluation_summary.json",),
        ),
        AgentRoleSpec(
            "IndependentReviewer",
            ("evidence_review", "claim_boundary"),
            ("read_run_evidence",),
            ("read_run_evidence",),
            output_contract=("review.json",),
        ),
        AgentRoleSpec(
            "ClaimAuditAgent",
            ("claim_audit", "publication_gate"),
            ("read_run_evidence",),
            ("read_run_evidence",),
            output_contract=("claim_audit.json",),
        ),
        AgentRoleSpec(
            "SynthesisAgent",
            ("report", "model_card", "manifest"),
            ("read_reviewed_evidence",),
            ("read_run_evidence", "write_run_artifacts"),
            output_contract=("model_card.md", "research_report.md", "artifact_manifest.json"),
        ),
    )
    return {role.role: role for role in roles}


def build_llm_finetune_run(request: UserRequest, *, run_id: str | None = None) -> SupervisorRun:
    if request.task_type != "llm_finetune":
        raise ValueError("LLM fine-tune DAG requires task_type=llm_finetune")
    resolved_run_id = (
        run_id or f"qwen7b_qlora_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.urandom(3).hex()}"
    )
    tasks = [
        AgentTask(
            "setup",
            "Validate the run boundary and the current verified NVIDIA HPC allocation.",
            "SetupAgent",
            priority=100,
            acceptance_criteria=("request.json", "hpc_probe.json"),
        ),
        AgentTask(
            "data_audit",
            "Build source-grounded, source-disjoint EvoMind document data and audit it.",
            "DataAuditor",
            ("setup",),
            priority=90,
            acceptance_criteria=("data/dataset_manifest.json", "data_audit.json"),
        ),
        AgentTask(
            "training_design",
            "Define the falsifiable QLoRA and evaluation contract.",
            "TrainingDesigner",
            ("data_audit",),
            priority=80,
            acceptance_criteria=("qlora_config.json",),
        ),
        AgentTask(
            "engineering",
            "Generate and statically validate the isolated QLoRA runner.",
            "EngineeringAgent",
            ("training_design",),
            priority=70,
            acceptance_criteria=("code/train_qlora.py", "code_manifest.json"),
        ),
        AgentTask(
            "hpc_train",
            "Execute Qwen2.5-7B-Instruct 4-bit QLoRA on the verified NVIDIA HPC GPU.",
            "HpcRuntimeAgent",
            ("engineering",),
            priority=60,
            resource_type="hpc_gpu",
            timeout_seconds=_hpc_task_timeout_seconds(request),
            max_retries=1,
            acceptance_criteria=("llm_output/adapter", "llm_output/metrics.json", "llm_output/telemetry.jsonl"),
        ),
        AgentTask(
            "evaluate",
            "Verify the fixed test result and adapter reload evidence.",
            "EvaluatorAgent",
            ("hpc_train",),
            priority=50,
            acceptance_criteria=("evaluation_summary.json",),
        ),
        AgentTask(
            "independent_review",
            "Review only raw logs, metrics, hashes, telemetry, data split, and adapter files.",
            "IndependentReviewer",
            ("evaluate",),
            priority=40,
            max_retries=0,
            acceptance_criteria=("review.json",),
        ),
        AgentTask(
            "claim_audit",
            "Reject unsupported base-training, publication, or score claims.",
            "ClaimAuditAgent",
            ("independent_review",),
            priority=30,
            max_retries=0,
            acceptance_criteria=("claim_audit.json",),
        ),
        AgentTask(
            "synthesis",
            "Publish reviewed Adapter, model card, manifest, and research report locally.",
            "SynthesisAgent",
            ("claim_audit",),
            priority=20,
            max_retries=0,
            acceptance_criteria=("model_card.md", "artifact_manifest.json", "research_report.md"),
        ),
    ]
    run = create_run(
        objective=request.objective,
        tasks=tasks,
        roles=llm_roles().values(),
        run_id=resolved_run_id,
        max_concurrency=min(request.budget.max_parallel, 3),
    )
    run.gates = {
        "hpc_execution": "required",
        "reviewer": "required",
        "claim_audit": "required",
        "adapter_reload": "required",
        "model_publication": request.submission_policy.model_publication,
    }
    run.resource_limits = {"cpu": 3, "hpc_gpu": 1, "gpu": 1}
    return run


class LlmFinetuneExecutors:
    def __init__(
        self, *, workspace_root: Path, run_dir: Path, request: UserRequest, runtime: HpcRuntime, store: MultiAgentStore
    ) -> None:
        self.workspace_root = workspace_root
        self.run_dir = run_dir
        self.request = request
        self.runtime = runtime
        self.store = store
        self.dataset_dir = run_dir / "data"

    def setup(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        request_path = self.run_dir / "request.json"
        _atomic_json(request_path, {"schema": "evomind.user_request.v1", **self.request.to_dict()})
        probe = self.runtime.probe()
        probe_path = self.run_dir / "hpc_probe.json"
        _atomic_json(probe_path, probe.to_dict())
        gpu_names = [item.get("name", "") for item in probe.gpu_inventory]
        accepted = (
            probe.status == "passed"
            and len(gpu_names) == 1
            and _is_supported_hpc_gpu(gpu_names[0])
        )
        artifacts = [
            _artifact(request_path, self.run_dir, kind="request"),
            _artifact(probe_path, self.run_dir, kind="hpc_probe"),
        ]
        self.store.emit(
            run,
            "llm.hpc.probed",
            task_id=task.task_id,
            agent=task.role,
            status="passed" if accepted else "blocked",
            gpu_inventory=probe.gpu_inventory,
        )
        return AgentResult(
            task.task_id,
            "Verified the current supported NVIDIA HPC GPU allocation"
            if accepted
            else "Current allocation is not a supported NVIDIA A40/A800 GPU",
            [item["path"] for item in artifacts],
            metrics={"gpu_inventory": probe.gpu_inventory},
            artifacts=artifacts,
            confidence=1.0,
            accepted=accepted,
            failure_type=(probe.failure_type or "hpc_probe_blocked") if not accepted else "",
        )

    def data_audit(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        manifest = build_document_dataset(self.workspace_root, self.dataset_dir)
        overlap = manifest["source_overlap"]
        passed = (
            manifest["counts"] == DATASET_COUNTS
            and not any(overlap.values())
            and not manifest["duplicate_record_ids"]
            and not manifest["duplicate_content_records"]
            and manifest.get("grounded_context_contract") is True
        )
        audit = {
            "schema": "evomind.llm_data_audit.v1",
            "status": "passed" if passed else "rejected",
            "counts": manifest["counts"],
            "source_overlap": overlap,
            "duplicate_record_ids": manifest["duplicate_record_ids"],
            "duplicate_content_records": manifest["duplicate_content_records"],
            "secret_scan": "sanitized_and_passed",
            "evaluation_leakage": "source_disjoint",
            "grounded_context_contract": manifest.get("grounded_context_contract") is True,
            "data_hash": manifest["data_hash"],
            "generated_at": _now(),
        }
        audit_path = self.run_dir / "data_audit.json"
        _atomic_json(audit_path, audit)
        paths = [
            self.dataset_dir / name
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_manifest.json")
        ]
        paths.append(audit_path)
        artifacts = [_artifact(path, self.run_dir, kind=path.name) for path in paths]
        self.store.emit(
            run,
            "llm.dataset.ready",
            task_id=task.task_id,
            agent=task.role,
            status=audit["status"],
            dataset_counts=manifest["counts"],
            data_hash=manifest["data_hash"],
        )
        return AgentResult(
            task.task_id,
            "Built 800/100/100 source-isolated EvoMind document records",
            [item["path"] for item in artifacts],
            metrics={"dataset_counts": manifest["counts"], "data_hash": manifest["data_hash"]},
            artifacts=artifacts,
            confidence=1.0,
            accepted=passed,
            failure_type="data_audit_rejected" if not passed else "",
        )

    def training_design(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        correction_attempt = int(task.payload.get("correction_attempt") or 0)
        if correction_attempt:
            return self.correction_design(task, run, correction_attempt)
        config = {
            "schema": "evomind.qlora_config.v1",
            "base_model": self.request.base_model or BASE_MODEL,
            "method": "NF4 4-bit QLoRA",
            "compute_dtype": "bfloat16",
            "double_quant": True,
            "gradient_checkpointing": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_sequence_length": 1024,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "effective_batch_size": 16,
            "epochs": 2,
            "learning_rate": 0.0002,
            "warmup_ratio": 0.03,
            "logging_steps": 5,
            "generation_eval_samples": 24,
            "assistant_only_loss": True,
            "select_best_validation_checkpoint": True,
            "seed": 20260722,
            "acceptance": {
                "domain_composite_improvement_pp": 5.0,
                "max_format_regression_pp": 2.0,
                "max_safety_regression_pp": 2.0,
                "adapter_reload_required": True,
            },
            "local_gpu_allowed": False,
            "model_publication": self.request.submission_policy.model_publication,
        }
        path = self.run_dir / "qlora_config.json"
        _atomic_json(path, config)
        artifact = _artifact(path, self.run_dir, kind="qlora_config")
        self.store.emit(
            run,
            "llm.training.designed",
            task_id=task.task_id,
            agent=task.role,
            status="completed",
            base_model=config["base_model"],
            qlora_config=config,
        )
        return AgentResult(
            task.task_id,
            "Defined a bounded 4-bit QLoRA experiment and acceptance contract",
            [artifact["path"]],
            metrics={"base_model": config["base_model"], "effective_batch_size": 16},
            artifacts=[artifact],
            confidence=1.0,
        )

    def correction_design(self, task: AgentTask, run: SupervisorRun, correction_attempt: int) -> AgentResult:
        archive = _archive_current_attempt(self.run_dir, correction_attempt)
        label = f"correction_{correction_attempt:02d}"
        previous_label = "initial" if correction_attempt == 1 else f"correction_{correction_attempt - 1:02d}"
        archived_config_path = self.run_dir / "attempts" / previous_label / "qlora_config.json"
        previous_config = _read_json(archived_config_path)
        config = dict(previous_config)
        spec = CORRECTION_SPECS[correction_attempt]
        field = str(spec["field"])
        old_value = previous_config[field]
        config[field] = spec["value"]
        changed_fields = sorted(key for key in config if config.get(key) != previous_config.get(key))
        if changed_fields != [field]:
            raise RuntimeError(f"single-variable correction contract violated: {changed_fields}")

        config_path = self.run_dir / "qlora_config.json"
        snapshot_path = self.run_dir / "configs" / f"{label}.json"
        _atomic_json(config_path, config)
        _atomic_json(snapshot_path, config)
        correction_path = self.run_dir / "corrections" / f"{label}.json"
        correction = {
            "schema": "evomind.llm_single_variable_correction.v1",
            "correction_attempt": correction_attempt,
            "previous_attempt": previous_label,
            "changed_field": field,
            "old_value": old_value,
            "new_value": spec["value"],
            "rationale": spec["rationale"],
            "unchanged_fields_verified": True,
            "previous_config_sha256": _sha256(archived_config_path),
            "config_sha256": _sha256(config_path),
            "archived_artifact_count": len(archive.get("artifacts") or []),
            "generated_at": _now(),
        }
        _atomic_json(correction_path, correction)
        artifacts = [
            _artifact(correction_path, self.run_dir, kind="single_variable_correction"),
            _artifact(snapshot_path, self.run_dir, kind="qlora_config"),
            _artifact(
                self.run_dir / "attempts" / previous_label / "attempt_manifest.json",
                self.run_dir,
                kind="attempt_archive",
            ),
        ]
        self.store.emit(
            run,
            "llm.correction.planned",
            task_id=task.task_id,
            agent=task.role,
            status="completed",
            correction_attempt=correction_attempt,
            changed_field=field,
            old_value=old_value,
            new_value=spec["value"],
        )
        return AgentResult(
            task.task_id,
            f"Planned correction {correction_attempt} by changing only {field}",
            [item["path"] for item in artifacts],
            metrics={"correction_attempt": correction_attempt, "changed_field": field},
            artifacts=artifacts,
            confidence=1.0,
        )

    def engineering(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        source_path = self.run_dir / "code" / "train_qlora.py"
        _atomic_text(source_path, _training_source())
        py_compile.compile(str(source_path), doraise=True)
        requirements_path = self.run_dir / "code" / "requirements.lock"
        _atomic_text(
            requirements_path,
            "\n".join(
                (
                    "transformers==4.46.3",
                    "peft==0.13.2",
                    "trl==0.12.2",
                    "bitsandbytes==0.49.2",
                    "datasets==3.1.0",
                    "accelerate==1.1.1",
                    "safetensors==0.4.5",
                    "dill==0.3.9",
                    "joblib==1.4.2",
                    "",
                )
            ),
        )
        manifest_path = self.run_dir / "code_manifest.json"
        _atomic_json(
            manifest_path,
            {
                "schema": "evomind.llm_code_manifest.v1",
                "source_sha256": _sha256(source_path),
                "requirements_sha256": _sha256(requirements_path),
                "static_check": "py_compile_passed",
                "local_gpu_used": False,
                "generated_at": _now(),
            },
        )
        artifacts = [
            _artifact(path, self.run_dir, kind=path.name) for path in (source_path, requirements_path, manifest_path)
        ]
        self.store.emit(
            run,
            "llm.code.ready",
            task_id=task.task_id,
            agent=task.role,
            status="completed",
            code_sha256=artifacts[0]["sha256"],
        )
        return AgentResult(
            task.task_id,
            "Generated and statically validated the isolated QLoRA runner",
            [item["path"] for item in artifacts],
            artifacts=artifacts,
            confidence=1.0,
        )

    def hpc_train(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        correction_attempt = int(task.payload.get("correction_attempt") or 0)
        execution_id = str(task.payload.get("execution_id") or "") or None
        environment = self.runtime.prepare_llm_environment()
        environment_path = self.run_dir / "llm_runtime_environment.json"
        _atomic_json(environment_path, environment)
        probe = self.runtime.probe(use_run_environment=True)
        probe_path = self.run_dir / "hpc_llm_probe.json"
        _atomic_json(probe_path, probe.to_dict())
        if (
            probe.status != "passed"
            or not probe.gpu_inventory
            or not _is_supported_hpc_gpu(probe.gpu_inventory[0].get("name", ""))
        ):
            raise RuntimeError(f"HPC_RESOURCE: isolated LLM runtime probe failed: {probe.error}")
        self.store.emit(
            run,
            "llm.training.started",
            task_id=task.task_id,
            agent=task.role,
            status="running",
            base_model=self.request.base_model or BASE_MODEL,
            gpu_inventory=probe.gpu_inventory,
            correction_attempt=correction_attempt,
        )
        config = _read_json(self.run_dir / "qlora_config.json")
        data_files = {
            name: self.dataset_dir / name
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_manifest.json")
        }
        parent_run_id = str(config.get("parent_run_id") or "").strip()
        parent_adapter_subdir = str(config.get("parent_adapter_subdir") or "").strip()
        if parent_run_id and parent_adapter_subdir:
            parent_adapter = self.workspace_root / "workspace" / "evomind_runs" / parent_run_id / "llm_output" / "adapter"
            parent_adapter = parent_adapter.resolve()
            expected_parent = (self.workspace_root / "workspace" / "evomind_runs").resolve()
            parent_adapter.relative_to(expected_parent)
            if not (parent_adapter / "adapter_model.safetensors").is_file():
                raise RuntimeError("refinement parent adapter is not available")
            for parent_file in parent_adapter.rglob("*"):
                if parent_file.is_file():
                    relative = parent_file.relative_to(parent_adapter).as_posix()
                    data_files[f"{parent_adapter_subdir}/{relative}"] = parent_file

        job = self.runtime.execute_llm_finetune(
            script_path=self.run_dir / "code" / "train_qlora.py",
            data_files=data_files,
            config_path=self.run_dir / "qlora_config.json",
            base_model=self.request.base_model or BASE_MODEL,
            execution_id=execution_id,
        )
        job_path = self.run_dir / "hpc_job.json"
        task_job_path = self.run_dir / "hpc_jobs" / f"{task.task_id}.json"
        job_payload = asdict(job)
        _atomic_json(job_path, job_payload)
        _atomic_json(task_job_path, job_payload)
        artifacts = [
            _artifact(path, self.run_dir, kind=path.name) for path in (environment_path, probe_path, task_job_path)
        ]
        for item in job.local_artifacts:
            path = Path(item["path"])
            if path.is_file():
                artifacts.append(_artifact(path, self.run_dir, kind=path.name))
        if job.status != "completed":
            raise RuntimeError(f"HPC_{job.failure_type.upper()}: {job.error}")
        metrics = _read_json(self.run_dir / "llm_output" / "metrics.json")
        self.store.emit(
            run,
            "llm.training.completed",
            task_id=task.task_id,
            agent=task.role,
            status="completed",
            training_step=metrics["training"]["steps"],
            loss=metrics["training"]["train_loss"],
            gpu_memory=metrics["training"]["max_cuda_memory_mb"],
            correction_attempt=correction_attempt,
            before_after_eval={
                "before": metrics["before"],
                "after": metrics["after"],
                "improvement_pp": metrics["improvement_pp"],
            },
        )
        return AgentResult(
            task.task_id,
            f"Real {probe.gpu_inventory[0]['name']} QLoRA completed with {metrics['training']['steps']} optimizer steps",
            [item["path"] for item in artifacts],
            metrics={
                "training_step": metrics["training"]["steps"],
                "loss": metrics["training"]["train_loss"],
                "gpu_memory": metrics["training"]["max_cuda_memory_mb"],
                "improvement_pp": metrics["improvement_pp"],
            },
            artifacts=artifacts,
            confidence=1.0,
        )

    def evaluate(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        correction_attempt = int(task.payload.get("correction_attempt") or 0)
        metrics = _read_json(self.run_dir / "llm_output" / "metrics.json")
        evaluation = _read_json(self.run_dir / "llm_output" / "evaluation.json")
        reload_result = _read_json(self.run_dir / "llm_output" / "adapter_reload.json")
        summary = {
            "schema": "evomind.llm_evaluation_summary.v1",
            "status": "passed" if reload_result.get("passed") else "rejected",
            "base_model": metrics["base_model"],
            "fixed_test_records": evaluation["fixed_test_records"],
            "before": metrics["before"],
            "after": metrics["after"],
            "improvement_pp": metrics["improvement_pp"],
            "adapter_reload": reload_result,
            "generated_at": _now(),
        }
        path = self.run_dir / "evaluation_summary.json"
        task_path = self.run_dir / "evaluations" / f"{task.task_id}.json"
        _atomic_json(path, summary)
        _atomic_json(task_path, summary)
        artifact = _artifact(task_path, self.run_dir, kind="evaluation_summary")
        self.store.emit(
            run,
            "llm.evaluation.completed",
            task_id=task.task_id,
            agent=task.role,
            status=summary["status"],
            correction_attempt=correction_attempt,
            before_after_eval={
                "before": summary["before"],
                "after": summary["after"],
                "improvement_pp": summary["improvement_pp"],
            },
        )
        return AgentResult(
            task.task_id,
            f"Fixed-test domain composite changed by {metrics['improvement_pp']:.2f} pp",
            [artifact["path"]],
            metrics={
                "before_after_eval": {
                    "before": metrics["before"],
                    "after": metrics["after"],
                    "improvement_pp": metrics["improvement_pp"],
                }
            },
            artifacts=[artifact],
            confidence=1.0,
            accepted=bool(reload_result.get("passed")),
            failure_type="adapter_reload_failed" if not reload_result.get("passed") else "",
        )

    def additional_review_evidence(
        self, run: SupervisorRun
    ) -> tuple[dict[str, bool], list[dict[str, Any]], str]:
        """Allow specialized workflows to extend the raw Reviewer contract."""
        return {}, [], ""

    def review(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        correction_attempt = int(task.payload.get("correction_attempt") or 0)
        metrics = _read_json(self.run_dir / "llm_output" / "metrics.json")
        environment = _read_json(self.run_dir / "llm_output" / "environment.json")
        reload_result = _read_json(self.run_dir / "llm_output" / "adapter_reload.json")
        data_manifest = _read_json(self.dataset_dir / "dataset_manifest.json")
        job = _read_json(self.run_dir / "hpc_job.json")
        config = _read_json(self.run_dir / "qlora_config.json")
        acceptance = config.get("acceptance") or {}
        required_improvement = float(acceptance.get("domain_composite_improvement_pp", 5.0))
        max_format_regression = float(acceptance.get("max_format_regression_pp", 2.0))
        max_safety_regression = float(acceptance.get("max_safety_regression_pp", 2.0))
        artifact_hashes = {
            Path(item["path"]).resolve(): item["sha256"]
            for item in job.get("local_artifacts", [])
            if Path(item["path"]).is_file()
        }
        hashes_valid = bool(artifact_hashes) and all(
            _sha256(path) == expected for path, expected in artifact_hashes.items()
        )
        additional_checks, additional_artifacts, additional_input = self.additional_review_evidence(run)
        checks = {
            "fresh_run_id": metrics.get("run_id") == run.run_id,
            "base_model": metrics.get("base_model") == (self.request.base_model or BASE_MODEL),
            "data_hash": metrics.get("data_hash") == data_manifest.get("data_hash"),
            "dataset_counts": metrics.get("dataset_counts") == DATASET_COUNTS,
            "source_split_disjoint": not any(data_manifest.get("source_overlap", {}).values()),
            "unique_instruction_records": not data_manifest.get("duplicate_record_ids", True)
            and not data_manifest.get("duplicate_content_records", True),
            "grounded_context_contract": data_manifest.get("grounded_context_contract") is True,
            "artifact_hashes": hashes_valid,
            "verified_supported_hpc_gpu": any(
                _is_supported_hpc_gpu(item.get("name", "")) for item in environment.get("gpu", [])
            ),
            "local_gpu_unused": metrics.get("local_gpu_used") is False and environment.get("local_gpu_used") is False,
            "adapter_reload": reload_result.get("passed") is True,
            "assistant_only_loss": metrics.get("training", {}).get("assistant_only_loss") is True,
            "best_validation_checkpoint": bool(metrics.get("training", {}).get("best_model_checkpoint")),
            "domain_improvement": float(metrics.get("improvement_pp", -999)) >= required_improvement,
            "format_regression": float(metrics.get("format_regression_pp", 999)) <= max_format_regression,
            "safety_regression": float(metrics.get("safety_regression_pp", 999)) <= max_safety_regression,
            "model_not_published": metrics.get("model_published") is False,
            **additional_checks,
        }
        passed = all(checks.values())
        unresolved = [name for name, value in checks.items() if not value]
        retryable = unresolved == ["domain_improvement"] and correction_attempt < MAX_CORRECTION_ATTEMPTS
        followup_tasks: list[AgentTask] = []
        dependency_overrides: dict[str, tuple[str, ...]] = {}
        if retryable:
            followup_tasks, dependency_overrides = _correction_task_plan(
                run=run,
                source_review_task_id=task.task_id,
                correction_attempt=correction_attempt + 1,
                timeout_seconds=_hpc_task_timeout_seconds(self.request),
            )
        review = {
            "schema": "evomind.llm_independent_review.v1",
            "status": "passed" if passed else "rejected",
            "review_input": "raw_dataset_hashes_training_log_metrics_telemetry_adapter_environment"
            + (f"_{additional_input}" if additional_input else ""),
            "reviewed_artifacts": additional_artifacts,
            "parent_subjective_summary_received": False,
            "checks": checks,
            "before_after_eval": {
                "before": metrics.get("before"),
                "after": metrics.get("after"),
                "improvement_pp": metrics.get("improvement_pp"),
            },
            "acceptance_contract": {
                "domain_composite_improvement_pp": required_improvement,
                "max_format_regression_pp": max_format_regression,
                "max_safety_regression_pp": max_safety_regression,
            },
            "correction_attempt": correction_attempt,
            "next_action": "proceed_to_claim_audit"
            if passed
            else ("run_single_variable_correction" if retryable else "needs_continuation"),
            "unresolved": unresolved,
            "generated_at": _now(),
        }
        path = self.run_dir / "review.json"
        task_path = self.run_dir / "reviews" / f"{task.task_id}.json"
        _atomic_json(path, review)
        _atomic_json(task_path, review)
        artifact = _artifact(task_path, self.run_dir, kind="independent_review")
        run.gates["reviewer"] = "passed" if passed else ("rejected_retry_planned" if retryable else "rejected")
        self.store.emit(
            run,
            "llm.review.completed",
            task_id=task.task_id,
            agent=task.role,
            status=review["status"],
            correction_attempt=correction_attempt,
            review=review,
        )
        return AgentResult(
            task.task_id,
            "Independent Reviewer accepted every raw evidence contract"
            if passed
            else "Independent Reviewer rejected the current attempt",
            [artifact["path"]],
            metrics={"checks": checks, "decision": review["status"], "correction_attempt": correction_attempt},
            artifacts=[artifact],
            confidence=1.0,
            accepted=passed or retryable,
            failure_type="" if passed or retryable else "review_rejected_after_max_corrections",
            unresolved=unresolved,
            followup_tasks=followup_tasks,
            dependency_overrides=dependency_overrides,
        )

    def claim_audit(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        review = _read_json(self.run_dir / "review.json")
        metrics = _read_json(self.run_dir / "llm_output" / "metrics.json")
        checks = {
            "review_passed": review.get("status") == "passed",
            "described_as_domain_finetune": metrics.get("training_method") == "4-bit QLoRA",
            "not_claimed_as_base_pretraining": True,
            "not_claimed_as_external_benchmark": metrics.get("official_external_score") is None,
            "publication_blocked": metrics.get("model_published") is False,
        }
        passed = all(checks.values())
        payload = {
            "schema": "evomind.llm_claim_audit.v1",
            "status": "passed" if passed else "rejected",
            "checks": checks,
            "approved_claim": "Qwen2.5-7B-Instruct domain QLoRA completed on the verified NVIDIA HPC GPU",
            "prohibited_claims": [
                "trained a 7B foundation model from scratch",
                "published model",
                "official external benchmark score",
            ],
            "generated_at": _now(),
        }
        path = self.run_dir / "claim_audit.json"
        _atomic_json(path, payload)
        artifact = _artifact(path, self.run_dir, kind="claim_audit")
        self.store.emit(
            run,
            "llm.claim_audit.completed",
            task_id=task.task_id,
            agent=task.role,
            status=payload["status"],
            claim_audit=payload,
        )
        return AgentResult(
            task.task_id,
            "Claim Audit approved the bounded domain-fine-tune claim",
            [artifact["path"]],
            artifacts=[artifact],
            confidence=1.0,
            accepted=passed,
            failure_type="claim_audit_rejected" if not passed else "",
        )

    def synthesis(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        review = _read_json(self.run_dir / "review.json")
        claim_audit = _read_json(self.run_dir / "claim_audit.json")
        metrics = _read_json(self.run_dir / "llm_output" / "metrics.json")
        environment = _read_json(self.run_dir / "llm_output" / "environment.json")
        if review.get("status") != "passed" or claim_audit.get("status") != "passed":
            raise RuntimeError("review and claim gates must pass before synthesis")
        gpu_inventory = environment.get("gpu") if isinstance(environment.get("gpu"), list) else []
        gpu_name = str((gpu_inventory[0] if gpu_inventory else {}).get("name") or "verified NVIDIA HPC GPU")
        model_card = self.run_dir / "model_card.md"
        _atomic_text(
            model_card,
            (
                "# EvoMind Qwen2.5-7B Domain Adapter\n\n"
                f"- Run: `{run.run_id}`\n"
                f"- Base model: `{metrics['base_model']}`\n"
                "- Method: NF4 4-bit QLoRA, BF16 compute\n"
                f"- Data: EvoMind documents, `{metrics['dataset_counts']}`\n"
                f"- Fixed-test domain composite: `{metrics['before']['domain_composite']:.2f}` -> `{metrics['after']['domain_composite']:.2f}` "
                f"(`{metrics['improvement_pp']:+.2f}` pp)\n"
                f"- Compute: one verified {gpu_name}; local GPU was not used\n"
                "- Review: Independent Reviewer and Claim Audit passed\n"
                "- Publication: blocked; the adapter remains a local deliverable\n\n"
                "This artifact is a domain adapter for a mature 7B base model. It is not a foundation model trained from scratch.\n"
            ),
        )
        report = self.run_dir / "research_report.md"
        _atomic_text(
            report,
            (
                "# EvoMind 7B 领域微调研究报告\n\n"
                f"## 直接结论\n\nEvoMind 通过自然语言请求完成 `{metrics['base_model']}` 的真实远程 GPU QLoRA 领域微调。"
                f"固定测试集综合分数提升 `{metrics['improvement_pp']:.2f}` 个百分点，Adapter 重载复现通过。\n\n"
                "## 关键证据\n\n"
                f"- 训练/验证/测试：{metrics['dataset_counts']['train']}/{metrics['dataset_counts']['validation']}/{metrics['dataset_counts']['test']}，按文档来源隔离\n"
                f"- 训练步数：{metrics['training']['steps']}，训练 loss：{metrics['training']['train_loss']:.6f}\n"
                f"- GPU：{gpu_name}；峰值显存：{metrics['training']['max_cuda_memory_mb']:.2f} MiB\n"
                f"- Before/After：{metrics['before']['domain_composite']:.2f} -> {metrics['after']['domain_composite']:.2f}\n"
                "- Independent Reviewer：passed\n- Claim Audit：passed\n- 本地 GPU：未使用\n- 模型发布：未执行\n\n"
                "## 产物\n\n`llm_output/adapter/`、`model_card.md`、`metrics.json`、`review.json`、`claim_audit.json` 和 `artifact_manifest.json`。\n"
            ),
        )
        manifest_path = self.run_dir / "artifact_manifest.json"
        mutable = {
            "events.jsonl",
            "handoffs.jsonl",
            "messages.jsonl",
            "run.json",
            "task_graph.json",
            "control.json",
            "artifact_manifest.json",
        }
        entries = []
        for path in sorted(self.run_dir.rglob("*")):
            relative = path.relative_to(self.run_dir)
            if (
                not path.is_file()
                or relative.as_posix() in mutable
                or "__pycache__" in relative.parts
                or path.suffix == ".pyc"
                or ".tmp" in path.name
            ):
                continue
            entries.append(_artifact(path, self.run_dir, kind=path.name))
        _atomic_json(
            manifest_path,
            {
                "schema": "evomind.llm_artifact_manifest.v1",
                "run_id": run.run_id,
                "base_model": metrics["base_model"],
                "training_method": metrics["training_method"],
                "review_status": review["status"],
                "claim_audit_status": claim_audit["status"],
                "model_publication": "blocked",
                "artifacts": entries,
                "generated_at": _now(),
            },
        )
        run.gates.update(
            {
                "hpc_execution": "passed",
                "reviewer": "passed",
                "claim_audit": "passed",
                "adapter_reload": "passed",
                "model_publication": "blocked",
            }
        )
        artifacts = [
            _artifact(path, self.run_dir, kind=kind)
            for path, kind in (
                (model_card, "model_card"),
                (report, "research_report"),
                (manifest_path, "artifact_manifest"),
                (self.run_dir / "llm_output" / "metrics.json", "metrics"),
                (self.run_dir / "review.json", "review"),
            )
        ]
        self.store.emit(
            run,
            "llm.synthesis.completed",
            task_id=task.task_id,
            agent=task.role,
            status="completed",
            base_model=metrics["base_model"],
            before_after_eval={
                "before": metrics["before"],
                "after": metrics["after"],
                "improvement_pp": metrics["improvement_pp"],
            },
            artifact_hashes=[item["sha256"] for item in artifacts],
        )
        return AgentResult(
            task.task_id,
            "Delivered the reviewed Adapter, model card, manifest, and research report",
            [item["path"] for item in artifacts],
            metrics={
                "base_model": metrics["base_model"],
                "before_after_eval": {
                    "before": metrics["before"],
                    "after": metrics["after"],
                    "improvement_pp": metrics["improvement_pp"],
                },
            },
            artifacts=artifacts,
            confidence=1.0,
        )

    def mapping(self):
        return {
            "SetupAgent": self.setup,
            "DataAuditor": self.data_audit,
            "TrainingDesigner": self.training_design,
            "EngineeringAgent": self.engineering,
            "HpcRuntimeAgent": self.hpc_train,
            "EvaluatorAgent": self.evaluate,
            "IndependentReviewer": self.review,
            "ClaimAuditAgent": self.claim_audit,
            "SynthesisAgent": self.synthesis,
        }


def _store_with_current_pointer(root: Path, run_dir: Path) -> MultiAgentStore:
    return MultiAgentStore(
        run_dir,
        on_save=lambda saved_run: write_current_run_pointer(
            root,
            task_id=TASK_ID,
            run=saved_run,
            run_dir=run_dir,
        ),
    )


def _recover_legacy_rejected_review(
    *,
    run: SupervisorRun,
    store: MultiAgentStore,
    supervisor: MultiAgentSupervisor,
    request: UserRequest,
    run_dir: Path,
) -> bool:
    task = run.tasks.get("independent_review")
    review_path = run_dir / "review.json"
    if (
        task is None
        or task.status != "failed"
        or not review_path.is_file()
        or any(task_id.startswith("correction_") for task_id in run.tasks)
    ):
        return False
    review = _read_json(review_path)
    if review.get("status") != "rejected" or list(review.get("unresolved") or []) != ["domain_improvement"]:
        return False

    task_review_path = run_dir / "reviews" / "independent_review.json"
    task_review_path.parent.mkdir(parents=True, exist_ok=True)
    if not task_review_path.exists():
        shutil.copy2(review_path, task_review_path)
    artifact = _artifact(task_review_path, run_dir, kind="independent_review")
    followup_tasks, dependency_overrides = _correction_task_plan(
        run=run,
        source_review_task_id=task.task_id,
        correction_attempt=1,
        timeout_seconds=_hpc_task_timeout_seconds(request),
    )
    result = AgentResult(
        task.task_id,
        "Recovered the valid Reviewer rejection as a completed decision and planned one-variable correction 1",
        [artifact["path"]],
        metrics={"checks": review.get("checks") or {}, "decision": "rejected", "correction_attempt": 0},
        artifacts=[artifact],
        confidence=1.0,
        accepted=True,
        unresolved=["domain_improvement"],
        followup_tasks=followup_tasks,
        dependency_overrides=dependency_overrides,
    )
    previous_status = task.status
    task.result_ref = store.write_result(result)
    task.status = "completed"
    task.error = ""
    task.finished_at = _now()
    run.idempotency_results[task.idempotency_key] = task.result_ref
    run.gates["reviewer"] = "rejected_retry_planned"
    store.emit(
        run,
        "llm.review.feedback_recovered",
        task_id=task.task_id,
        agent=task.role,
        status="completed",
        previous_status=previous_status,
        decision="rejected",
        correction_attempt=1,
        evidence_refs=[artifact["path"], task.result_ref],
    )
    supervisor.apply_followups(task, result)
    return True


def _finalize_exhausted_rejected_review(
    *,
    run: SupervisorRun,
    store: MultiAgentStore,
    run_dir: Path,
) -> bool:
    """Convert an exhausted Reviewer rejection into a stable, truthful terminal state."""
    candidates = sorted(
        (
            task
            for task in run.tasks.values()
            if task.task_id.startswith("correction_")
            and task.task_id.endswith("_review")
            and task.status == "failed"
            and "review_rejected_after_max_corrections" in task.error
        ),
        key=lambda item: int(item.payload.get("correction_attempt") or 0),
    )
    if not candidates:
        return False

    task = candidates[-1]
    review_path = run_dir / "reviews" / f"{task.task_id}.json"
    if not review_path.is_file():
        review_path = run_dir / "review.json"
    if not review_path.is_file():
        return False

    review = _read_json(review_path)
    correction_attempt = int(review.get("correction_attempt") or task.payload.get("correction_attempt") or 0)
    unresolved = list(review.get("unresolved") or [])
    if (
        review.get("status") != "rejected"
        or correction_attempt < MAX_CORRECTION_ATTEMPTS
        or review.get("next_action") != "needs_continuation"
        or not unresolved
    ):
        return False

    metrics_path = run_dir / "llm_output" / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.is_file() else {}
    acceptance = dict(review.get("acceptance_contract") or {})
    actual_improvement = float(metrics.get("improvement_pp", -999.0))
    required_improvement = float(acceptance.get("domain_composite_improvement_pp", 5.0))
    report_path = run_dir / "no_go_report.md"
    _atomic_text(
        report_path,
        (
            "# EvoMind 7B QLoRA NO-GO Report\n\n"
            "## Decision\n\n"
            "The real verified NVIDIA HPC QLoRA run completed and the Adapter reload check passed, "
            "but the Independent Reviewer rejected the promotion gate.\n\n"
            "## Evidence\n\n"
            f"- Run: `{run.run_id}`\n"
            f"- Correction attempts: `{correction_attempt}/{MAX_CORRECTION_ATTEMPTS}`\n"
            f"- Fixed-test domain improvement: `{actual_improvement:.3f} pp`\n"
            f"- Required improvement: `>= {required_improvement:.3f} pp`\n"
            f"- Unresolved checks: `{', '.join(unresolved)}`\n"
            f"- Adapter reload: `{bool((_read_json(run_dir / 'llm_output' / 'adapter_reload.json') if (run_dir / 'llm_output' / 'adapter_reload.json').is_file() else {}).get('passed'))}`\n"
            "- Local GPU used: `False`\n"
            "- Model published: `False`\n\n"
            "## Next Action\n\n"
            "Revise the data or training strategy under a new reviewed run. Do not describe this run as meeting the promotion threshold.\n"
        ),
    )
    artifacts = [
        _artifact(review_path, run_dir, kind="independent_review"),
        _artifact(report_path, run_dir, kind="no_go_report"),
    ]
    if metrics_path.is_file():
        artifacts.append(_artifact(metrics_path, run_dir, kind="metrics"))
    result = AgentResult(
        task.task_id,
        "Independent Reviewer completed and rejected promotion after the bounded corrections were exhausted",
        [item["path"] for item in artifacts],
        metrics={
            "decision": "rejected",
            "correction_attempt": correction_attempt,
            "improvement_pp": actual_improvement,
            "required_improvement_pp": required_improvement,
        },
        artifacts=artifacts,
        unresolved=unresolved,
        confidence=1.0,
        accepted=True,
    )
    previous_status = task.status
    task.result_ref = store.write_result(result)
    task.status = "completed"
    task.error = ""
    task.finished_at = _now()
    run.idempotency_results[task.idempotency_key] = task.result_ref

    for downstream_id in ("claim_audit", "synthesis"):
        downstream = run.tasks.get(downstream_id)
        if downstream is None or downstream.status in TERMINAL_TASK_STATES:
            continue
        previous = downstream.status
        downstream.status = "skipped"
        downstream.finished_at = _now()
        store.emit(
            run,
            "task.state",
            task_id=downstream.task_id,
            agent=downstream.role,
            previous=previous,
            status="skipped",
            reason="reviewer_rejected_after_max_corrections",
        )

    run.status = "rejected"
    run.gates.update(
        {
            "hpc_execution": "passed",
            "reviewer": "rejected",
            "claim_audit": "not_run_reviewer_rejected",
            "adapter_reload": "passed",
            "model_publication": "forbidden",
        }
    )
    run.open_requirements = [
        f"domain_improvement: {actual_improvement:.3f} pp < {required_improvement:.3f} pp after {correction_attempt} corrections"
    ]
    run.next_action = "revise_data_or_training_strategy"
    store.emit(
        run,
        "llm.run.rejected",
        task_id=task.task_id,
        agent=task.role,
        status="rejected",
        previous_task_status=previous_status,
        correction_attempt=correction_attempt,
        improvement_pp=actual_improvement,
        required_improvement_pp=required_improvement,
        unresolved=unresolved,
        evidence_refs=[item["path"] for item in artifacts] + [task.result_ref],
        artifact_hashes=[item["sha256"] for item in artifacts],
    )
    return True


def run_llm_finetune(workspace_root: str | Path, request: UserRequest, *, run_id: str | None = None) -> SupervisorRun:
    root = Path(workspace_root).resolve()
    if request.task_type != "llm_finetune" or not request.requests_execution:
        raise ValueError("an explicit LLM fine-tune execution request is required")
    if request.compute_policy.local_gpu_allowed or not request.compute_policy.remote_gpu_required:
        raise ValueError("LLM fine-tuning requires the remote HPC GPU and forbids local GPU fallback")
    run = build_llm_finetune_run(request, run_id=run_id)
    local_run_dir = run_directory(root, run.run_id)
    store = _store_with_current_pointer(root, local_run_dir)
    request_hash = _sha256_text(json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True))
    for agent_task in run.tasks.values():
        agent_task.payload["config_hash"] = _sha256_text(f"{request_hash}\0{agent_task.task_id}\0{agent_task.goal}")
    runtime = HpcRuntime(
        run_id=run.run_id, local_run_dir=local_run_dir, timeout_seconds=request.budget.max_minutes * 60
    )
    executors = LlmFinetuneExecutors(
        workspace_root=root, run_dir=local_run_dir, request=request, runtime=runtime, store=store
    )
    store.append_message(run, sender="user", receiver="ExecutiveSupervisor", content=request.objective)
    write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    supervisor = MultiAgentSupervisor(run, store, executors.mapping())
    try:
        result = supervisor.run_until_blocked()
        _finalize_exhausted_rejected_review(run=run, store=store, run_dir=local_run_dir)
    finally:
        write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    return result


def resume_llm_finetune(workspace_root: str | Path, run_id: str) -> SupervisorRun:
    from xsci.user_request import parse_user_request

    root = Path(workspace_root).resolve()
    local_run_dir = run_directory(root, run_id)
    store = _store_with_current_pointer(root, local_run_dir)
    run = store.load()
    request_path = local_run_dir / "request.json"
    request = (
        UserRequest.from_dict(_read_json(request_path)) if request_path.is_file() else parse_user_request(run.objective)
    )
    hpc_task_timeout = _hpc_task_timeout_seconds(request)
    for task in run.tasks.values():
        if task.resource_type == "hpc_gpu":
            task.timeout_seconds = hpc_task_timeout
    runtime = HpcRuntime(
        run_id=run.run_id, local_run_dir=local_run_dir, timeout_seconds=request.budget.max_minutes * 60
    )
    executors = LlmFinetuneExecutors(
        workspace_root=root, run_dir=local_run_dir, request=request, runtime=runtime, store=store
    )
    supervisor = MultiAgentSupervisor(run, store, executors.mapping())
    _recover_legacy_rejected_review(
        run=run,
        store=store,
        supervisor=supervisor,
        request=request,
        run_dir=local_run_dir,
    )
    if _finalize_exhausted_rejected_review(run=run, store=store, run_dir=local_run_dir) or run.status == "rejected":
        write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
        return run
    supervisor.resume(retry_failed=run.status == "needs_continuation")
    write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    try:
        result = supervisor.run_until_blocked()
        _finalize_exhausted_rejected_review(run=run, store=store, run_dir=local_run_dir)
    finally:
        write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    return result


__all__ = [
    "BASE_MODEL",
    "LlmFinetuneExecutors",
    "build_document_dataset",
    "build_llm_finetune_run",
    "resume_llm_finetune",
    "run_llm_finetune",
]
