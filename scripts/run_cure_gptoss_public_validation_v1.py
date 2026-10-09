"""Run the public CURE-Bench validation set with the prepared GPT-OSS-20B.

This adapter is inference-only. It never reads either private phase-2 file,
never submits, and stores only final answers plus digests of generated text.
The persistent checkpoint makes a timed managed attempt resumable without
repeating already completed questions.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
DATA_ROOT = ROOT / "competition_data/cure_bench"
MODEL_ROOT = DATA_ROOT / "managed_runtime/gpt-oss-20b-r1"
MODEL_DIR = MODEL_ROOT / "model"
VALIDATION_PATH = DATA_ROOT / "data/curebench_valset_pharse1.jsonl"
CHECKPOINT_ROOT = MODEL_ROOT / "validation-runs/cure-public-validation-v1"
LOCK_PATH = CHECKPOINT_ROOT / ".run.lock"
REPO = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
DATA_MANIFEST_SHA256 = "b2af0ade5f012bae9722192c1c7575815f788228b8f8de509be6506438b90197"
VALIDATION_SHA256 = "6b45f291fb776c41492273a8fec694381fc63078bb340c59559a8be29b738ca2"
INFERENCE_BATCH_SIZE = 4
MAX_NEW_TOKENS = 512
MODEL_FILES = {
    "model-00000-of-00002.safetensors": (4792272488, "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427"),
    "model-00001-of-00002.safetensors": (4798702184, "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea"),
    "model-00002-of-00002.safetensors": (4170342232, "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303"),
}
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    resolved.relative_to(ROOT.resolve(strict=True))
    if path.is_symlink():
        raise ValueError("symlink_rejected")
    return resolved


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_model_assets() -> dict[str, Any]:
    manifest_path = bounded_path(MODEL_ROOT / "model-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repo") != REPO or manifest.get("revision") != REVISION or manifest.get("status") != "completed":
        raise ValueError("model_manifest_identity_invalid")
    declared = {str(item["name"]): item for item in manifest.get("files", [])}
    required = set(MODEL_FILES) | set(MODEL_METADATA_FILES)
    if not required.issubset(declared) or len(declared) != 13:
        raise ValueError("model_manifest_file_set_invalid")
    checked = []
    for name in sorted(required):
        path = bounded_path(MODEL_DIR / name)
        row = declared[name]
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError("model_asset_size_mismatch_" + name)
        digest = sha256_file(path)
        expected = MODEL_FILES.get(name, digest)[1]
        if name in MODEL_FILES and digest != expected:
            raise ValueError("model_asset_sha256_mismatch_" + name)
        if row.get("sha256") != digest:
            raise ValueError("model_manifest_sha256_mismatch_" + name)
        checked.append({"name": name, "bytes": path.stat().st_size, "sha256": digest})
    return {"repo": REPO, "revision": REVISION, "manifest_sha256": sha256_file(manifest_path), "files": checked}


def load_validation() -> list[dict[str, Any]]:
    path = bounded_path(VALIDATION_PATH)
    if sha256_file(path) != VALIDATION_SHA256:
        raise ValueError("validation_sha256_mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 459 or len({str(row.get("id")) for row in rows}) != 459:
        raise ValueError("validation_row_count_or_id_mismatch")
    expected_types = {"multi_choice", "open_ended_multi_choice", "open_ended"}
    if not all(str(row.get("question_type")) in expected_types for row in rows):
        raise ValueError("validation_question_type_invalid")
    if any("phase2" in json.dumps(row, ensure_ascii=True).lower() for row in rows):
        raise ValueError("phase2_reference_in_validation_rejected")
    return rows


def normalize_choice(text: str) -> str:
    value = str(text or "").strip().upper()
    matches = re.findall(r"(?:FINAL\s+ANSWER|ANSWER|CHOICE|OPTION)\s*(?:IS|:|=)?\s*\(?([A-E])\)?\b", value)
    if matches:
        return matches[-1]
    matches = re.findall(r"\b([A-E])\s*[\).:]", value)
    if matches:
        return matches[-1]
    if len(value) <= 3 and value[:1] in "ABCDE":
        return value[:1]
    return ""


def final_text(text: str) -> str:
    value = str(text or "").strip()
    # Keep a bounded answer excerpt, never the full hidden reasoning trace.
    for marker in ("<|channel|>final", "final:", "Final answer:", "FINAL ANSWER:"):
        if marker.lower() in value.lower():
            value = value[value.lower().rfind(marker.lower()) + len(marker):].strip()
    value = re.sub(r"<\|[^>]+\|>", " ", value)
    return re.sub(r"\s+", " ", value).strip()[:4000]


def prompt_for(row: dict[str, Any]) -> str:
    question_type = str(row["question_type"])
    question = str(row.get("question") or "")
    options = row.get("options") or {}
    option_text = "\n".join(f"{key}: {options[key]}" for key in sorted(options))
    if question_type in {"multi_choice", "open_ended_multi_choice"}:
        return (
            "You are answering a public CURE-Bench validation question. "
            "Reason internally, then end your response with exactly one line in the form `FINAL ANSWER: X`, "
            "where X is one of A, B, C, or D. Do not use external tools.\n\n"
            f"Question:\n{question}\n\nOptions:\n{option_text}\n"
        )
    return (
        "You are answering a public CURE-Bench validation question. "
        "Reason internally and provide a concise, clinically responsible final answer. "
        "Do not use external tools.\n\nQuestion:\n" + question
    )


def load_checkpoint() -> dict[str, dict[str, Any]]:
    path = CHECKPOINT_ROOT / "predictions.jsonl"
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != "evomind.cure.public_prediction.v1":
            raise ValueError("checkpoint_schema_invalid")
        key = str(row.get("id"))
        if key in result:
            raise ValueError("checkpoint_duplicate_id")
        result[key] = row
    return result


def run_inference(rows: list[dict[str, Any]], progress_path: Path, event_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), local_files_only=True, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    checkpoint = load_checkpoint()
    counts = {"completed": len(checkpoint), "total": len(rows)}
    write_json(progress_path, {"schema": "evomind.cure.public_inference_progress.v1", "status": "running", **counts, "updated_at": utc_now()})
    tokenizer.padding_side = "left"
    pending = [(index, row) for index, row in enumerate(rows) if str(row["id"]) not in checkpoint]
    for batch_start in range(0, len(pending), INFERENCE_BATCH_SIZE):
        batch = pending[batch_start:batch_start + INFERENCE_BATCH_SIZE]
        conversations = [[
            {"role": "system", "content": "You are a careful clinical reasoning assistant."},
            {"role": "user", "content": prompt_for(row)},
        ] for _index, row in batch]
        try:
            encoded = tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                reasoning_effort="high",
                model_identity="You are a careful clinical reasoning assistant.",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
        except (TypeError, KeyError):
            rendered = []
            for messages in conversations:
                try:
                    rendered.append(tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        reasoning_effort="high",
                        model_identity="You are a careful clinical reasoning assistant.",
                    ))
                except TypeError:
                    rendered.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
            encoded = tokenizer(rendered, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        for output_row, (index, row) in zip(output, batch):
            row_id = str(row["id"])
            raw = tokenizer.decode(output_row[input_ids.shape[-1]:], skip_special_tokens=True)
            answer = final_text(raw)
            question_type = str(row["question_type"])
            prediction = {
                "schema": "evomind.cure.public_prediction.v1",
                "id": row_id,
                "question_type": question_type,
                "choice": normalize_choice(answer) if question_type != "open_ended" else "NOTAVALUE",
                "final_answer": answer,
                "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "response_chars": len(raw),
                "model": REPO,
                "revision": REVISION,
            }
            append_jsonl(CHECKPOINT_ROOT / "predictions.jsonl", prediction)
            checkpoint[row_id] = prediction
            counts["completed"] = len(checkpoint)
            write_json(progress_path, {"schema": "evomind.cure.public_inference_progress.v1", "status": "running", **counts, "last_id": row_id, "last_index": index, "batch_size": INFERENCE_BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS, "updated_at": utc_now()})
            append_jsonl(event_path, {"schema": "evomind.cure.public_inference_event.v1", "event": "prediction_completed", "completed": counts["completed"], "total": counts["total"], "id": row_id, "batch_size": INFERENCE_BATCH_SIZE, "at_utc": utc_now()})
    metrics = score_predictions(rows, checkpoint)
    return checkpoint, metrics


def score_predictions(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = {"total": len(rows), "multi_choice_total": 0, "multi_choice_correct": 0, "open_ended_multi_choice_total": 0, "open_ended_multi_choice_correct": 0, "open_ended_total": 0}
    for row in rows:
        prediction = predictions.get(str(row["id"]))
        if prediction is None:
            raise ValueError("prediction_coverage_incomplete")
        question_type = str(row["question_type"])
        if question_type == "open_ended":
            counts["open_ended_total"] += 1
            continue
        key = "multi_choice" if question_type == "multi_choice" else "open_ended_multi_choice"
        counts[key + "_total"] += 1
        expected = str(row.get("correct_answer", row.get("answer", ""))).strip().upper()
        if prediction.get("choice") == expected:
            counts[key + "_correct"] += 1
    scored_total = counts["multi_choice_total"] + counts["open_ended_multi_choice_total"]
    scored_correct = counts["multi_choice_correct"] + counts["open_ended_multi_choice_correct"]
    return {
        "schema": "evomind.cure.public_validation_metrics.v1",
        "competition": "cure_bench",
        "dataset": "curebench_valset_pharse1.jsonl",
        "dataset_scope": "public_validation_only",
        "official_hidden_score": None,
        "accuracy_definition": "exact choice match on MC and OE-MC; open-ended excluded from exact accuracy as in starter framework",
        "accuracy": scored_correct / scored_total if scored_total else None,
        "correct_predictions": scored_correct,
        "scored_questions": scored_total,
        "counts": counts,
        "model": REPO,
        "revision": REVISION,
        "submission_performed": False,
        "phase2_accessed": False,
        "terms_accepted": False,
        "paper_comparison": "NOT_COMPARABLE_TO_PHASE1_TEST_OR_PHASE2_HIDDEN_SCORE",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    data_dir = bounded_path(Path(args.data_dir))
    if data_dir != DATA_ROOT.resolve(strict=True):
        raise ValueError("managed_cure_data_root_mismatch")
    output = Path(args.out_dir).resolve()
    output.relative_to(ROOT.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        model_identity = validate_model_assets()
        rows = load_validation()
        write_json(CHECKPOINT_ROOT / "run-identity.json", {
            "schema": "evomind.cure.public_validation_run.v1",
            "competition": "cure_bench",
            "model": model_identity,
            "data_manifest_sha256": DATA_MANIFEST_SHA256,
            "validation_sha256": VALIDATION_SHA256,
            "validation_rows": len(rows),
            "phase2_accessed": False,
            "submission_performed": False,
            "terms_accepted": False,
            "started_at": utc_now(),
        })
        progress_path = output / "inference-progress.json"
        event_path = output / "inference-events.jsonl"
        try:
            predictions, metrics = run_inference(rows, progress_path, event_path)
            ordered = [predictions[str(row["id"])] for row in rows]
            write_json(output / "metrics.json", metrics)
            write_json(output / "model-identity.json", model_identity)
            write_json(output / "run-identity.json", json.loads((CHECKPOINT_ROOT / "run-identity.json").read_text(encoding="utf-8")))
            (output / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in ordered), encoding="utf-8")
            write_json(output / "inference-environment.json", {
                "schema": "evomind.cure.public_inference_environment.v1",
                "python": sys.version,
                "platform": platform.platform(),
                "torch": __import__("torch").__version__,
                "transformers": __import__("transformers").__version__,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "gpu_used": True,
                "phase2_accessed": False,
                "submission_performed": False,
            })
            write_json(progress_path, {"schema": "evomind.cure.public_inference_progress.v1", "status": "completed", "completed": len(rows), "total": len(rows), "updated_at": utc_now()})
            result = {"schema": "evomind.cure.public_validation_execution.v1", "status": "PASS", "metrics": metrics, "model_identity": model_identity, "predictions": len(ordered), "checkpoint_root": str(CHECKPOINT_ROOT), "phase2_accessed": False, "submission_performed": False, "terms_accepted": False, "elapsed_seconds": round(time.monotonic() - started, 3)}
        except Exception as error:
            write_json(progress_path, {"schema": "evomind.cure.public_inference_progress.v1", "status": "failed", "error_class": type(error).__name__, "completed": len(load_checkpoint()), "total": len(rows) if 'rows' in locals() else None, "updated_at": utc_now()})
            result = {"schema": "evomind.cure.public_validation_execution.v1", "status": "FAIL", "error_class": type(error).__name__, "partial_checkpoint_preserved": True, "phase2_accessed": False, "submission_performed": False, "terms_accepted": False, "elapsed_seconds": round(time.monotonic() - started, 3)}
            write_json(output / "execution-result.json", result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
            return 1
        write_json(output / "execution-result.json", result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
