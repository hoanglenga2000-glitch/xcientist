#!/usr/bin/env python3
"""Prototype a fold-clean character Transformer for Russian `_trans` tokens on HPC CPU."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import mlebench_remote_ops as remote_ops
from scripts.validate_russian_normalization_remote import (
    REMOTE_DATASET,
    REMOTE_ROOT,
    _run_remote_source,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "text_normalization_russian_transliteration_fold0_current.json"
)


def remote_source(*, rows: int, fold: int, folds: int, epochs: int, source_stress: bool = False) -> str:
    return f'''from __future__ import annotations
import csv
import io
import json
import math
import random
import re
import zipfile
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATASET = {REMOTE_DATASET!r}
ROW_LIMIT = {rows}
HELD_OUT_FOLD = {fold}
FOLDS = {folds}
EPOCHS = {epochs}
SOURCE_STRESS = {source_stress!r}
SEED = 42
MAX_INPUT = 40
MAX_OUTPUT = 48
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


def read_rows():
    result = []
    with zipfile.ZipFile(DATASET) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            for index, row in enumerate(reader):
                if ROW_LIMIT and index >= ROW_LIMIT:
                    break
                result.append((row["sentence_id"], row["class"], row["before"], row["after"]))
    return result


def group_folds(rows):
    sizes = Counter(row[0] for row in rows)
    loads = [0] * FOLDS
    assignment = {{}}
    for sentence_id, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        selected = min(range(FOLDS), key=lambda index: (loads[index], index))
        assignment[sentence_id] = selected
        loads[selected] += size
    return assignment, loads


def transliteration_target(after):
    units = after.split()
    if not units:
        return None
    target = []
    for unit in units:
        match = re.fullmatch(r"([а-яё])_trans", unit, flags=re.IGNORECASE)
        if not match:
            return None
        target.append(match.group(1).lower())
    return "".join(target)


def best_map(rows, assignment, train=True):
    pairs = Counter()
    for sentence_id, token_class, before, after in rows:
        if is_training(sentence_id) == train:
            pairs[(before, after)] += 1
    best = {{}}
    for (before, after), count in pairs.items():
        current = best.get(before)
        if current is None or count > current[0] or (count == current[0] and after < current[1]):
            best[before] = (count, after)
    return {{before: after for before, (count, after) in best.items()}}


rows = read_rows()
assignment, fold_sizes = group_folds(rows)
ordered_sentence_ids = list(dict.fromkeys(row[0] for row in rows))
source_split = max(1, int(len(ordered_sentence_ids) * 0.8))
source_training_ids = set(ordered_sentence_ids[:source_split])


def is_training(sentence_id):
    if SOURCE_STRESS:
        return sentence_id in source_training_ids
    return assignment[sentence_id] != HELD_OUT_FOLD


training_word_pairs = Counter()
for sentence_id, token_class, before, after in rows:
    if not is_training(sentence_id) or not re.fullmatch(r"[A-Za-z]{{2,40}}", before):
        continue
    target = transliteration_target(after)
    if target and len(target) <= MAX_OUTPUT - 2:
        training_word_pairs[(before.lower(), target)] += 1

best_words = {{}}
for (word, target), count in training_word_pairs.items():
    current = best_words.get(word)
    if current is None or count > current[0] or (count == current[0] and target < current[1]):
        best_words[word] = (count, target)
pairs = [(word, target) for word, (count, target) in best_words.items()]
random.shuffle(pairs)

source_symbols = ["<pad>", "<bos>", "<eos>"] + list("abcdefghijklmnopqrstuvwxyz")
target_chars = sorted({{char for word, target in pairs for char in target}})
target_symbols = ["<pad>", "<bos>", "<eos>"] + target_chars
source_to_id = {{value: index for index, value in enumerate(source_symbols)}}
target_to_id = {{value: index for index, value in enumerate(target_symbols)}}
target_from_id = {{index: value for value, index in target_to_id.items()}}
PAD = 0
BOS = 1
EOS = 2


def encode_source(word):
    return [BOS] + [source_to_id[char] for char in word.lower() if char in source_to_id] + [EOS]


def encode_target(target):
    return [BOS] + [target_to_id[char] for char in target] + [EOS]


class PairDataset(Dataset):
    def __len__(self): return len(pairs)
    def __getitem__(self, index):
        word, target = pairs[index]
        return encode_source(word), encode_target(target)


def collate(batch):
    source_length = max(len(item[0]) for item in batch)
    target_length = max(len(item[1]) for item in batch)
    source = torch.full((len(batch), source_length), PAD, dtype=torch.long)
    target = torch.full((len(batch), target_length), PAD, dtype=torch.long)
    for index, (source_ids, target_ids) in enumerate(batch):
        source[index, :len(source_ids)] = torch.tensor(source_ids)
        target[index, :len(target_ids)] = torch.tensor(target_ids)
    return source, target


class Transliterator(nn.Module):
    def __init__(self):
        super().__init__()
        dimension = 96
        self.source_embedding = nn.Embedding(len(source_symbols), dimension, padding_idx=PAD)
        self.target_embedding = nn.Embedding(len(target_symbols), dimension, padding_idx=PAD)
        self.source_position = nn.Embedding(MAX_INPUT + 2, dimension)
        self.target_position = nn.Embedding(MAX_OUTPUT + 2, dimension)
        self.transformer = nn.Transformer(
            d_model=dimension, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=256, dropout=0.10, batch_first=True,
        )
        self.output = nn.Linear(dimension, len(target_symbols))

    def encode(self, source):
        positions = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
        embedded = self.source_embedding(source) + self.source_position(positions)
        return self.transformer.encoder(embedded, src_key_padding_mask=source.eq(PAD))

    def decode(self, target, memory, source_padding):
        positions = torch.arange(target.shape[1], device=target.device).unsqueeze(0)
        embedded = self.target_embedding(target) + self.target_position(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(target.shape[1], device=target.device)
        decoded = self.transformer.decoder(
            embedded, memory, tgt_mask=causal,
            tgt_key_padding_mask=target.eq(PAD), memory_key_padding_mask=source_padding,
        )
        return self.output(decoded)

    def forward(self, source, target):
        memory = self.encode(source)
        return self.decode(target, memory, source.eq(PAD))


model = Transliterator()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)
loader = DataLoader(PairDataset(), batch_size=192, shuffle=True, collate_fn=collate)
history = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for source, target in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(source, target[:, :-1])
        labels = target[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens = labels.ne(PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    history.append(total_loss / max(total_tokens, 1))


def predict_words(words):
    result = {{}}
    model.eval()
    ordered = sorted(words)
    with torch.inference_mode():
        for start in range(0, len(ordered), 256):
            batch = ordered[start:start + 256]
            encoded = [encode_source(word) for word in batch]
            length = max(len(item) for item in encoded)
            source = torch.full((len(batch), length), PAD, dtype=torch.long)
            for index, item in enumerate(encoded):
                source[index, :len(item)] = torch.tensor(item)
            memory = model.encode(source)
            generated = torch.full((len(batch), 1), BOS, dtype=torch.long)
            finished = torch.zeros(len(batch), dtype=torch.bool)
            for step in range(MAX_OUTPUT):
                logits = model.decode(generated, memory, source.eq(PAD))
                next_id = logits[:, -1].argmax(dim=-1)
                generated = torch.cat([generated, next_id[:, None]], dim=1)
                finished |= next_id.eq(EOS)
                if finished.all():
                    break
            for word, ids in zip(batch, generated[:, 1:].tolist()):
                chars = []
                for value in ids:
                    if value == EOS: break
                    if value not in (PAD, BOS): chars.append(target_from_id[value])
                result[word] = "".join(chars)
    return result


token_map = best_map(rows, assignment, train=True)
validation_words = {{
    before.lower() for sentence_id, token_class, before, after in rows
    if not is_training(sentence_id)
    and before not in token_map
    and re.fullmatch(r"[A-Za-z]{{2,40}}", before)
}}
word_predictions = predict_words(validation_words)

base_correct = 0
candidate_correct = 0
candidate_applied = 0
candidate_gain = 0
candidate_loss = 0
trans_rows = 0
trans_exact = 0
unseen_trans_rows = 0
unseen_trans_exact = 0
top_errors = Counter()
validation_rows = 0
for sentence_id, token_class, before, after in rows:
    if is_training(sentence_id):
        continue
    validation_rows += 1
    base = token_map.get(before, before)
    candidate = base
    predicted_target = word_predictions.get(before.lower()) if before not in token_map else None
    if predicted_target:
        candidate = " ".join(f"{{char}}_trans" for char in predicted_target)
        candidate_applied += 1
    base_ok = base == after
    candidate_ok = candidate == after
    base_correct += int(base_ok)
    candidate_correct += int(candidate_ok)
    candidate_gain += int(candidate_ok and not base_ok)
    candidate_loss += int(base_ok and not candidate_ok)
    true_target = transliteration_target(after)
    if true_target is not None and re.fullmatch(r"[A-Za-z]{{2,40}}", before):
        trans_rows += 1
        trans_exact += int(predicted_target == true_target)
        if before not in token_map:
            unseen_trans_rows += 1
            unseen_trans_exact += int(predicted_target == true_target)
            if predicted_target != true_target:
                top_errors[(before, true_target, predicted_target or "")] += 1

payload = {{
    "rows": len(rows), "folds": FOLDS, "held_out_fold": HELD_OUT_FOLD,
    "validation_mode": "source_ordered_last_20pct" if SOURCE_STRESS else "grouped_fold",
    "fold_sizes": fold_sizes, "training_pairs": len(pairs),
    "target_vocabulary": target_chars, "epochs": EPOCHS, "loss_history": history,
    "validation_rows": validation_rows,
    "base_accuracy": base_correct / validation_rows,
    "candidate_accuracy": candidate_correct / validation_rows,
    "candidate_applied_rows": candidate_applied,
    "candidate_gain": candidate_gain, "candidate_loss": candidate_loss,
    "transliteration_rows": trans_rows,
    "transliteration_exact_accuracy": trans_exact / trans_rows if trans_rows else None,
    "unseen_transliteration_rows": unseen_trans_rows,
    "unseen_transliteration_exact_accuracy": unseen_trans_exact / unseen_trans_rows if unseen_trans_rows else None,
    "top_unseen_errors": [
        {{"before": key[0], "target": key[1], "prediction": key[2], "count": count}}
        for key, count in top_errors.most_common(80)
    ],
}}
print("EVOMIND_RESULT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3_600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-stress", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.fold < args.folds):
        raise SystemExit("--fold must be within [0, --folds)")
    client = remote_ops._connect()
    try:
        preflight = (
            f"test -d {REMOTE_ROOT} && test -w {REMOTE_ROOT} && "
            f"test -r {REMOTE_DATASET} && printf 'PREFLIGHT_OK'"
        )
        code, output, error = remote_ops._run_remote(client, preflight, timeout=30)
        if code != 0 or output.strip() != "PREFLIGHT_OK":
            raise RuntimeError("Remote transliteration preflight failed")
        code, output, error = _run_remote_source(
            client,
            remote_source(
                rows=args.rows,
                fold=args.fold,
                folds=args.folds,
                epochs=args.epochs,
                source_stress=args.source_stress,
            ),
            timeout=args.timeout,
        )
    finally:
        client.close()
    if code != 0:
        raise RuntimeError("Remote transliteration prototype failed:\n" + "\n".join(error.splitlines()[-20:]))
    marker = "EVOMIND_RESULT="
    matches = [line for line in output.splitlines() if line.startswith(marker)]
    if len(matches) != 1:
        raise RuntimeError("Remote transliteration prototype returned no unique result marker")
    result = json.loads(matches[0][len(marker) :])
    payload = {
        "schema": "evomind.mlebench.russian_transliteration_oof.v1",
        "access_mode": "remote_read_only_cpu_probe",
        "path_scope_verified": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "remote_stderr_type": "nonempty" if error else "empty",
        **result,
    }
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(path),
        "fold": payload["held_out_fold"],
        "validation_mode": payload["validation_mode"],
        "epochs": payload["epochs"],
        "training_pairs": payload["training_pairs"],
        "candidate_gain": payload["candidate_gain"],
        "candidate_loss": payload["candidate_loss"],
        "unseen_transliteration_exact_accuracy": payload["unseen_transliteration_exact_accuracy"],
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
