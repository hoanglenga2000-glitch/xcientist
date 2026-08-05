#!/usr/bin/env python3
"""Fold-clean character Transformer for Russian text-normalization `_trans` outputs."""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any, Iterable

import pandas as pd


def extract_russian_transliteration_target(after: str) -> str | None:
    units = str(after).split()
    if not units:
        return None
    target: list[str] = []
    for unit in units:
        match = re.fullmatch(r"([а-яё])_trans", unit, flags=re.IGNORECASE)
        if not match:
            return None
        target.append(match.group(1).lower())
    return "".join(target)


def collect_russian_transliteration_pairs(
    frame: pd.DataFrame,
    *,
    max_input_length: int = 40,
    max_output_length: int = 48,
) -> list[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for before, after in zip(frame["before"].astype(str), frame["after"].astype(str)):
        if not re.fullmatch(rf"[A-Za-z]{{2,{max_input_length}}}", before):
            continue
        target = extract_russian_transliteration_target(after)
        if target and len(target) <= max_output_length - 2:
            counts[(before.lower(), target)] += 1
    best: dict[str, tuple[int, str]] = {}
    for (word, target), count in counts.items():
        current = best.get(word)
        if current is None or count > current[0] or (count == current[0] and target < current[1]):
            best[word] = (count, target)
    return sorted((word, target) for word, (count, target) in best.items())


def predict_russian_transliterations(
    training_frame: pd.DataFrame,
    tokens: Iterable[str],
    *,
    epochs: int = 40,
    seed: int = 42,
    device: str = "cpu",
    batch_size: int = 192,
    max_input_length: int = 40,
    max_output_length: int = 48,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Train only on the supplied fold and predict unique unseen Latin tokens."""

    requested = sorted({
        str(token).lower()
        for token in tokens
        if re.fullmatch(rf"[A-Za-z]{{2,{max_input_length}}}", str(token))
    })
    pairs = collect_russian_transliteration_pairs(
        training_frame,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
    )
    if epochs <= 0 or not requested or not pairs:
        return {}, {
            "training_pairs": len(pairs),
            "requested_tokens": len(requested),
            "predicted_tokens": 0,
            "epochs": max(0, epochs),
            "device": "disabled",
            "loss_history": [],
        }

    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for Russian transliteration") from exc

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Russian transliteration requested CUDA but CUDA is unavailable")
    if resolved_device == "cpu":
        # This runner is intentionally CPU-light so it can coexist with an
        # unrelated GPU job, but two threads left most of Job 89441's CPU
        # allocation idle.  Eight intra-op threads is a bounded compromise for
        # the small Transformer matrices and avoids oversubscribing the shared
        # node.  Inter-op changes can be rejected after PyTorch starts work, so
        # keep that tuning best-effort.
        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
        try:
            torch.set_num_interop_threads(min(2, max(1, torch.get_num_interop_threads())))
        except RuntimeError:
            pass

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    source_symbols = ["<pad>", "<bos>", "<eos>"] + list("abcdefghijklmnopqrstuvwxyz")
    target_chars = sorted({char for word, target in pairs for char in target})
    target_symbols = ["<pad>", "<bos>", "<eos>"] + target_chars
    source_to_id = {value: index for index, value in enumerate(source_symbols)}
    target_to_id = {value: index for index, value in enumerate(target_symbols)}
    target_from_id = {index: value for value, index in target_to_id.items()}
    pad, bos, eos = 0, 1, 2

    def encode_source(word: str) -> list[int]:
        return [bos] + [source_to_id[char] for char in word if char in source_to_id] + [eos]

    def encode_target(target: str) -> list[int]:
        return [bos] + [target_to_id[char] for char in target] + [eos]

    def collate(items: list[tuple[str, str]]) -> tuple[Any, Any]:
        encoded_source = [encode_source(word) for word, target in items]
        encoded_target = [encode_target(target) for word, target in items]
        source = torch.full((len(items), max(map(len, encoded_source))), pad, dtype=torch.long)
        target = torch.full((len(items), max(map(len, encoded_target))), pad, dtype=torch.long)
        for index, values in enumerate(encoded_source):
            source[index, : len(values)] = torch.tensor(values)
        for index, values in enumerate(encoded_target):
            target[index, : len(values)] = torch.tensor(values)
        return source.to(resolved_device), target.to(resolved_device)

    class Transliterator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = 96
            self.source_embedding = nn.Embedding(len(source_symbols), dimension, padding_idx=pad)
            self.target_embedding = nn.Embedding(len(target_symbols), dimension, padding_idx=pad)
            self.source_position = nn.Embedding(max_input_length + 2, dimension)
            self.target_position = nn.Embedding(max_output_length + 2, dimension)
            self.transformer = nn.Transformer(
                d_model=dimension,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=256,
                dropout=0.10,
                batch_first=True,
            )
            self.output = nn.Linear(dimension, len(target_symbols))

        def encode(self, source: Any) -> Any:
            positions = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
            embedded = self.source_embedding(source) + self.source_position(positions)
            return self.transformer.encoder(embedded, src_key_padding_mask=source.eq(pad))

        def decode(self, target: Any, memory: Any, source_padding: Any) -> Any:
            positions = torch.arange(target.shape[1], device=target.device).unsqueeze(0)
            embedded = self.target_embedding(target) + self.target_position(positions)
            # Use a boolean causal mask so its dtype matches the boolean
            # key-padding masks.  Besides removing a PyTorch deprecation
            # warning, this prevents one warning write for every decoder call
            # during both training and autoregressive inference.
            causal = torch.triu(
                torch.ones(
                    (target.shape[1], target.shape[1]),
                    dtype=torch.bool,
                    device=target.device,
                ),
                diagonal=1,
            )
            decoded = self.transformer.decoder(
                embedded,
                memory,
                tgt_mask=causal,
                tgt_key_padding_mask=target.eq(pad),
                memory_key_padding_mask=source_padding,
            )
            return self.output(decoded)

        def forward(self, source: Any, target: Any) -> Any:
            return self.decode(target, self.encode(source), source.eq(pad))

    model = Transliterator().to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad)
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []
    best_training_loss = float("inf")
    stale_epochs = 0
    minimum_epochs = min(12, epochs)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(pairs), generator=generator).tolist()
        total_loss = 0.0
        total_tokens = 0
        for start in range(0, len(order), batch_size):
            batch = [pairs[index] for index in order[start : start + batch_size]]
            source, target = collate(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, target[:, :-1])
            labels = target[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(labels.ne(pad).sum().item())
            total_loss += float(loss.item()) * count
            total_tokens += count
        epoch_loss = total_loss / max(total_tokens, 1)
        history.append(epoch_loss)
        if epoch_loss < best_training_loss * (1.0 - 1e-4):
            best_training_loss = epoch_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if len(history) >= minimum_epochs and stale_epochs >= 6:
            break

    predictions: dict[str, str] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(requested), 256):
            words = requested[start : start + 256]
            source, _ = collate([(word, pairs[0][1]) for word in words])
            memory = model.encode(source)
            generated = torch.full((len(words), 1), bos, dtype=torch.long, device=resolved_device)
            finished = torch.zeros(len(words), dtype=torch.bool, device=resolved_device)
            for _ in range(max_output_length):
                logits = model.decode(generated, memory, source.eq(pad))
                next_id = logits[:, -1].argmax(dim=-1)
                generated = torch.cat([generated, next_id[:, None]], dim=1)
                finished |= next_id.eq(eos)
                if bool(finished.all()):
                    break
            for word, values in zip(words, generated[:, 1:].cpu().tolist()):
                chars: list[str] = []
                for value in values:
                    if value == eos:
                        break
                    if value not in (pad, bos):
                        chars.append(target_from_id[value])
                if chars:
                    predictions[word] = " ".join(f"{char}_trans" for char in chars)

    return predictions, {
        "training_pairs": len(pairs),
        "requested_tokens": len(requested),
        "predicted_tokens": len(predictions),
        "epochs": epochs,
        "epochs_completed": len(history),
        "device": resolved_device,
        "intraop_threads": int(torch.get_num_threads()),
        "boolean_attention_masks": True,
        "loss_history": history,
    }
