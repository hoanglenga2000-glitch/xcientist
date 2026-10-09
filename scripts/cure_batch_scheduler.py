"""Deterministic, CPU-only batch planning; never truncates or reads labels.

This planner is a candidate for a new frozen execution protocol. It must not be
silently applied to v1/v2 checkpoints, whose source and protocol remain frozen.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchLimits:
    # 32 x 512 was measured; 64 x 512 OOMed. These are deliberately
    # lower prefill/cache bounds, not proof that a 4096-token run is validated.
    max_batch_size: int = 32
    max_padded_input_tokens: int = 8192
    max_cache_tokens: int = 131072
    max_new_tokens: int = 4096


def plan_batches(tasks: list[dict], limits: BatchLimits = BatchLimits()) -> list[dict]:
    """Group compatible stages/types and nearby lengths, preserving every ID once."""
    if any(type(value) is not int or value <= 0 for value in vars(limits).values()):
        raise ValueError('positive_integer_limits_required')
    groups = defaultdict(list)
    ids = set()
    for index, task in enumerate(tasks):
        identifier = task.get('id')
        length = task.get('input_tokens')
        kind = task.get('kind')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('invalid_or_duplicate_task_id')
        if type(length) is not int or length <= 0:
            raise ValueError('invalid_input_token_count')
        if kind not in ('multi_choice', 'open_ended_multi_choice', 'open_ended', 'mapping'):
            raise ValueError('invalid_stage_or_question_type')
        if length > limits.max_padded_input_tokens or length + limits.max_new_tokens > limits.max_cache_tokens:
            raise ValueError('one_task_exceeds_capacity_no_truncation')
        ids.add(identifier)
        groups[kind].append((length, index, identifier))
    batches = []
    for kind in ('multi_choice', 'open_ended_multi_choice', 'open_ended', 'mapping'):
        group = sorted(groups[kind], key=lambda value: (-value[0], value[1]))
        pending = []

        def publish():
            if not pending:
                return
            longest = pending[0][0]
            batches.append({'kind': kind, 'ids': [item[2] for item in pending],
                'batch_size': len(pending), 'max_input_tokens': longest,
                'padded_input_tokens': len(pending) * longest,
                'cache_token_bound': len(pending) * (longest + limits.max_new_tokens)})

        for item in group:
            longest = pending[0][0] if pending else item[0]
            size = len(pending) + 1
            if pending and (size > limits.max_batch_size
                    or size * longest > limits.max_padded_input_tokens
                    or size * (longest + limits.max_new_tokens) > limits.max_cache_tokens):
                publish()
                pending = []
            pending.append(item)
        publish()
    if sum(batch['batch_size'] for batch in batches) != len(tasks):
        raise AssertionError('batch_coverage_invariant')
    return batches
