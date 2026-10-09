import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('cure_batch_scheduler', Path(__file__).resolve().parents[1] / 'scripts/cure_batch_scheduler.py')
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)


def test_heterogeneous_lengths_preserve_exact_ids_and_limits():
    tasks = [{'id': str(index), 'kind': ('multi_choice', 'open_ended_multi_choice', 'open_ended')[index % 3],
              'input_tokens': (128, 512, 1500, 3000)[index % 4]} for index in range(459)]
    batches = scheduler.plan_batches(tasks)
    flattened = [identifier for batch in batches for identifier in batch['ids']]
    assert set(flattened) == {task['id'] for task in tasks}
    assert len(flattened) == len(set(flattened)) == 459
    assert batches == scheduler.plan_batches(tasks)
    for batch in batches:
        assert batch['batch_size'] <= 32
        assert batch['padded_input_tokens'] <= 8192
        assert batch['cache_token_bound'] <= 131072
        assert len({tasks[int(identifier)]['kind'] for identifier in batch['ids']}) == 1


def test_mapping_is_not_mixed_with_initial_answers():
    tasks = [{'id': 'map', 'kind': 'mapping', 'input_tokens': 1200},
             {'id': 'answer', 'kind': 'open_ended_multi_choice', 'input_tokens': 500}]
    batches = scheduler.plan_batches(tasks)
    assert [batch['kind'] for batch in batches] == ['open_ended_multi_choice', 'mapping']


@pytest.mark.parametrize('task', [
    {'id': '', 'kind': 'mapping', 'input_tokens': 10},
    {'id': 'a', 'kind': 'hidden_test', 'input_tokens': 10},
    {'id': 'a', 'kind': 'mapping', 'input_tokens': 0},
    {'id': 'a', 'kind': 'mapping', 'input_tokens': True},
    {'id': 'a', 'kind': 'mapping', 'input_tokens': 50000},
])
def test_invalid_or_oversize_inputs_are_rejected_not_truncated(task):
    with pytest.raises(ValueError):
        scheduler.plan_batches([task])


def test_duplicate_id_rejected():
    task = {'id': 'a', 'kind': 'multi_choice', 'input_tokens': 100}
    with pytest.raises(ValueError):
        scheduler.plan_batches([task, task])


def test_cache_limit_can_constrain_even_short_inputs():
    tasks = [{'id': str(index), 'kind': 'multi_choice', 'input_tokens': 10} for index in range(10)]
    batches = scheduler.plan_batches(tasks, scheduler.BatchLimits(max_cache_tokens=9000))
    assert [batch['batch_size'] for batch in batches] == [2] * 5


def test_empty_tasks_do_not_generate_work():
    assert scheduler.plan_batches([]) == []
