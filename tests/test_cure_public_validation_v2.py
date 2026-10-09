"""CPU-only protocol tests; importing this module must not import torch."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cure_public_v2', ROOT / 'scripts/run_cure_gptoss_public_validation_v2.py')
cure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cure)


def example(kind='multi_choice', index=0):
    return {'id': str(index), 'question_type': kind, 'question': 'QUESTION_ONLY',
            'options': {'A': 'OPTION_SECRET_ALPHA', 'B': 'OPTION_SECRET_BETA', 'E': 'OPTION_SECRET_EPSILON'},
            'correct_answer': 'A', 'answer': 'LABEL_MUST_NOT_ENTER_PROMPT'}


def test_oe_mc_options_hidden_until_mapping():
    row = example('open_ended_multi_choice')
    prompt = cure.prompt_for(row)
    assert 'QUESTION_ONLY' in prompt
    assert 'OPTION_SECRET' not in prompt
    assert 'LABEL_MUST_NOT_ENTER_PROMPT' not in prompt
    mapped = cure.mapping_prompt(row, 'Already frozen open answer')
    assert 'OPTION_SECRET_EPSILON' in mapped
    assert 'Already frozen open answer' in mapped
    assert 'LABEL_MUST_NOT_ENTER_PROMPT' not in mapped
    with pytest.raises(ValueError):
        cure.mapping_prompt(row, '')


def test_prompt_parity_with_pinned_starter():
    source = (ROOT / 'artifacts/cure-official-starter-20260907/dataset_utils.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CureBenchDataset')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__getitem__')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<pinned-starter-dataset>', 'exec'), namespace)
    for kind in cure.PROTOCOL['type_counts']:
        row = example(kind)
        _, _, question, _, meta = namespace['__getitem__'](SimpleNamespace(data=[row]), 0)
        if kind == 'multi_choice':
            expected = f'The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n\nAnswer:'
        else:
            expected = f'The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question}\n\nAnswer:'
        assert cure.prompt_for(row) == expected
        if meta:
            assert cure.mapping_prompt(row, 'test answer') == f"{meta}Agent's answer: test answer\n\nMulti-choice answer:"


@pytest.mark.parametrize('raw,expected,status', [
    ('<|channel|>analysis<|message|>Answer is B', '', 'missing_final_channel'),
    ('<|channel|>analysis<|message|>A<|end|><|start|>assistant<|channel|>final<|message|>E<|return|>', 'E', 'completed'),
    ('<|meta_sep|>final<|im_sep|>B<|fim_suffix|><|pad|>', 'B', 'completed'),
    ('<|channel|>final<|message|>A', '', 'incomplete_final_channel'),
    ('<|channel|>final<|message|>A<|end|>analysis<|return|>', '', 'unexpected_final_structure'),
    ('<|channel|>final<|message|>  <|return|>', '', 'empty_final_channel'),
])
def test_only_complete_final_is_retained(raw, expected, status):
    assert cure.final_channel(raw) == (expected, status)


def test_choice_matches_official_method():
    # The starter has unrelated syntax problems; extract just this method.
    source = (ROOT / 'artifacts/cure-official-starter-20260907/eval_framework.py').read_text(encoding='utf-8')
    fragment = source.split('    def _extract_multiple_choice_answer(', 1)[1].split('    def save_submission(', 1)[0]
    import textwrap
    fragment = textwrap.dedent('    def _extract_multiple_choice_answer(' + fragment)
    namespace = {}
    exec(compile(fragment, '<pinned-starter-choice>', 'exec'), namespace)
    for answer in ('A', 'B)', 'The answer is E', 'None', '', 'Answer: C', 'E. medicine', 'Option D', 'unclear'):
        assert cure.extract_choice(answer) == namespace['_extract_multiple_choice_answer'](None, answer)


def record(row, stage='prediction'):
    return {'id': row['id'], 'question_type': row['question_type'], 'stage': stage,
            'protocol_sha256': cure.PROTOCOL_SHA256, 'choice': 'A',
            'answer_status': 'completed', 'final_answer': 'A'}


def test_checkpoint_rejects_protocol_duplicate_unknown_and_torn(tmp_path):
    rows = [example()]
    path = tmp_path / 'predictions.jsonl'
    valid = record(rows[0])
    path.write_text(json.dumps(valid) + '\n', encoding='utf-8')
    assert set(cure.load_records(path, rows, 'prediction')) == {'0'}
    for values in ([{**valid, 'protocol_sha256': 'old'}], [valid, valid], [{**valid, 'id': 'unknown'}], [{**valid, 'stage': 'answer'}]):
        path.write_text('\n'.join(json.dumps(value) for value in values), encoding='utf-8')
        with pytest.raises(ValueError):
            cure.load_records(path, rows, 'prediction')
    path.write_text(json.dumps(valid) + '\n{"id":', encoding='utf-8')
    with pytest.raises(ValueError):
        cure.load_records(path, rows, 'prediction')


def test_checkpoint_source_binding_and_no_legacy_import(tmp_path, monkeypatch):
    monkeypatch.setattr(cure, 'CHECKPOINT_ROOT', tmp_path)
    identity = {'protocol_sha256': cure.PROTOCOL_SHA256, 'source_sha256': 'version-one'}
    cure.bind_checkpoint(identity)
    cure.bind_checkpoint(identity)
    with pytest.raises(ValueError, match='identity_drift'):
        cure.bind_checkpoint({**identity, 'source_sha256': 'version-two'})
    assert 'cure-public-validation-v1' not in str(cure.CHECKPOINT_ROOT)
    assert len(cure.MODEL_NAMES) == 13


def test_scoring_full_413_denominator_invalids_count_as_wrong():
    rows = []
    for kind, size in cure.PROTOCOL['type_counts'].items():
        offset = len(rows)
        rows.extend(example(kind, index + offset) for index in range(size))
    predictions = {row['id']: record(row) for row in rows}
    metrics = cure.score_predictions(rows, predictions)
    assert metrics['scored_questions'] == 413
    assert metrics['accuracy'] == 1
    first = rows[0]['id']
    predictions[first].update(choice='', answer_status='missing_final_channel')
    assert cure.score_predictions(rows, predictions)['accuracy'] == 412 / 413
    assert metrics['open_ended_score'] is None
    del predictions[first]
    with pytest.raises(ValueError, match='coverage_incomplete'):
        cure.score_predictions(rows, predictions)


def test_batched_resume_preserves_answer_and_maps_only_after_answer(tmp_path, monkeypatch):
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'USE_TF', 'USE_FLAX'):
        monkeypatch.setenv(key, cure.os.environ.get(key, ''))
    rows = [example('multi_choice', 0), example('open_ended_multi_choice', 1), example('open_ended', 2)]
    monkeypatch.setattr(cure, 'CHECKPOINT_ROOT', tmp_path)
    monkeypatch.setattr(cure, 'MODEL_DIR', tmp_path / 'model')
    old_answer = record(rows[1], 'answer')
    old_answer['final_answer'] = 'Frozen open response'
    cure.append_jsonl(tmp_path / 'answers.jsonl', old_answer)
    answer_bytes = (tmp_path / 'answers.jsonl').read_bytes()
    prompts = []

    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = (len(values), len(values[0]))
        def to(self, device):
            return self
        def __getitem__(self, key):
            return [row[key[1]] for row in self.values[key[0]]]

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9
        def apply_chat_template(self, conversations, **kwargs):
            assert kwargs['padding'] is True
            prompts.extend(conversation[0]['content'] for conversation in conversations)
            return {'input_ids': Tensor([[1, 2]] * len(conversations)), 'attention_mask': Tensor([[1, 1]] * len(conversations))}
        def decode(self, tokens, **kwargs):
            assert kwargs['skip_special_tokens'] is False
            return '<|channel|>final<|message|>A<|return|>'

    class Model:
        def eval(self):
            pass
        def parameters(self):
            return [SimpleNamespace(device=SimpleNamespace(type='cuda'))]
        def generate(self, input_ids, **kwargs):
            return Tensor([[1, 2, 3]] * input_ids.shape[0])

    import contextlib
    fake_torch = SimpleNamespace(manual_seed=lambda seed: None, bfloat16='bf16',
        inference_mode=contextlib.nullcontext, __version__='fake',
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                            get_device_name=lambda device: 'FAKE_TEST', memory_allocated=lambda: 0))
    fake_transformers = SimpleNamespace(__version__='fake', Mxfp4Config=lambda **kw: kw,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    predictions = cure.run_batches(rows, tmp_path, cure.time.monotonic() + 500)
    assert len(predictions) == 3
    assert len(prompts) == 3  # Two new answers and one mapping; no repeated open answer.
    assert "Agent's answer: Frozen open response" in prompts[-1]
    assert predictions['1']['mapping']['answer_status'] == 'completed'
    assert (tmp_path / 'answers.jsonl').read_bytes().startswith(answer_bytes)
    before = (tmp_path / 'predictions.jsonl').read_bytes()
    cure.run_batches(rows, tmp_path, cure.time.monotonic() + 500)
    assert (tmp_path / 'predictions.jsonl').read_bytes() == before
    assert len(prompts) == 3
