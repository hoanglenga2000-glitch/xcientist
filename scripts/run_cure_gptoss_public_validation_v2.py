"""Protocol-bound CURE public validation; managed executor entry, never a submission.

V1 is retained unchanged. Its OE-MC option exposure and unbound checkpoints make
those predictions ineligible for this protocol. Only final-channel text is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra')
DATA_ROOT = ROOT / 'competition_data/cure_bench'
MODEL_ROOT = DATA_ROOT / 'managed_runtime/gpt-oss-20b-r1'
MODEL_DIR = MODEL_ROOT / 'model'
VALIDATION_PATH = DATA_ROOT / 'data/curebench_valset_pharse1.jsonl'
CHECKPOINT_ROOT = MODEL_ROOT / 'validation-runs/cure-public-validation-v2'
REPO = 'openai/gpt-oss-20b'
REVISION = '6cee5e81ee83917806bbde320786a8fb61efebee'
DATA_MANIFEST_SHA256 = 'b2af0ade5f012bae9722192c1c7575815f788228b8f8de509be6506438b90197'
VALIDATION_SHA256 = '6b45f291fb776c41492273a8fec694381fc63078bb340c59559a8be29b738ca2'
MODEL_FILES = {
    'model-00000-of-00002.safetensors': (4792272488, '16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427'),
    'model-00001-of-00002.safetensors': (4798702184, '4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea'),
    'model-00002-of-00002.safetensors': (4170342232, 'a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303'),
}
MODEL_NAMES = set(MODEL_FILES) | {
    'LICENSE', 'README.md', 'USAGE_POLICY', 'chat_template.jinja', 'config.json',
    'generation_config.json', 'model.safetensors.index.json',
    'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
}
PROTOCOL = {
    'schema': 'evomind.cure.public_protocol.v2', 'model': REPO,
    'revision': REVISION, 'validation_sha256': VALIDATION_SHA256,
    'data_manifest_sha256': DATA_MANIFEST_SHA256,
    'rows': 459, 'type_counts': {'multi_choice': 183, 'open_ended_multi_choice': 230, 'open_ended': 46},
    'starter_dataset_sha256': '2bc673654c6fd26cc953d6f045b919da683e4065fa45ab82af17245a8af0b97a',
    'starter_framework_sha256': '12ae2cad79694832b984530a9d0bdb359030f50d434cf9af6b2097a82484d2fc',
    'oe_mc': 'answer_without_options_then_separate_option_mapping',
    'batch_size': 4, 'max_new_tokens': 1024, 'reasoning_effort': 'medium',
    'do_sample': False, 'seed': 93207, 'dtype': 'bfloat16', 'device': 'cuda:0_only',
    'answer_parser': 'complete_harmony_final_channel_only_v2',
    'choice_parser': 'official_starter_extract_multiple_choice_answer',
    'accuracy_denominator': 413, 'open_ended_scoring': 'not_available',
    'paper_comparison': 'NOT_COMPARABLE_TO_PHASE1_TEST_OR_PHASE2_HIDDEN_SCORE',
    'differences_from_starter': ['greedy_not_temperature_1_sampling', 'batched_inference', 'strict_final_channel_no_reasoning_fallback'],
    'phase2_accessed': False, 'submission_performed': False,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))


PROTOCOL_SHA256 = hashlib.sha256(canonical(PROTOCOL).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def within(path, *, existing=True):
    # Reject symlinks in the entire path, not only the final component.
    absolute = path.absolute()
    absolute.relative_to(ROOT.absolute())
    for part in (absolute, *absolute.parents):
        if part == ROOT.parent:
            break
        if part.is_symlink():
            raise ValueError('symlink_rejected')
    resolved = path.resolve(strict=existing)
    resolved.relative_to(ROOT.resolve(strict=True))
    return resolved


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(canonical(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path, row):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(canonical(row) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def validate_assets():
    data_manifest = within(DATA_ROOT / '.evomind/data-manifest.json')
    if sha256_file(data_manifest) != DATA_MANIFEST_SHA256:
        raise ValueError('data_manifest_drift')
    manifest_path = within(MODEL_ROOT / 'model-manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('repo'), manifest.get('revision'), manifest.get('status')) != (REPO, REVISION, 'completed'):
        raise ValueError('model_identity_invalid')
    files = manifest.get('files', [])
    if len(files) != 13 or {row['name'] for row in files} != MODEL_NAMES:
        raise ValueError('model_file_set_invalid')
    checked = []
    for row in files:
        path = within(MODEL_DIR / row['name'])
        size, digest = path.stat().st_size, sha256_file(path)
        if size != row['bytes'] or digest != row['sha256']:
            raise ValueError('model_file_integrity_invalid')
        if row['name'] in MODEL_FILES and (size, digest) != MODEL_FILES[row['name']]:
            raise ValueError('official_weight_integrity_invalid')
        checked.append({'name': row['name'], 'bytes': size, 'sha256': digest})
    return {'manifest_sha256': sha256_file(manifest_path), 'files': checked}


def load_validation():
    path = within(VALIDATION_PATH)
    if sha256_file(path) != VALIDATION_SHA256:
        raise ValueError('validation_drift')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 459 or len({str(row['id']) for row in rows}) != 459:
        raise ValueError('validation_coverage_invalid')
    if dict(Counter(row['question_type'] for row in rows)) != PROTOCOL['type_counts']:
        raise ValueError('validation_types_invalid')
    return rows


def prompt_for(row):
    # Match the pinned starter's dataset_utils + _get_prediction_with_trace.
    question = str(row['question'])
    if row['question_type'] == 'multi_choice':
        options = '\n'.join(f'{key}: {row["options"][key]}' for key in sorted(row['options']))
        return f'The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n{options}\n\nAnswer:'
    if row['question_type'] == 'open_ended':
        question = f'The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question}\n\nAnswer:'
    return f'The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question}\n\nAnswer:'


def mapping_prompt(row, answer):
    if row['question_type'] != 'open_ended_multi_choice' or not answer:
        raise ValueError('mapping_requires_completed_open_answer')
    options = '\n'.join(f'{key}: {row["options"][key]}' for key in sorted(row['options']))
    return ("The following is a multiple choice question about medicine and the agent's open-ended answer to the question. "
            "Convert the agent's answer to the final answer format using the corresponding option label, e.g., 'A', 'B', 'C', 'D', 'E' or 'None'. \n\n"
            f"Question: {row['question']}\n{options}\n\nAgent's answer: {answer.strip()}\n\nMulti-choice answer:")


def extract_choice(response):
    # Exact pinned starter behavior, including its permissive initial-letter rule.
    if not response:
        return ''
    response = response.strip().upper()
    if response and response[0] in 'ABCDE':
        return response[0]
    for pattern in (r'(?:answer is|answer:|is)\s*([ABCDE])', r'([ABCDE])\)', r'\b([ABCDE])\b'):
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return ''


def final_channel(raw):
    # Decode with special tokens retained. Never guess an answer from analysis.
    matches = list(re.finditer(r'(?:<\|channel\|>|<\|meta_sep\|>)final(?:<\|message\|>|<\|im_sep\|>)', raw))
    if not matches:
        return '', 'missing_final_channel'
    rest = raw[matches[-1].end():]
    stop = re.search(r'<\|(?:return|fim_suffix)\|>', rest)
    if not stop:
        return '', 'incomplete_final_channel'
    answer = rest[:stop.start()].strip()
    if re.search(r'<\|[^>]+\|>', answer):
        return '', 'unexpected_final_structure'
    return answer, 'completed' if answer else 'empty_final_channel'


def load_records(path, rows, stage):
    result = {}
    if not path.exists():
        return result
    expected = {str(row['id']): row['question_type'] for row in rows}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)  # A torn record fails closed; never silently skip it.
        key = row.get('id')
        if key not in expected or key in result:
            raise ValueError('checkpoint_id_invalid')
        if row.get('protocol_sha256') != PROTOCOL_SHA256 or row.get('stage') != stage or row.get('question_type') != expected[key]:
            raise ValueError('checkpoint_protocol_mismatch')
        result[key] = row
    return result


def bind_checkpoint(identity):
    target = CHECKPOINT_ROOT / 'identity.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != identity:
            raise ValueError('checkpoint_identity_drift')
    else:
        if any(CHECKPOINT_ROOT.glob('*.jsonl')):
            raise ValueError('checkpoint_identity_missing')
        write_json(target, identity)


def score_predictions(rows, predictions):
    expected = {str(row['id']) for row in rows}
    if set(predictions) != expected:
        raise ValueError('prediction_coverage_incomplete')
    counts = {}
    for question_type in PROTOCOL['type_counts']:
        subset = [row for row in rows if row['question_type'] == question_type]
        counts[question_type] = {'total': len(subset), 'valid_final': sum(predictions[str(row['id'])]['answer_status'] == 'completed' for row in subset)}
        if question_type != 'open_ended':
            counts[question_type]['correct'] = sum(
                predictions[str(row['id'])].get('choice') == str(row.get('correct_answer', row.get('answer', ''))).strip().upper()
                and predictions[str(row['id'])].get('choice') in 'ABCDE'
                and bool(predictions[str(row['id'])].get('choice')) for row in subset)
    total = counts['multi_choice']['total'] + counts['open_ended_multi_choice']['total']
    correct = counts['multi_choice']['correct'] + counts['open_ended_multi_choice']['correct']
    if total != PROTOCOL['accuracy_denominator']:
        raise ValueError('scoring_denominator_invalid')
    return {'schema': 'evomind.cure.public_metrics.v2', 'accuracy': correct / total,
            'correct': correct, 'scored_questions': total, 'counts': counts,
            'protocol_sha256': PROTOCOL_SHA256, 'official_hidden_score': None,
            'paper_comparison': PROTOCOL['paper_comparison'], 'phase2_accessed': False,
            'submission_performed': False, 'open_ended_score': None}


def run_batches(rows, output, deadline):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('single_cuda_device_required')
    torch.manual_seed(PROTOCOL['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 0}, quantization_config=Mxfp4Config(dequantize=True))
    model.eval()
    if {parameter.device.type for parameter in model.parameters()} != {'cuda'}:
        raise ValueError('cpu_offload_forbidden')
    write_json(output / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
        'transformers': __import__('transformers').__version__, 'gpu_name': torch.cuda.get_device_name(0),
        'allocated_bytes': torch.cuda.memory_allocated(), 'source_sha256': sha256_file(Path(__file__)),
        'protocol_sha256': PROTOCOL_SHA256, 'cpu_offload': False, 'network_model_fetch': False})
    answers_path = CHECKPOINT_ROOT / 'answers.jsonl'
    predictions_path = CHECKPOINT_ROOT / 'predictions.jsonl'
    answers = load_records(answers_path, rows, 'answer')
    predictions = load_records(predictions_path, rows, 'prediction')

    def generate(prompts):
        conversations = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
        encoded = tokenizer.apply_chat_template(conversations, add_generation_prompt=True,
            return_tensors='pt', return_dict=True, padding=True,
            reasoning_effort=PROTOCOL['reasoning_effort'],
            model_identity='You are ChatGPT, a large language model trained by OpenAI.')
        encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=PROTOCOL['max_new_tokens'],
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                max_time=max(1.0, deadline - time.monotonic() - 15))
        values = []
        for tokens in generated[:, encoded['input_ids'].shape[-1]:]:
            raw = tokenizer.decode(tokens, skip_special_tokens=False)
            answer, status = final_channel(raw)
            values.append({'final_answer': answer, 'answer_status': status,
                'response_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                'generated_tokens': len(tokens), 'raw_reasoning_saved': False})
        return values

    def progress():
        write_json(output / 'inference-progress.json', {'status': 'running', 'completed': len(predictions),
            'answered': len(answers), 'total': len(rows), 'protocol_sha256': PROTOCOL_SHA256, 'at_utc': utc_now()})

    progress()
    for start in range(0, len(rows), PROTOCOL['batch_size']):
        batch = [row for row in rows[start:start + PROTOCOL['batch_size']] if str(row['id']) not in predictions]
        if not batch:
            continue
        if deadline - time.monotonic() < 90:
            break
        fresh = [row for row in batch if str(row['id']) not in answers]
        if fresh:
            for row, generated in zip(fresh, generate([prompt_for(row) for row in fresh]), strict=True):
                if time.monotonic() >= deadline - 20 and generated['answer_status'] != 'completed':
                    continue  # Interrupted batch is resumable, not a scored model answer.
                record = {'id': str(row['id']), 'question_type': row['question_type'], 'stage': 'answer',
                          'protocol_sha256': PROTOCOL_SHA256, **generated}
                append_jsonl(answers_path, record)
                answers[record['id']] = record
        mapping_rows = [row for row in batch if row['question_type'] == 'open_ended_multi_choice'
                        and str(row['id']) in answers and answers[str(row['id'])]['answer_status'] == 'completed']
        if mapping_rows and deadline - time.monotonic() < 90:
            progress()
            break
        mapped = dict(zip((str(row['id']) for row in mapping_rows), generate([
            mapping_prompt(row, answers[str(row['id'])]['final_answer']) for row in mapping_rows]) if mapping_rows else [], strict=True))
        for row in batch:
            key = str(row['id'])
            if key not in answers:
                continue
            answer = answers[key]
            mapping = mapped.get(key)
            if mapping and time.monotonic() >= deadline - 20 and mapping['answer_status'] != 'completed':
                continue
            choice_source = mapping['final_answer'] if mapping else answer['final_answer']
            record = {**answer, 'stage': 'prediction',
                'choice': 'NOTAVALUE' if row['question_type'] == 'open_ended' else extract_choice(choice_source),
                'mapping': mapping, 'answer_status': mapping['answer_status'] if mapping else answer['answer_status']}
            append_jsonl(predictions_path, record)
            predictions[key] = record
        progress()
        append_jsonl(output / 'events.jsonl', {'event': 'batch_checkpointed', 'completed': len(predictions),
            'answered': len(answers), 'at_utc': utc_now(), 'protocol_sha256': PROTOCOL_SHA256})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if within(Path(args.data_dir)) != DATA_ROOT.resolve(strict=True):
        raise ValueError('data_root_invalid')
    output = within(Path(args.out_dir), existing=False)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_validation()
    if args.score_only:
        predictions = load_records(output / 'predictions.jsonl', rows, 'prediction')
        metrics = score_predictions(rows, predictions)
        write_json(output / 'independent-metrics.json', {**metrics, 'scorer_pid': os.getpid(),
            'predictions_sha256': sha256_file(output / 'predictions.jsonl'),
            'scorer_source_sha256': sha256_file(Path(__file__))})
        return 0
    import fcntl
    within(CHECKPOINT_ROOT, existing=False).mkdir(parents=True, exist_ok=True)
    with (CHECKPOINT_ROOT / '.run.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        identity = {'protocol': PROTOCOL, 'protocol_sha256': PROTOCOL_SHA256,
                    'source_sha256': sha256_file(Path(__file__)), 'model': validate_assets()}
        bind_checkpoint(identity)
        write_json(output / 'identity.json', identity)
        predictions = run_batches(rows, output, started + 780)
        ordered = [predictions[str(row['id'])] for row in rows if str(row['id']) in predictions]
        with (output / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
            handle.write(''.join(canonical(row) + '\n' for row in ordered))
        complete = len(predictions) == len(rows)
        result = {'status': 'INFERENCE_COMPLETE' if complete else 'PARTIAL', 'completed': len(predictions),
            'total': len(rows), 'protocol_sha256': PROTOCOL_SHA256, 'parent_run': os.environ.get('EVOMIND_RUN_ID'),
            'legacy_checkpoint_preserved': True, 'legacy_predictions_imported': 0,
            'phase2_accessed': False, 'submission_performed': False, 'inference_pid': os.getpid(),
            'elapsed_seconds': time.monotonic() - started, 'scientific_verdict': 'HOLD'}
        if complete:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--score-only',
                '--data-dir', str(DATA_ROOT), '--out-dir', str(output)], check=True, timeout=30)
            result['independent_metrics_sha256'] = sha256_file(output / 'independent-metrics.json')
        write_json(output / 'execution-result.json', result)
        write_json(output / 'inference-progress.json', result)
        print(canonical(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
