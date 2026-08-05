from __future__ import annotations

import inspect

from scripts import probe_siim_candidate_batch as probe


def test_probe_parser_requires_explicit_report_batch_and_torch_home(tmp_path):
    args = probe.build_parser().parse_args([
        "--report",
        str(tmp_path / "probe.json"),
        "--batch-size",
        "8",
        "--torch-home",
        str(tmp_path / "torch"),
    ])
    assert args.batch_size == 8
    assert args.image_size == 384
    assert args.full_backbone == "convnext_small"
    assert args.lesion_backbone == "efficientnet_v2_s"
    assert args.expected_gpu_name == "A800"
    assert args.memory_limit_mib == 55 * 1024
    assert args.effective_batch_size == 384


def test_probe_parser_enforces_fixed_effective_batch(tmp_path):
    parser = probe.build_parser()
    args = parser.parse_args([
        "--report",
        str(tmp_path / "probe.json"),
        "--batch-size",
        "96",
        "--torch-home",
        str(tmp_path / "torch"),
    ])
    assert args.effective_batch_size // args.batch_size == 4
    assert "set_per_process_memory_fraction" in inspect.getsource(probe.run_probe)


def test_probe_contract_contains_no_grader_submission_or_process_signal_path():
    source = inspect.getsource(probe)
    assert '"official_grader_executed": False' in source
    assert '"kaggle_submission_executed": False' in source
    assert '"process_signals_sent": 0' in source
    assert "private_grade(" not in source
    assert "terminate(" not in source
    assert ".kill(" not in source
