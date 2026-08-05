from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import watch_siim_narration_audio as watcher


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    narration = tmp_path / "narration.json"
    audio = tmp_path / "audio"
    builder = tmp_path / "builder.py"
    builder.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setattr(watcher, "NARRATION_PATH", narration)
    monkeypatch.setattr(watcher, "AUDIO_ROOT", audio)
    monkeypatch.setattr(watcher, "BUILDER", builder)
    return narration, audio, builder


def test_build_audio_requires_same_ready_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    narration, _audio, _builder = configure(monkeypatch, tmp_path)
    write_json(
        narration,
        {
            "run_id": "another_run",
            "status": "ready",
            "duration_seconds": 92.0,
            "voice_count": 1,
            "background_music": False,
        },
    )
    with pytest.raises(watcher.NarrationAudioWatchError):
        watcher.build_audio()


def test_build_audio_verifies_all_outputs_and_qa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    narration, audio, _builder = configure(monkeypatch, tmp_path)
    write_json(
        narration,
        {
            "run_id": watcher.RUN_ID,
            "status": "ready",
            "duration_seconds": 92.0,
            "voice_count": 1,
            "background_music": False,
        },
    )

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        audio.mkdir(parents=True, exist_ok=True)
        for name in (
            "narration-user-first-48k.wav",
            "narration-user-first-48k.m4a",
            "subtitles-user-first.srt",
            "narration-build-manifest.json",
        ):
            (audio / name).write_bytes(b"fixture")
        write_json(audio / "narration-audio-qa.json", {"status": "passed"})
        return SimpleNamespace(stdout="ready")

    monkeypatch.setattr(watcher.subprocess, "run", fake_run)
    result = watcher.build_audio()
    assert set(result["outputs"]) == {"wav", "aac", "subtitles", "manifest", "qa"}
    assert result["builder_stdout_tail"] == "ready"
