#!/usr/bin/env python3
"""Build and verify the editable JianYing project for the EvoMind 0.3.0 demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "video-production" / "evomind-product-demo-0.3.0"
EDL_PATH = PROJECT_ROOT / "project" / "edit-decision-list.json"
BUILD_MANIFEST_PATH = PROJECT_ROOT / "project" / "build-manifest.json"
SRT_PATH = PROJECT_ROOT / "project" / "subtitles.zh-CN.srt"
NARRATION_PATH = PROJECT_ROOT / "assets" / "narration-timeline.wav"
BGM_PATH = PROJECT_ROOT / "assets" / "evomind-ambient-bed.wav"
FINAL_MIX_PATH = PROJECT_ROOT / "assets" / "final-mix.wav"
EVIDENCE_PARENT = PROJECT_ROOT / "project"
EVIDENCE_COPY = EVIDENCE_PARENT / "jianying-editable-0.3.0"
PROJECT_NAME = "EvoMind_Product_Demo_zh_CN_0.3.0_EDITABLE"
OWNER_SCHEMA = "evomind.jianying_project_owner.v1"
MANIFEST_SCHEMA = "evomind.jianying_project.v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seconds_to_us(value: float) -> int:
    return round(float(value) * 1_000_000)


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("required video project assets are missing: " + "; ".join(missing))


def install_skill_runtime() -> tuple[type, Any]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_root = os.getenv("JY_SKILL_ROOT", "").strip()
    skill_candidates = [
        env_root,
        os.path.join(current_dir, ".agent", "skills", "jianying-editor"),
        os.path.join(current_dir, ".trae", "skills", "jianying-editor"),
        os.path.join(current_dir, ".claude", "skills", "jianying-editor"),
        os.path.join(current_dir, "skills", "jianying-editor"),
        os.path.abspath(".agent/skills/jianying-editor"),
        os.path.dirname(current_dir),
    ]
    scripts_path = None
    attempted: list[str] = []
    for candidate in skill_candidates:
        if not candidate:
            continue
        resolved = os.path.abspath(candidate)
        attempted.append(resolved)
        if os.path.exists(os.path.join(resolved, "scripts", "jy_wrapper.py")):
            scripts_path = os.path.join(resolved, "scripts")
            break
    if not scripts_path:
        raise ImportError(
            "Could not find jianying-editor/scripts/jy_wrapper.py\nTried:\n- "
            + "\n- ".join(attempted)
        )
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from jy_wrapper import JyProject  # type: ignore[import-not-found]
    import pyJianYingDraft as draft  # type: ignore[import-not-found]

    return JyProject, draft


def replace_owned_evidence(source: Path, destination: Path) -> None:
    parent = EVIDENCE_PARENT.resolve()
    target = destination.resolve()
    if target.parent != parent or target.name != "jianying-editable-0.3.0":
        raise RuntimeError(f"refusing to replace unowned JianYing evidence path: {target}")
    if target.exists():
        owner = target / ".evomind-project-owner.json"
        if not owner.is_file():
            raise RuntimeError(f"existing JianYing evidence has no owner sentinel: {target}")
        payload = load_json(owner)
        if payload != {"schema": OWNER_SCHEMA, "project_name": PROJECT_NAME}:
            raise RuntimeError(f"existing JianYing evidence owner sentinel is invalid: {target}")
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=False)
    (target / ".evomind-project-owner.json").write_text(
        json.dumps({"schema": OWNER_SCHEMA, "project_name": PROJECT_NAME}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def track_summary(draft_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(track.get("name")): {
            "type": track.get("type"),
            "segments": len(track.get("segments") or []),
            "segment_ids": [segment.get("id") for segment in track.get("segments") or []],
        }
        for track in draft_payload.get("tracks") or []
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts-root", type=Path, help="override JianYing drafts root")
    parser.add_argument("--json", action="store_true", help="emit one machine-readable result")
    args = parser.parse_args()

    require_files([EDL_PATH, BUILD_MANIFEST_PATH, SRT_PATH, NARRATION_PATH, BGM_PATH, FINAL_MIX_PATH])
    edl = load_json(EDL_PATH)
    build_manifest = load_json(BUILD_MANIFEST_PATH)
    source = ROOT / build_manifest["source"]
    final_video = ROOT / build_manifest["output"]
    require_files([source, final_video])
    if sha256_file(source) != build_manifest["source_sha256"]:
        raise RuntimeError("source recording hash drift")
    if sha256_file(final_video) != build_manifest["output_sha256"]:
        raise RuntimeError("final demo hash drift")

    JyProject, draft = install_skill_runtime()
    drafts_root = str(args.drafts_root.resolve()) if args.drafts_root else None
    project = JyProject(PROJECT_NAME, width=1920, height=1080, drafts_root=drafts_root, overwrite=True)

    reference = project.add_media_safe(str(final_video.resolve()), start_time="0s", track_name="FinalReference")
    if reference is None:
        raise RuntimeError("failed to add final reference video")
    reference.volume = 0.0

    # Parse the long MKV once and reuse one material object for every edit.
    # Reconstructing it for each segment triggers a sporadic MediaInfo fallback
    # in the wrapper; that fallback derives material duration from the requested
    # clip length and makes later source offsets appear out of range.
    source_duration_us = seconds_to_us(edl["source_duration_seconds"])
    source_material = draft.VideoMaterial(str(source.resolve()), duration=source_duration_us)
    if abs(int(source_material.duration) - source_duration_us) > 1_000:
        raise RuntimeError(
            f"JianYing source duration drift: {source_material.duration} != {source_duration_us}"
        )
    project.script.add_track(draft.TrackType.video, "EditableSource")

    source_segments = []
    for row in edl["segments"]:
        source_start_us = seconds_to_us(row["source_in"])
        clip_source_duration_us = seconds_to_us(row["source_out"] - row["source_in"])
        target_start_us = seconds_to_us(row["output_in"])
        target_duration_us = seconds_to_us(row["output_out"] - row["output_in"])
        segment = draft.VideoSegment(
            source_material,
            draft.Timerange(target_start_us, target_duration_us),
            source_timerange=draft.Timerange(source_start_us, clip_source_duration_us),
            speed=float(row["speed"]),
            volume=0.0,
        )
        segment.clip_settings.alpha = 0.0
        project.script.add_segment(segment, "EditableSource")
        source_segments.append(segment)

    final_mix = project.add_audio_safe(str(FINAL_MIX_PATH.resolve()), start_time="0s", track_name="FinalMix")
    narration = project.add_audio_safe(str(NARRATION_PATH.resolve()), start_time="0s", track_name="NarrationStem")
    bgm = project.add_audio_safe(str(BGM_PATH.resolve()), start_time="0s", track_name="BGMStem")
    if final_mix is None or narration is None or bgm is None:
        raise RuntimeError("failed to add one or more audio assets")
    final_mix.volume = 1.0
    narration.volume = 0.0
    bgm.volume = 0.0

    project.script.import_srt(str(SRT_PATH.resolve()), "Subtitles")
    subtitle_track = project.script.tracks.get("Subtitles")
    if subtitle_track is None or not subtitle_track.segments:
        raise RuntimeError("SRT import produced no subtitle segments")
    for segment in subtitle_track.segments:
        segment.clip_settings.alpha = 0.0

    save_result = project.save()
    draft_path = Path(save_result["draft_path"]).resolve()
    draft_info = draft_path / "draft_info.json"
    draft_meta = draft_path / "draft_meta_info.json"
    require_files([draft_info, draft_meta])

    draft_payload = load_json(draft_info)
    tracks = track_summary(draft_payload)
    expected_counts = {
        "FinalReference": 1,
        "EditableSource": len(edl["segments"]),
        "FinalMix": 1,
        "NarrationStem": 1,
        "BGMStem": 1,
        "Subtitles": 14,
    }
    failures = [
        f"{name}: expected {count}, found {tracks.get(name, {}).get('segments')}"
        for name, count in expected_counts.items()
        if tracks.get(name, {}).get("segments") != count
    ]
    actual_speeds = [round(float(segment.speed.speed), 6) for segment in source_segments]
    expected_speeds = [round(float(row["speed"]), 6) for row in edl["segments"]]
    if actual_speeds != expected_speeds:
        failures.append("editable source speed sequence drift")
    if failures:
        raise RuntimeError("JianYing project validation failed: " + "; ".join(failures))

    replace_owned_evidence(draft_path, EVIDENCE_COPY)
    copied_info = EVIDENCE_COPY / "draft_info.json"
    copied_meta = EVIDENCE_COPY / "draft_meta_info.json"
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": generated_at,
        "status": "passed",
        "project_name": PROJECT_NAME,
        "live_draft_path": str(draft_path),
        "evidence_copy": str(EVIDENCE_COPY.relative_to(ROOT)).replace("\\", "/"),
        "canvas": {"width": 1920, "height": 1080},
        "duration_seconds": float(edl["segments"][-1]["output_out"]),
        "tracks": tracks,
        "source_segments": len(source_segments),
        "source_speed_sequence": actual_speeds,
        "subtitle_segments": tracks["Subtitles"]["segments"],
        "assets": {
            "source": {"path": str(source.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(source)},
            "final_video": {"path": str(final_video.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(final_video)},
            "srt": {"path": str(SRT_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(SRT_PATH)},
            "final_mix": {"path": str(FINAL_MIX_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(FINAL_MIX_PATH)},
            "narration": {"path": str(NARRATION_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(NARRATION_PATH)},
            "bgm": {"path": str(BGM_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(BGM_PATH)},
        },
        "draft_files": {
            "draft_info.json": {"bytes": copied_info.stat().st_size, "sha256": sha256_file(copied_info)},
            "draft_meta_info.json": {"bytes": copied_meta.stat().st_size, "sha256": sha256_file(copied_meta)},
        },
        "acceptance": {
            "draft_folder_exists": EVIDENCE_COPY.is_dir(),
            "project_save_succeeded": save_result.get("status") == "SUCCESS",
            "video_track_present": tracks["FinalReference"]["segments"] > 0,
            "editable_source_timeline_present": tracks["EditableSource"]["segments"] == 9,
            "bgm_on_audio_track": tracks["BGMStem"]["type"] == "audio",
            "narration_and_subtitles_present": tracks["NarrationStem"]["segments"] == 1 and tracks["Subtitles"]["segments"] == 14,
        },
    }
    manifest["passed"] = all(manifest["acceptance"].values())
    manifest_path = EVIDENCE_COPY / "evomind-jianying-project-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "ok": manifest["passed"],
        "status": manifest["status"],
        "project_name": PROJECT_NAME,
        "live_draft_path": str(draft_path),
        "evidence_copy": str(EVIDENCE_COPY),
        "manifest": str(manifest_path),
        "tracks": {name: value["segments"] for name, value in tracks.items()},
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
