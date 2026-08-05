#!/usr/bin/env python3
"""Build the EvoMind 0.3.0 product demo from the frozen real Chrome capture.

The script is deliberately deterministic after the narration assets have been
materialized: it verifies the source hash, analyzes the source, builds a fixed
EDL, mixes one Mandarin voice with a generated ambient bed, burns the frozen
SRT, and renders H.264/AAC with FFmpeg.  It never launches or controls Chrome,
OBS, EvoMind, a gateway, or a training process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "video-production" / "evomind-product-demo-0.3.0"
SOURCE = PROJECT / "sources" / "real-chrome-evolution-demo-final-20260803.mkv"
EDL_PATH = PROJECT / "project" / "edit-decision-list.json"
SRT_PATH = PROJECT / "project" / "subtitles.zh-CN.srt"
ASSETS = PROJECT / "assets"
ANALYSIS = PROJECT / "analysis"
OUTPUT = PROJECT / "output" / "EvoMind-Product-Demo-zh-CN-0.3.0.mp4"
EXPECTED_SOURCE_SHA256 = (
    "2aec9b0812e0297c7d63f758ec351deb2f9bc0810229fa03dffa13c52f6c2b37"
)
EDGE_TTS_CANDIDATES = (
    Path(os.environ.get("EDGE_TTS_EXE", "")),
    Path(r"E:\EvoMind-release-validation\video-tooling\.venv\Scripts\edge-tts.exe"),
)
FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def display_command(args: Iterable[str]) -> str:
    rendered: list[str] = []
    root_text = str(ROOT)
    for item in args:
        value = str(item).replace(root_text, "${WORKSPACE}")
        rendered.append(f'"{value}"' if any(ch.isspace() for ch in value) else value)
    return " ".join(rendered)


def run(
    args: list[str],
    *,
    capture: bool = False,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print(f"[run] {display_command(args)}", flush=True)
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if check and completed.returncode:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {display_command(args)}\n{message}"
        )
    return completed


def ffprobe(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            "--",
            rel(path),
        ],
        capture=True,
    )
    return scrub_workspace_paths(json.loads(result.stdout))


def scrub_workspace_paths(value: Any) -> Any:
    """Replace local workspace paths before reports enter the evidence pack."""
    if isinstance(value, dict):
        return {key: scrub_workspace_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_workspace_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "${WORKSPACE}").replace(
            str(ROOT).replace("\\", "/"), "${WORKSPACE}"
        )
    return value


def duration(path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "--",
            rel(path),
        ],
        capture=True,
        quiet=True,
    )
    return float(result.stdout.strip())


def parse_timestamp(value: str) -> float:
    hours, minutes, seconds_ms = value.split(":")
    seconds, milliseconds = seconds_ms.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


def parse_srt(path: Path) -> list[Cue]:
    blocks = re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8").strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError(f"Invalid SRT block: {block!r}")
        index = int(lines[0].strip())
        start_text, end_text = [part.strip() for part in lines[1].split("-->")]
        text = " ".join(line.strip() for line in lines[2:] if line.strip())
        cues.append(Cue(index, parse_timestamp(start_text), parse_timestamp(end_text), text))
    return cues


def detector_values(log: str, name: str) -> list[float]:
    return [float(value) for value in re.findall(rf"{re.escape(name)}:\s*(-?\d+(?:\.\d+)?)", log)]


def analyze_source(force: bool) -> dict[str, Any]:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    probe_path = ANALYSIS / "source-ffprobe.json"
    hash_path = ANALYSIS / "source-sha256.json"
    detector_log_path = ANALYSIS / "source-black-freeze-silence.log"
    report_path = ANALYSIS / "source-media-analysis.json"
    contact_path = ANALYSIS / "source-contact-sheet-10s.jpg"

    probe = ffprobe(SOURCE)
    probe_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hash_payload = {
        "schema": "evomind.media_hash.v1",
        "file": rel(SOURCE),
        "sha256": sha256(SOURCE),
        "bytes": SOURCE.stat().st_size,
    }
    hash_path.write_text(
        json.dumps(hash_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if force or not detector_log_path.exists():
        detected = run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                rel(SOURCE),
                "-vf",
                "blackdetect=d=0.20:pix_th=0.10,freezedetect=n=-50dB:d=4",
                "-af",
                "silencedetect=n=-45dB:d=1.5",
                "-f",
                "null",
                "NUL",
            ],
            capture=True,
        )
        detector_log_path.write_text(
            (detected.stderr or detected.stdout or "") + "\n", encoding="utf-8"
        )
    detector_log = detector_log_path.read_text(encoding="utf-8")

    if force or not contact_path.exists():
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                rel(SOURCE),
                "-vf",
                (
                    "fps=1/10,scale=640:360,"
                    "drawtext=fontfile='C\\:/Windows/Fonts/msyh.ttc':"
                    "text='%{pts\\:hms}':x=12:y=12:fontsize=24:fontcolor=white:"
                    "box=1:boxcolor=black@0.75,tile=5x6:padding=4:margin=4"
                ),
                "-frames:v",
                "1",
                "-update",
                "1",
                "-q:v",
                "2",
                rel(contact_path),
            ]
        )

    freeze_starts = detector_values(detector_log, "lavfi.freezedetect.freeze_start")
    freeze_durations = detector_values(detector_log, "lavfi.freezedetect.freeze_duration")
    freeze_ends = detector_values(detector_log, "lavfi.freezedetect.freeze_end")
    report = {
        "schema": "evomind.source_media_analysis.v1",
        "source": rel(SOURCE),
        "source_sha256": hash_payload["sha256"],
        "duration_seconds": float(probe["format"]["duration"]),
        "black_intervals": [
            {"start": start, "end": end, "duration": duration_value}
            for start, end, duration_value in zip(
                detector_values(detector_log, "black_start"),
                detector_values(detector_log, "black_end"),
                detector_values(detector_log, "black_duration"),
            )
        ],
        "freeze_intervals": [
            {"start": start, "end": end, "duration": duration_value}
            for start, end, duration_value in zip(
                freeze_starts, freeze_ends, freeze_durations
            )
        ],
        "silence_intervals": [
            {"start": start, "end": end, "duration": duration_value}
            for start, end, duration_value in zip(
                detector_values(detector_log, "silence_start"),
                detector_values(detector_log, "silence_end"),
                detector_values(detector_log, "silence_duration"),
            )
        ],
        "observed_state_change_boundaries_seconds": [
            20.454,
            44.888,
            66.921,
            89.354,
            200.721,
            221.254,
            243.888,
            270.754,
        ],
        "contact_sheet": rel(contact_path),
        "interpretation": (
            "The source audio is intentionally silent and the long freezes are stable UI "
            "states between real user actions. The EDL preserves chronology, labels every "
            "accelerated interval, and adds narration/music plus a subtle source-only camera move."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def edge_tts_executable() -> Path | None:
    for candidate in EDGE_TTS_CANDIDATES:
        if str(candidate) and candidate.is_file():
            return candidate
    located = shutil.which("edge-tts")
    return Path(located) if located else None


def encode_powershell(script: str) -> str:
    import base64

    return base64.b64encode(script.encode("utf-16le")).decode("ascii")


def generate_sapi(text_path: Path, wav_path: Path) -> None:
    text_literal = str(text_path).replace("'", "''")
    wav_literal = str(wav_path).replace("'", "''")
    script = f"""
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$text=Get-Content -LiteralPath '{text_literal}' -Raw -Encoding UTF8
$s=[System.Speech.Synthesis.SpeechSynthesizer]::new()
$s.SelectVoice('Microsoft Huihui Desktop')
$s.Rate=0
$s.Volume=100
$s.SetOutputToWaveFile('{wav_literal}')
$s.Speak($text)
$s.Dispose()
"""
    run(["pwsh", "-NoProfile", "-EncodedCommand", encode_powershell(script)])


def tempo_chain(ratio: float) -> str:
    if ratio <= 1.0005:
        return "anull"
    filters: list[str] = []
    while ratio > 2.0:
        filters.append("atempo=2.0")
        ratio /= 2.0
    filters.append(f"atempo={ratio:.8f}")
    return ",".join(filters)


def generate_narration_segments(cues: list[Cue], force: bool) -> list[Path]:
    segment_dir = ASSETS / "narration-segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    edge = edge_tts_executable()
    outputs: list[Path] = []
    for cue in cues:
        stem = f"{cue.index:02d}"
        text_path = segment_dir / f"{stem}.txt"
        source_audio = segment_dir / f"{stem}.mp3"
        sapi_audio = segment_dir / f"{stem}-sapi.wav"
        processed = segment_dir / f"{stem}.wav"
        text_path.write_text(cue.text + "\n", encoding="utf-8")
        if force:
            for candidate in (source_audio, sapi_audio, processed):
                if candidate.exists():
                    candidate.unlink()
        if not processed.exists():
            generated: Path
            if edge is not None:
                edge_result = run(
                    [
                        str(edge),
                        "--voice",
                        "zh-CN-XiaoxiaoNeural",
                        "--rate=+8%",
                        "--file",
                        str(text_path),
                        "--write-media",
                        str(source_audio),
                    ],
                    capture=True,
                    check=False,
                )
                generated = source_audio
                if edge_result.returncode or not source_audio.exists():
                    print("[warn] Edge TTS unavailable; using local Microsoft Huihui.")
                    generate_sapi(text_path, sapi_audio)
                    generated = sapi_audio
            else:
                generate_sapi(text_path, sapi_audio)
                generated = sapi_audio
            source_duration = duration(generated)
            available = max(0.5, cue.end - cue.start - 0.12)
            speed_ratio = max(1.0, source_duration / available)
            run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    rel(generated),
                    "-af",
                    f"{tempo_chain(speed_ratio)},aresample=48000",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s24le",
                    rel(processed),
                ]
            )
        outputs.append(processed)
    return outputs


def loudnorm_measure(path: Path, integrated: float, peak: float, lra: float) -> dict[str, Any]:
    measured = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            rel(path),
            "-af",
            f"loudnorm=I={integrated}:TP={peak}:LRA={lra}:print_format=json",
            "-f",
            "null",
            "NUL",
        ],
        capture=True,
    )
    log = measured.stderr or measured.stdout or ""
    objects = re.findall(r"\{\s*\"input_i\".*?\}", log, flags=re.DOTALL)
    if not objects:
        raise RuntimeError(f"Unable to parse loudnorm output for {path}:\n{log}")
    return json.loads(objects[-1])


def loudnorm_two_pass(
    source: Path,
    target: Path,
    *,
    integrated: float,
    peak: float,
    lra: float,
) -> dict[str, Any]:
    measurement = loudnorm_measure(source, integrated, peak, lra)
    filter_text = (
        f"loudnorm=I={integrated}:TP={peak}:LRA={lra}:"
        f"measured_I={measurement['input_i']}:"
        f"measured_TP={measurement['input_tp']}:"
        f"measured_LRA={measurement['input_lra']}:"
        f"measured_thresh={measurement['input_thresh']}:"
        f"offset={measurement['target_offset']}:linear=true:print_format=summary"
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            rel(source),
            "-af",
            filter_text,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s24le",
            rel(target),
        ]
    )
    return measurement


def build_audio(edl: dict[str, Any], force: bool) -> dict[str, Any]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cues = parse_srt(SRT_PATH)
    total_duration = float(edl["segments"][-1]["output_out"])
    narration_segments = generate_narration_segments(cues, force)
    narration_raw = ASSETS / "narration-timeline-raw.wav"
    narration = ASSETS / "narration-timeline.wav"
    bgm_raw = ASSETS / "evomind-ambient-bed-raw.wav"
    bgm = ASSETS / "evomind-ambient-bed.wav"
    mix_raw = ASSETS / "final-mix-raw.wav"
    final_mix = ASSETS / "final-mix.wav"

    if force or not narration_raw.exists():
        args = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=48000:cl=stereo:d={total_duration:.6f}",
        ]
        for path in narration_segments:
            args.extend(["-i", rel(path)])
        filters = [f"[0:a]atrim=0:{total_duration:.6f},asetpts=PTS-STARTPTS[base]"]
        labels = ["[base]"]
        for input_index, (cue, _path) in enumerate(zip(cues, narration_segments), start=1):
            label = f"voice{input_index}"
            delay_ms = int(round(cue.start * 1000))
            filters.append(
                f"[{input_index}:a]adelay=delays={delay_ms}:all=1,"
                f"apad,atrim=0:{total_duration:.6f}[{label}]"
            )
            labels.append(f"[{label}]")
        filters.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:duration=first:normalize=0:dropout_transition=0,"
            + "alimiter=limit=0.95[narr]"
        )
        args.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[narr]",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s24le",
                rel(narration_raw),
            ]
        )
        run(args)
    if force or not narration.exists():
        loudnorm_two_pass(narration_raw, narration, integrated=-16, peak=-1.5, lra=7)

    if force or not bgm_raw.exists():
        expression = (
            "0.055*sin(2*PI*110*t)+0.028*sin(2*PI*164.81*t)+"
            "0.020*sin(2*PI*220*t)+0.012*sin(2*PI*329.63*t)"
        )
        expression_right = (
            "0.052*sin(2*PI*110*t+0.35)+0.027*sin(2*PI*164.81*t+0.2)+"
            "0.019*sin(2*PI*220*t+0.5)+0.011*sin(2*PI*329.63*t+0.1)"
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"aevalsrc={expression}|{expression_right}:s=48000:d={total_duration:.6f}",
                "-af",
                (
                    f"highpass=f=55,lowpass=f=1800,"
                    f"afade=t=in:st=0:d=1.5,"
                    f"afade=t=out:st={max(0.0, total_duration - 1.8):.6f}:d=1.8"
                ),
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s24le",
                rel(bgm_raw),
            ]
        )
    if force or not bgm.exists():
        loudnorm_two_pass(bgm_raw, bgm, integrated=-32, peak=-6, lra=7)

    if force or not mix_raw.exists():
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                rel(narration),
                "-i",
                rel(bgm),
                "-filter_complex",
                (
                    f"[0:a]atrim=0:{total_duration:.6f},asetpts=PTS-STARTPTS[n];"
                    f"[1:a]atrim=0:{total_duration:.6f},asetpts=PTS-STARTPTS[b];"
                    "[n][b]amix=inputs=2:duration=longest:normalize=0:dropout_transition=0[mix]"
                ),
                "-map",
                "[mix]",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s24le",
                rel(mix_raw),
            ]
        )
    final_measurement: dict[str, Any] = {}
    if force or not final_mix.exists():
        final_measurement = loudnorm_two_pass(
            mix_raw, final_mix, integrated=-14, peak=-1, lra=7
        )
    return {
        "narration": rel(narration),
        "background_music": rel(bgm),
        "final_mix": rel(final_mix),
        "duration_seconds": duration(final_mix),
        "final_loudnorm_first_pass": final_measurement,
        "narration_segment_hashes": {
            rel(path): sha256(path) for path in narration_segments
        },
    }


def escape_filter_path(path: Path) -> str:
    return rel(path).replace("'", "\\'").replace(":", "\\:")


def build_video(edl: dict[str, Any], force: bool) -> None:
    if OUTPUT.exists() and not force:
        print(f"[skip] {rel(OUTPUT)} already exists")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    header_dir = ASSETS / "headers"
    header_dir.mkdir(parents=True, exist_ok=True)
    brand_path = header_dir / "brand.txt"
    brand_path.write_text("EVOMIND · 真实系统", encoding="utf-8")
    footer_path = header_dir / "footer.txt"
    footer_path.write_text("本地可复核证据 · 非官方竞赛成绩", encoding="utf-8")

    segments = edl["segments"]
    total_duration = float(segments[-1]["output_out"])
    split_outputs = "".join(f"[src{i}]" for i in range(len(segments)))
    filters = [f"[0:v]split={len(segments)}{split_outputs}"]
    concat_inputs: list[str] = []
    font_regular = str(FONT_REGULAR).replace("\\", "/").replace(":", "\\:")
    font_bold = str(FONT_BOLD).replace("\\", "/").replace(":", "\\:")

    for index, segment in enumerate(segments):
        title_path = header_dir / f"{segment['id']}-title.txt"
        speed_path = header_dir / f"{segment['id']}-speed.txt"
        title_path.write_text(segment["chapter"], encoding="utf-8")
        speed_path.write_text(segment["speed_label"], encoding="utf-8")
        out_label = f"seg{index}"
        filters.append(
            f"[src{index}]"
            f"trim=start={float(segment['source_in']):.6f}:end={float(segment['source_out']):.6f},"
            f"setpts=(PTS-STARTPTS)/{float(segment['speed']):.8f},"
            "fps=30,crop=1920:870:0:210,"
            "zoompan=z='1.012+0.008*sin(on/75)':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            "d=1:s=1920x870:fps=30,setsar=1,"
            "pad=1920:1080:0:105:color=0x030806,"
            "drawbox=x=0:y=0:w=1920:h=105:color=0x07110e:t=fill,"
            "drawbox=x=0:y=975:w=1920:h=105:color=0x07110e:t=fill,"
            f"drawtext=fontfile='{font_bold}':textfile='{escape_filter_path(brand_path)}':"
            "x=36:y=24:fontsize=34:fontcolor=0xE8FFF7,"
            f"drawtext=fontfile='{font_bold}':textfile='{escape_filter_path(title_path)}':"
            "x=430:y=25:fontsize=32:fontcolor=0x66E3C4,"
            f"drawtext=fontfile='{font_regular}':textfile='{escape_filter_path(speed_path)}':"
            "x=w-tw-38:y=18:fontsize=25:fontcolor=0x9FF7DC,"
            f"drawtext=fontfile='{font_regular}':textfile='{escape_filter_path(footer_path)}':"
            f"x=36:y=1017:fontsize=21:fontcolor=0x8AA49B[{out_label}]"
        )
        concat_inputs.append(f"[{out_label}]")

    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(segments)}:v=1:a=0[vcat]"
    )
    filters.append(
        f"[vcat]drawtext=fontfile='{font_regular}':"
        "text='REAL CHROME  %{pts\\:hms}':x=w-tw-38:y=56:"
        "fontsize=22:fontcolor=0xB6CAC4,"
        f"subtitles=filename='{escape_filter_path(SRT_PATH)}':"
        "fontsdir='C\\:/Windows/Fonts':"
        "force_style='FontName=Microsoft YaHei,FontSize=25,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00101916,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,"
        "MarginV=20'[vout]"
    )

    filter_script = PROJECT / "project" / "render-filter.ffscript"
    filter_script.write_text(";\n".join(filters) + "\n", encoding="utf-8")
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            rel(SOURCE),
            "-i",
            rel(ASSETS / "final-mix.wav"),
            "-filter_complex_script",
            rel(filter_script),
            "-map",
            "[vout]",
            "-map",
            "1:a:0",
            "-t",
            f"{total_duration:.6f}",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-profile:v",
            "high",
            "-level:v",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            rel(OUTPUT),
        ]
    )


def write_build_manifest(
    edl: dict[str, Any], source_analysis: dict[str, Any], audio: dict[str, Any]
) -> None:
    probe = ffprobe(OUTPUT)
    manifest = {
        "schema": "evomind.product_demo.build.v1",
        "version": "0.3.0",
        "source": rel(SOURCE),
        "source_sha256": sha256(SOURCE),
        "edl": rel(EDL_PATH),
        "edl_sha256": sha256(EDL_PATH),
        "subtitles": rel(SRT_PATH),
        "subtitles_sha256": sha256(SRT_PATH),
        "output": rel(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "output_bytes": OUTPUT.stat().st_size,
        "output_duration_seconds": float(probe["format"]["duration"]),
        "output_probe": probe,
        "source_analysis": source_analysis,
        "audio": audio,
        "real_product_footage_ratio": edl["visual_contract"][
            "real_product_footage_ratio"
        ],
        "mock_or_synthetic_ui": False,
        "processes_started_or_signaled": [],
    }
    path = PROJECT / "project" / "build-manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    delivery_srt = OUTPUT.with_suffix(".srt")
    shutil.copy2(SRT_PATH, delivery_srt)
    hashes = PROJECT / "project" / "SHA256SUMS.txt"
    tracked = [SOURCE, EDL_PATH, SRT_PATH, delivery_srt, ASSETS / "narration-timeline.wav", ASSETS / "evomind-ambient-bed.wav", ASSETS / "final-mix.wav", OUTPUT]
    hashes.write_text(
        "".join(f"{sha256(path)}  {rel(path)}\n" for path in tracked), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate audio and video")
    parser.add_argument(
        "--skip-source-analysis", action="store_true", help="Reuse existing source analysis"
    )
    args = parser.parse_args()

    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required command missing: {command}")
    for path in (SOURCE, EDL_PATH, SRT_PATH, FONT_REGULAR, FONT_BOLD):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_hash = sha256(SOURCE)
    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Source hash mismatch: expected {EXPECTED_SOURCE_SHA256}, got {actual_hash}"
        )

    edl = json.loads(EDL_PATH.read_text(encoding="utf-8"))
    source_analysis = (
        json.loads((ANALYSIS / "source-media-analysis.json").read_text(encoding="utf-8"))
        if args.skip_source_analysis
        else analyze_source(args.force)
    )
    audio = build_audio(edl, args.force)
    build_video(edl, args.force)
    write_build_manifest(edl, source_analysis, audio)
    print(
        json.dumps(
            {
                "ok": True,
                "output": rel(OUTPUT),
                "sha256": sha256(OUTPUT),
                "duration_seconds": duration(OUTPUT),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
