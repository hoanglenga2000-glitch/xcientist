from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
from pathlib import Path

import edge_tts

VIDEO_ROOT = Path(r"E:\EvoMind-release-validation\evomind-commercial-video-v2-20260722-165206")
FFMPEG = Path(r"D:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe")
FFPROBE = Path(r"D:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe")
VOICE = "zh-CN-XiaoxiaoNeural"
RATE = "-3%"
PITCH = "+0Hz"
TOTAL_SECONDS = 208.0
TEAL_ASS = "&H00BFD42D&"
WHITE_ASS = "&H00FFFFFF&"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def duration(path: Path) -> float:
    result = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def ass_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    for keyword in ("EvoMind", "Human Gate", "Independent Reviewer", "Nature Skills", "远程 GPU"):
        escaped = escaped.replace(keyword, f"{{\\c{TEAL_ASS}}}{keyword}{{\\c{WHITE_ASS}}}")
    return escaped


def reused_voice_segments() -> dict[str, Path]:
    manifest = VIDEO_ROOT / "audio" / "narration-manifest-v4.json"
    if not manifest.is_file():
        return {}
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        str(item["text"]): Path(str(item["path"]))
        for item in entries
        if isinstance(item, dict) and Path(str(item.get("path", ""))).is_file()
    }


async def render_voice() -> list[dict[str, object]]:
    segments = json.loads((VIDEO_ROOT / "narration-v5.json").read_text(encoding="utf-8"))
    cache = reused_voice_segments()
    output_dir = VIDEO_ROOT / "audio" / "narration-v5"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment["text"])
        cached = cache.get(text)
        if cached:
            path = cached
        else:
            key = hashlib.sha256(f"{VOICE}|{RATE}|{PITCH}|{text}".encode()).hexdigest()[:12]
            path = output_dir / f"{index:02d}-{key}.mp3"
            if not path.is_file() or path.stat().st_size < 1024:
                await edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH).save(str(path))
        source_duration = duration(path)
        next_start = float(segments[index]["start"]) if index < len(segments) else TOTAL_SECONDS
        available = max(0.75, next_start - float(segment["start"]) - 0.18)
        tempo = max(1.0, source_duration / available)
        if tempo > 1.25:
            raise RuntimeError(f"narration segment {index} is too dense ({source_duration:.2f}s > {available:.2f}s)")
        rendered.append({**segment, "path": str(path), "source_duration": source_duration, "tempo": tempo, "duration": source_duration / tempo})
    (VIDEO_ROOT / "audio" / "narration-manifest-v5.json").write_text(json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8")
    return rendered


def write_subtitles(rendered: list[dict[str, object]]) -> tuple[Path, Path]:
    cues: list[dict[str, object]] = []
    for segment in rendered:
        start = float(segment["start"])
        end = min(TOTAL_SECONDS - 0.12, start + float(segment["duration"]))
        captions = [str(item).strip() for item in segment["captions"]]
        weights = [max(1, len(re.sub(r"\s+", "", caption))) for caption in captions]
        cursor = start
        for index, (caption, weight) in enumerate(zip(captions, weights, strict=True)):
            cue_end = end if index == len(captions) - 1 else cursor + (end - start) * weight / sum(weights)
            if max(map(len, caption.split("\n"))) > 26:
                raise RuntimeError(f"subtitle line exceeds 26 characters: {caption}")
            cues.append({"start": cursor, "end": cue_end, "text": caption})
            cursor = cue_end

    srt = VIDEO_ROOT / "subtitles-v5.srt"
    srt.write_text("\n".join(
        line
        for index, cue in enumerate(cues, start=1)
        for line in (str(index), f"{srt_time(float(cue['start']))} --> {srt_time(float(cue['end']))}", str(cue["text"]), "")
    ), encoding="utf-8")
    ass = VIDEO_ROOT / "subtitles-v5.ass"
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Microsoft YaHei,46,&H00FFFFFF,&H00FFFFFF,&HCC071018,&H00000000,-1,0,0,0,100,100,0,0,1,2.2,1.0,2,110,110,86,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.rstrip(), *[
        f"Dialogue: 0,{ass_time(float(cue['start']))},{ass_time(float(cue['end']))},Caption,,0,0,0,,{ass_text(str(cue['text']))}"
        for cue in cues
    ]]
    ass.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return srt, ass


def mix(rendered: list[dict[str, object]]) -> Path:
    command = [str(FFMPEG), "-y"]
    filters: list[str] = []
    voices: list[str] = []
    for index, segment in enumerate(rendered):
        command.extend(["-i", str(segment["path"])])
        delay = round(float(segment["start"]) * 1000)
        label = f"v{index}"
        filters.append(
            f"[{index}:a]aresample=48000,atempo={float(segment['tempo']):.6f},highpass=f=72,lowpass=f=14500,"
            "equalizer=f=165:t=q:w=1:g=1.4,equalizer=f=3300:t=q:w=1.1:g=1,deesser=i=0.18:m=0.42:f=0.48,"
            f"acompressor=threshold=-21dB:ratio=3:attack=14:release=170:makeup=2.5,adelay={delay}|{delay}[{label}]"
        )
        voices.append(f"[{label}]")
    music_index = len(rendered)
    command.extend(["-stream_loop", "-1", "-i", str(VIDEO_ROOT / "audio" / "evomind-industrial-score-v4.wav")])
    filters.extend([
        "".join(voices)
        + f"amix=inputs={len(voices)}:normalize=0,apad=whole_dur={TOTAL_SECONDS},atrim=0:{TOTAL_SECONDS},"
        + f"loudnorm=I=-15:LRA=6:TP=-1.8,asetpts=N/SR/TB,apad=whole_dur={TOTAL_SECONDS},atrim=0:{TOTAL_SECONDS}[voice]",
        f"[{music_index}:a]aresample=48000,atrim=0:{TOTAL_SECONDS},volume=3.0[music]",
        "[music][voice]sidechaincompress=threshold=0.016:ratio=9:attack=20:release=460:makeup=1[ducked]",
        f"[ducked][voice]amix=inputs=2:weights='1 1':normalize=0,atrim=0:{TOTAL_SECONDS},loudnorm=I=-14:LRA=7:TP=-1,volume=7.5dB,"
        f"alimiter=limit=0.891:attack=5:release=80:level=0:latency=1,asetpts=N/SR/TB,"
        f"apad=whole_dur={TOTAL_SECONDS},atrim=0:{TOTAL_SECONDS}[aout]",
    ])
    output = VIDEO_ROOT / "audio" / "final-mix-v5.wav"
    command.extend([
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[aout]",
        "-t",
        str(TOTAL_SECONDS),
        "-ar",
        "48000",
        "-ac",
        "2",
        str(output),
    ])
    run(command)
    return output


def main() -> int:
    rendered = asyncio.run(render_voice())
    srt, ass = write_subtitles(rendered)
    output = mix(rendered)
    mix_duration = duration(output)
    if abs(mix_duration - TOTAL_SECONDS) > 0.02:
        raise RuntimeError(f"final mix duration is {mix_duration:.6f}s, expected {TOTAL_SECONDS:.6f}s")
    print(json.dumps({
        "voice": VOICE,
        "segments": len(rendered),
        "duration_seconds": mix_duration,
        "mix": str(output),
        "srt": str(srt),
        "ass": str(ass),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
