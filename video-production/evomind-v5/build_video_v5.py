from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIDEO_ROOT = Path(r"E:\EvoMind-release-validation\evomind-commercial-video-v2-20260722-165206")
DELIVERY_ROOT = Path(r"E:\EvoMind-release-validation\evomind-v5-professional-video-20260723")
FFMPEG = Path(r"D:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe")
FFPROBE = Path(r"D:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe")
FONT = "C\\:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C\\:/Windows/Fonts/msyhbd.ttc"
TRANSITION = 0.25
TOTAL_SECONDS = 208.0
BRAND_SECONDS = 4.25
NAV_LABELS = ("输入", "拆解", "协作", "训练", "Human Gate", "审核", "交付", "交互微调")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")


def drawtext(value: str, x: str, y: str, size: int, color: str, *, bold: bool = False) -> str:
    font = FONT_BOLD if bold else FONT
    return f"drawtext=fontfile='{font}':text='{escape(value)}':fontcolor={color}:fontsize={size}:x={x}:y={y}"


def nav(active: int) -> str:
    widths = (112, 112, 112, 112, 176, 112, 112, 150)
    gap = 9
    x = (1920 - sum(widths) - gap * 7) // 2
    filters = ["drawbox=x=0:y=0:w=iw:h=112:color=0x071116@0.98:t=fill", "drawbox=x=0:y=110:w=iw:h=2:color=0x29434B@0.9:t=fill"]
    for index, (label, width) in enumerate(zip(NAV_LABELS, widths, strict=True)):
        box = "0x19BFAF@0.96" if index == active else "0x315F52@0.92" if index < active else "0x223038@0.94"
        text = "0x041514" if index == active else "0xC7E7DC" if index < active else "0x8FA2AA"
        filters.extend([f"drawbox=x={x}:y=42:w={width}:h=42:color={box}:t=fill", drawtext(label, f"{x}+({width}-text_w)/2", "50", 20, text, bold=True)])
        x += width + gap
    return ",".join(filters)


def render_v4(source: Path, output: Path) -> None:
    reviewer_panel = ",".join(
        [
            "drawbox=x=1058:y=278:w=654:h=394:color=0x111B1F@0.99:t=fill:enable='between(t,132.5,137.4)'",
            "drawbox=x=1058:y=278:w=654:h=394:color=0x2CA99B@0.92:t=2:enable='between(t,132.5,137.4)'",
            drawtext("审核结论", "1090", "310", 31, "0xEFFAF7", bold=True) + ":enable='between(t,132.5,137.4)'",
            "drawbox=x=1090:y=366:w=584:h=48:color=0x203A3E@0.96:t=fill:enable='between(t,132.5,137.4)'",
            drawtext("训练方法与产物  ·  PASS", "1110", "377", 25, "0xB8E7DC") + ":enable='between(t,132.5,137.4)'",
            "drawbox=x=1090:y=422:w=584:h=48:color=0x203A3E@0.96:t=fill:enable='between(t,132.5,137.4)'",
            drawtext("审计与发布边界  ·  PASS", "1110", "433", 25, "0xC8D8DD") + ":enable='between(t,132.5,137.4)'",
        ]
    )
    run(
        [
            str(FFMPEG),
            "-y",
            "-ss",
            "0",
            "-t",
            "138.25",
            "-i",
            str(source),
            "-vf",
            reviewer_panel,
            "-an",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-preset",
            "medium",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def render_capture(
    source: Path,
    output: Path,
    source_in: float,
    duration: float,
    active: int,
    label: str,
    *,
    hide_taskbar: bool = False,
    hide_run_identity: bool = False,
) -> None:
    background = VIDEO_ROOT / "assets" / "evomind-brand-background-image2.png"
    crop_height = 968 if hide_taskbar else 1048
    privacy_filters = [
        # Replace the internal task slug with the public project name in every
        # V5 product capture. The underlying recording remains unchanged.
        "drawbox=x=192:y=208:w=520:h=42:color=0xF7FAFB@0.99:t=fill",
        drawtext("EvoMind 7B 领域微调", "220", "218", 17, "0x18313A", bold=True),
    ]
    if hide_run_identity:
        privacy_filters.extend(
            [
                # The run UUID and workspace fragment are audit-only values.
                "drawbox=x=198:y=386:w=620:h=86:color=0xF7FAFB@0.99:t=fill",
                "drawbox=x=220:y=400:w=566:h=50:color=0xEAF7F3@0.99:t=fill",
                drawtext("当前版本 · V2 审核完成", "246", "412", 19, "0x1B6557", bold=True),
            ]
        )
    privacy = ",".join(privacy_filters)
    filter_complex = (
        "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,eq=brightness=-0.46:saturation=0.55,"
        "drawbox=x=0:y=0:w=iw:h=ih:color=0x03090D@0.54:t=fill[bg];"
        f"[1:v]crop=1672:{crop_height}:248:32,scale=1600:900:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1600:900:(ow-iw)/2:(oh-ih)/2:color=0xF8FAFB,setsar=1[ui];"
        "[bg]drawbox=x=140:y=130:w=1640:h=940:color=black@0.38:t=fill[stage];"
        "[stage][ui]overlay=x=160:y=150:shortest=1[comp];"
        f"[comp]{privacy},drawbox=x=158:y=148:w=1604:h=904:color=0x426B70@0.84:t=2,{nav(active)},"
        f"{drawtext(label, '160', '121', 24, '0xDDF8F4', bold=True)},format=yuv420p[vout]"
    )
    run([
        str(FFMPEG), "-y", "-loop", "1", "-framerate", "30", "-i", str(background),
        "-ss", f"{source_in:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
        "-t", f"{duration:.3f}", "-filter_complex", filter_complex, "-map", "[vout]", "-an", "-r", "30",
        "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(output),
    ])


def render_brand(output: Path) -> None:
    background = VIDEO_ROOT / "assets" / "evomind-brand-background-image2.png"
    filters = ",".join(
        [
            "scale=2000:1125:flags=lanczos",
            f"crop=1920:1080:x='40*t/{BRAND_SECONDS}':y=22",
            "eq=brightness=-0.18:saturation=0.72",
            "drawbox=x=0:y=0:w=iw:h=ih:color=0x02090D@0.42:t=fill",
            "drawbox=x=156:y=286:w=6:h=430:color=0x19BFAF@0.98:t=fill",
            drawtext("EvoMind", "194", "302", 92, "0xF2FBF9", bold=True),
            drawtext("一句话启动。", "200", "458", 40, "0xD7F4EC", bold=True),
            drawtext("全流程执行。", "200", "522", 40, "0xD7F4EC", bold=True),
            drawtext("用结果继续迭代。", "200", "586", 40, "0xD7F4EC", bold=True),
            drawtext("让模型研发可执行、可审核、可复现，也可持续优化。", "200", "692", 31, "0xB5CBC8"),
            "format=yuv420p",
        ]
    )
    run(
        [
            str(FFMPEG),
            "-y",
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(background),
            "-t",
            str(BRAND_SECONDS),
            "-vf",
            filters,
            "-an",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-preset",
            "medium",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def compose(parts: list[Path], output: Path) -> None:
    command = [str(FFMPEG), "-y"]
    for part in parts:
        command.extend(["-i", str(part)])
    filters = [f"[{index}:v]setpts=PTS-STARTPTS,format=yuv420p[v{index}]" for index in range(len(parts))]
    offsets = (138.0, 178.0, 194.0, 199.0, 203.75)
    previous = "v0"
    for index, offset in enumerate(offsets, start=1):
        target = "vfinal" if index == len(parts) - 1 else f"x{index}"
        filters.append(f"[{previous}][v{index}]xfade=transition=fade:duration={TRANSITION}:offset={offset:.3f}[{target}]")
        previous = target
    command.extend(["-filter_complex", ";".join(filters), "-map", "[vfinal]", "-t", str(TOTAL_SECONDS), "-an", "-r", "30", "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(output)])
    run(command)


def mux(picture: Path, output: Path, codec: str) -> None:
    audio = VIDEO_ROOT / "audio" / "final-mix-v5.wav"
    subtitles = VIDEO_ROOT / "subtitles-v5.ass"
    subtitle_filter_path = str(subtitles).replace("\\", "/").replace(":", "\\:")
    if codec == "h264":
        video_args = ["-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        audio_args = ["-c:a", "aac", "-b:a", "320k"]
    else:
        video_args = ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"]
        audio_args = ["-c:a", "pcm_s24le"]
    run([
        str(FFMPEG), "-y", "-i", str(picture), "-i", str(audio), "-filter_complex", f"[0:v]ass='{subtitle_filter_path}'[v]",
        "-map", "[v]", "-map", "1:a:0", "-t", str(TOTAL_SECONDS), "-r", "30", *video_args, *audio_args, str(output),
    ])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-master", action="store_true")
    args = parser.parse_args()
    gate = REPO / "video-production" / "evomind-v5" / "verify_v5_evidence.py"
    public_text_gate = REPO / "video-production" / "evomind-v5" / "verify_v5_public_text.py"
    qa = DELIVERY_ROOT / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    run(["python", str(gate), "--workspace-root", str(REPO), "--video-root", str(VIDEO_ROOT), "--ffprobe", str(FFPROBE), "--require-captures", "--json-out", str(qa / "v5-evidence-gate.json")])
    run(["python", str(public_text_gate), "--video-root", str(VIDEO_ROOT), "--require-generated", "--json-out", str(qa / "v5-public-text-gate.json")])
    for required in (VIDEO_ROOT / "audio" / "final-mix-v5.wav", VIDEO_ROOT / "subtitles-v5.ass"):
        if not required.is_file():
            raise FileNotFoundError(required)

    renders = VIDEO_ROOT / "draft" / "v5-rendered"
    renders.mkdir(parents=True, exist_ok=True)
    parts = [renders / f"{index:02d}.mp4" for index in range(1, 7)]
    render_v4(VIDEO_ROOT / "draft" / "evomind-enterprise-v4-picture-edit.mp4", parts[0])
    render_capture(VIDEO_ROOT / "captures" / "v5-report-delivery-v1.mp4", parts[1], 1.0, 40.25, 6, "Nature Skills · Scientific Report")
    render_capture(VIDEO_ROOT / "captures" / "v5-refinement-request.mp4", parts[2], 3.5, 16.25, 7, "Natural-language refinement · Human Gate")
    render_capture(
        VIDEO_ROOT / "captures" / "v5-refinement-completed.mp4",
        parts[3],
        0.0,
        5.25,
        3,
        "Incremental training · Independent Review",
        hide_taskbar=True,
        hide_run_identity=True,
    )
    render_capture(
        VIDEO_ROOT / "captures" / "v5-comparison-final-delivery.mp4",
        parts[4],
        0.0,
        5.0,
        6,
        "V1 / V2 comparison · Final delivery",
        hide_taskbar=True,
    )
    render_brand(parts[5])
    picture = VIDEO_ROOT / "draft" / "evomind-enterprise-v5-picture-edit.mp4"
    compose(parts, picture)

    DELIVERY_ROOT.mkdir(parents=True, exist_ok=True)
    public = DELIVERY_ROOT / "EvoMind-Enterprise-AI-Workflow-V5.mp4"
    mux(picture, public, "h264")
    outputs = [public]
    if not args.skip_master:
        master = DELIVERY_ROOT / "EvoMind-Enterprise-AI-Workflow-V5-ProRes422.mov"
        mux(picture, master, "prores")
        outputs.append(master)
    manifest = {"schema": "evomind.video.v5_build.v1", "duration_seconds": TOTAL_SECONDS, "outputs": [{"path": str(item), "bytes": item.stat().st_size, "sha256": sha256(item)} for item in outputs]}
    (DELIVERY_ROOT / "v5-build-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
