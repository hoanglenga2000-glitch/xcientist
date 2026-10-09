from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ENV_ROOT = os.getenv("JY_SKILL_ROOT", "").strip()
SKILL_CANDIDATES = [
    ENV_ROOT,
    PROJECT_ROOT / ".agent" / "skills" / "jianying-editor",
    PROJECT_ROOT / ".trae" / "skills" / "jianying-editor",
    PROJECT_ROOT / ".claude" / "skills" / "jianying-editor",
    Path.home() / ".codex" / "skills" / "jianying-editor",
]
SKILL_ROOT = next(
    (Path(candidate).resolve() for candidate in SKILL_CANDIDATES if candidate and (Path(candidate) / "scripts" / "jy_wrapper.py").is_file()),
    None,
)
if SKILL_ROOT is None:
    raise ImportError("jianying-editor skill was not found")
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from jy_wrapper import JyProject  # noqa: E402
import pyJianYingDraft as draft  # noqa: E402
from pyJianYingDraft.keyframe import KeyframeProperty as KP  # noqa: E402

PROJECT_NAME = "EvoMind_From_Request_to_Result_20260809"
DELIVERY_STEM = "EvoMind_From_Request_to_Result_20260809"
BGM_ID = "7377843352954243081"
BGM_NAME = "商务宣传 科技 产品展示"
BGM_SOURCE_START = 0.56
BGM_LOOP_DURATION = 111.25
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="named local video or image source; repeat for each source",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-name", default=PROJECT_NAME)
    return parser.parse_args()


def parse_sources(specs: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        if not separator:
            name, raw_path = "default", spec
        name = name.strip()
        path = Path(raw_path.strip()).resolve()
        if not name or not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
            raise FileNotFoundError(f"invalid local media source: {spec}")
        if name in sources:
            raise ValueError(f"duplicate source name: {name}")
        sources[name] = path
    return sources


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_brand_cards(bundle: Path) -> tuple[Path, Path]:
    """Render restrained title cards from the repository's real brand assets."""

    brand_root = PROJECT_ROOT / "web" / "research-agent-workstation" / "public" / "brand"
    background_path = brand_root / "evomind-deep-sea-desktop.webp"
    logo_path = brand_root / "evomind-mark.png"
    if not background_path.is_file() or not logo_path.is_file():
        raise FileNotFoundError("EvoMind brand assets are missing")
    output_dir = bundle / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    intro_path = output_dir / "EvoMind_Product_Story_Intro.png"
    outro_path = output_dir / "EvoMind_Product_Story_Outro.png"

    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    font_path = next((path for path in font_candidates if path.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("a Chinese title font was not found")

    background = Image.open(background_path).convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS)
    logo = Image.open(logo_path).convert("RGBA")
    logo.thumbnail((156, 156), Image.Resampling.LANCZOS)

    def render(path: Path, headline: str, subline: str) -> None:
        canvas = background.copy().convert("RGBA")
        veil = Image.new("RGBA", canvas.size, (0, 8, 12, 48))
        canvas = Image.alpha_composite(canvas, veil)
        canvas.alpha_composite(logo, (160, 158))
        draw = ImageDraw.Draw(canvas)
        title_font = ImageFont.truetype(str(font_path), 58)
        sub_font = ImageFont.truetype(str(font_path), 34)
        draw.text((352, 174), "EvoMind", font=title_font, fill=(239, 250, 250, 255))
        draw.text((352, 252), "Scientific Research Agent", font=sub_font, fill=(95, 221, 205, 255))
        headline_font = ImageFont.truetype(str(font_path), 52)
        subline_font = ImageFont.truetype(str(font_path), 28)
        draw.text((160, 596), headline, font=headline_font, fill=(242, 249, 249, 255))
        draw.text((162, 682), subline, font=subline_font, fill=(159, 184, 184, 255))
        canvas.convert("RGB").save(path, format="PNG", optimize=True)

    render(intro_path, "从研究目标，到可复现实验", "一次指令，推进一次完整的机器学习任务")
    render(outro_path, "从一个目标，到一套可复现结果", "EvoMind · 科研智能体")
    return intro_path, outro_path


def ensure_internal_id_redaction(bundle: Path) -> Path:
    """Create a transparent, UI-matched mask for the internal run identifier.

    The public film keeps the real run UI and metrics, but the implementation-only
    run identifier contains an old ``demo`` token.  A small card-coloured patch is
    motion-keyframed over that single line in Jianying; no result or metric is
    altered.
    """

    output = bundle / "assets" / "EvoMind_Internal_Run_Id_Redaction.png"
    canvas = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (535, 770, 1235, 826),
        radius=12,
        fill=(4, 15, 14, 255),
    )
    canvas.save(output, format="PNG", optimize=True)
    return output


def probe_dimensions(path: Path) -> tuple[int, int]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def crop_transform(path: Path, crop_rect: list[float] | None) -> tuple[float, float, float]:
    if not crop_rect:
        return 1.0, 0.0, 0.0
    if len(crop_rect) != 4:
        raise ValueError(f"crop_rect requires x,y,w,h: {crop_rect!r}")
    source_width, source_height = probe_dimensions(path)
    x, y, width, height = map(float, crop_rect)
    if min(x, y, width, height) < 0 or x + width > source_width or y + height > source_height:
        raise ValueError(f"crop_rect escapes source bounds for {path.name}: {crop_rect!r}")
    if abs((width / height) - (16 / 9)) > 0.01:
        raise ValueError(f"crop_rect must be 16:9 for {path.name}: {crop_rect!r}")
    scale = max(source_width / width, source_height / height)
    center_x = x + width / 2
    center_y = y + height / 2
    position_x = ((source_width / 2) - center_x) / (source_width / 2) * scale
    # Jianying's positive Y pans the visible source window downward.  A crop whose
    # centre is below the source centre therefore uses a positive transform.
    position_y = (center_y - (source_height / 2)) / (source_height / 2) * scale
    return scale, position_x, position_y


def load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "evomind.product_story.edit_plan.v2":
        raise ValueError("invalid edit plan schema")
    clips = payload.get("clips")
    narration = payload.get("narration")
    if not isinstance(clips, list) or not clips or not isinstance(narration, list) or not narration:
        raise ValueError("edit plan requires clips and narration")
    duration = sum(float(item["source_duration"]) / float(item.get("speed") or 1.0) for item in clips)
    if not 110 <= duration <= 112:
        raise ValueError(f"final timeline must be 110-112 seconds, got {duration:.3f}")
    payload["timeline_duration"] = duration
    return payload


def ensure_narration(plan: dict[str, Any], bundle: Path) -> tuple[Path, list[Path]]:
    narration_dir = bundle / "final" / "narration"
    narration_dir.mkdir(parents=True, exist_ok=True)
    edge_tts = PROJECT_ROOT / ".venv" / "Scripts" / "edge-tts.exe"
    if not edge_tts.is_file():
        raise FileNotFoundError("edge-tts is missing from the project virtual environment")
    srt_lines: list[str] = []
    clips: list[Path] = []
    for index, item in enumerate(plan["narration"], 1):
        start = float(item["start"])
        end = float(item["end"])
        text = str(item["text"]).strip()
        if not text or end <= start:
            raise ValueError(f"invalid narration item {index}")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        raw_output = narration_dir / f"product-story-{index:02d}-{text_hash}.mp3"
        output = narration_dir / f"product-story-{index:02d}-{text_hash}-loudnorm.wav"
        if not output.is_file() or output.stat().st_size <= 44:
            subprocess.run(
                [str(edge_tts), "--voice", "zh-CN-XiaoxiaoNeural", "--rate", "+4%", "--text", text, "--write-media", str(raw_output)],
                check=True,
                cwd=PROJECT_ROOT,
            )
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-y", "-loglevel", "error", "-i", str(raw_output),
                    "-af", "loudnorm=I=-14:TP=-1:LRA=7", "-ar", "48000", "-ac", "2", str(output),
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )
        clips.append(output)
        srt_lines.extend([str(index), f"{srt_timestamp(start)} --> {srt_timestamp(end)}", text, ""])
    srt = bundle / "final" / f"{DELIVERY_STEM}.zh-CN.srt"
    srt.write_text("\n".join(srt_lines), encoding="utf-8")
    return srt, clips


def main() -> int:
    args = parse_args()
    plan_path = args.plan.resolve()
    bundle = PROJECT_ROOT / "video-production" / "evomind-real-gpu-training-20260808"
    ensure_brand_cards(bundle)
    internal_id_redaction = ensure_internal_id_redaction(bundle)
    sources = parse_sources(args.source)
    plan = load_plan(plan_path)
    srt, narration_clips = ensure_narration(plan, bundle)
    project = JyProject(args.project_name, overwrite=True, width=1920, height=1080)
    # The wrapper treats the conventional 1920x1080 default as auto-resolution.
    # This production is explicitly 1080p, so prevent the first 2560x1440 source
    # from silently changing the draft canvas.
    project._explicit_res = True

    cursor = 0.0
    video_segments = []
    for index, clip in enumerate(plan["clips"]):
        source_name = str(clip.get("source") or "default")
        if source_name not in sources:
            raise KeyError(f"clip {index + 1} references unknown source {source_name!r}")
        source = sources[source_name]
        source_start = float(clip.get("source_start") or 0.0)
        source_duration = float(clip["source_duration"])
        speed = float(clip.get("speed") or 1.0)
        target_duration = source_duration / speed
        is_still = source.suffix.lower() in IMAGE_SUFFIXES
        media_args: dict[str, Any] = {
            "start_time": f"{cursor:.6f}s",
            "duration": f"{source_duration:.6f}s",
            "track_name": "ProductStory",
        }
        if not is_still:
            media_args["source_start"] = f"{source_start:.6f}s"
        segment = project.add_media_safe(str(source), **media_args)
        if segment is None:
            raise RuntimeError(f"failed to add source clip {index + 1}")
        if is_still and speed != 1.0:
            raise ValueError("still-image clips cannot be time-compressed")
        if not is_still:
            segment.speed.speed = speed
        segment.target_timerange.duration = round(target_duration * 1_000_000)
        base_scale, base_x, base_y = crop_transform(source, clip.get("crop_rect"))
        zoom_keyframes = clip.get("zoom_keyframes") or []
        if clip.get("crop_rect") and not zoom_keyframes:
            zoom_keyframes = [{"at": 0.0}, {"at": target_duration}]
        previous_at = -1.0
        for keyframe in zoom_keyframes:
            at = float(keyframe["at"])
            if not 0 <= at <= target_duration:
                raise ValueError(f"zoom keyframe escapes clip {index + 1}")
            if at <= previous_at:
                raise ValueError(f"zoom keyframes must be strictly increasing for clip {index + 1}")
            relative_scale = float(keyframe.get("scale", 1.0))
            if not 1.0 <= relative_scale <= 1.15:
                raise ValueError(f"zoom keyframe scale is outside the restrained 1.00-1.15 range for clip {index + 1}")
            segment.add_keyframe(KP.uniform_scale, f"{at:.6f}s", base_scale * relative_scale)
            segment.add_keyframe(KP.position_x, f"{at:.6f}s", base_x + float(keyframe.get("x", 0.0)))
            segment.add_keyframe(KP.position_y, f"{at:.6f}s", base_y + float(keyframe.get("y", 0.0)))
            previous_at = at
        if index and bool(clip.get("transition", True)):
            transition = project.add_transition_simple("叠化", video_segment=segment, duration="0.2s")
            if transition is None:
                raise RuntimeError(f"failed to add the chapter transition for clip {index + 1}")
        speed_label = str(clip.get("speed_label") or "").strip()
        if speed_label:
            project.add_text_simple(
                speed_label,
                start_time=f"{cursor + 0.4:.3f}s",
                duration=f"{min(2.2, target_duration - 0.4):.3f}s",
                track_name="SpeedLabels",
                font_size=1.8,
                color_rgb=(0.72, 0.91, 0.88),
                transform_x=0.82,
                transform_y=0.78,
                bold=True,
                align=2,
                max_line_width=0.28,
            )
        video_segments.append(segment)
        cursor += target_duration

    for mask_index, mask in enumerate(plan.get("privacy_masks") or [], 1):
        start = float(mask["start"])
        duration = float(mask["duration"])
        mask_segment = project.add_media_safe(
            str(internal_id_redaction),
            start_time=f"{start:.6f}s",
            duration=f"{duration:.6f}s",
            track_name=f"InternalIdRedaction{mask_index}",
        )
        if mask_segment is None:
            raise RuntimeError(f"failed to add internal-id redaction {mask_index}")
        mask_segment.target_timerange.duration = round(duration * 1_000_000)
        positions = mask.get("positions") or [{"at": 0.0, "y": 800.0}, {"at": duration, "y": 800.0}]
        for position in positions:
            at = float(position["at"])
            if not 0 <= at <= duration:
                raise ValueError(f"redaction keyframe escapes mask {mask_index}")
            output_y = float(position["y"])
            # The patch is centred at y=798 in a 1080p transparent canvas.
            # JianYing's overlay coordinate system is vertically inverted here:
            # a negative position_y moves this transparent-canvas asset down.
            normalized_y = (798.0 - output_y) / 540.0
            mask_segment.add_keyframe(KP.position_y, f"{at:.6f}s", normalized_y)

    for overlay_index, overlay in enumerate(plan.get("overlays") or [], 1):
        project.add_text_simple(
            str(overlay["text"]),
            start_time=f"{float(overlay['start']):.3f}s",
            duration=f"{float(overlay['duration']):.3f}s",
            track_name=f"PromoOverlay{overlay_index}",
            font_size=float(overlay.get("font_size") or 3.8),
            color_rgb=tuple(overlay.get("color_rgb") or (1.0, 1.0, 1.0)),
            transform_x=float(overlay.get("transform_x", -0.54)),
            transform_y=float(overlay.get("transform_y", 0.72)),
            bold=bool(overlay.get("bold", True)),
            align=int(overlay.get("align", 0)),
            auto_wrapping=True,
            max_line_width=float(overlay.get("max_line_width", 0.62)),
            anim_in=overlay.get("anim_in"),
            anim_loop=overlay.get("anim_loop"),
            anim_loop_duration=overlay.get("anim_loop_duration"),
        )

    for item, audio_path in zip(plan["narration"], narration_clips, strict=True):
        narration_segment = project.add_audio_safe(
            str(audio_path), start_time=f"{float(item['start']):.3f}s", track_name="ChineseFemaleNarration"
        )
        if narration_segment is None:
            raise RuntimeError(f"failed to add narration {audio_path.name}")
        narration_segment.volume = 1.05

    subtitle_reference = draft.TextSegment(
        "subtitle-style-reference",
        draft.trange(0, 1),
        style=draft.TextStyle(
            size=2.2,
            color=(1.0, 1.0, 1.0),
            align=1,
            auto_wrapping=True,
            max_line_width=0.68,
            line_spacing=-2,
        ),
        border=draft.TextBorder(color=(0.0, 0.0, 0.0), alpha=0.88, width=14.0),
        shadow=draft.TextShadow(color=(0.0, 0.0, 0.0), alpha=0.62, diffuse=6.0, distance=2.0),
        clip_settings=draft.ClipSettings(transform_y=-0.83),
    )
    project.script.import_srt(
        str(srt),
        track_name="ChineseSubtitles",
        style_reference=subtitle_reference,
        clip_settings=None,
    )

    bgm_path = next((bundle / "assets").glob(f"{BGM_ID}_*.m4a"), None)
    if bgm_path is None or not bgm_path.is_file():
        raise FileNotFoundError("the verified local JianYing technology BGM is missing")
    bgm_cursor = 0.0
    while bgm_cursor < cursor:
        duration = min(BGM_LOOP_DURATION, cursor - bgm_cursor)
        bgm = project.add_audio_safe(
            str(bgm_path),
            start_time=f"{bgm_cursor:.3f}s",
            duration=f"{duration:.3f}s",
            track_name="TechnologyBGM",
            source_start=f"{BGM_SOURCE_START:.3f}s",
        )
        if bgm is None:
            raise RuntimeError("failed to add the verified local JianYing BGM")
        bgm.volume = 0.10
        fade_in = "0.45s" if bgm_cursor == 0.0 else "0s"
        fade_out = "0.45s" if bgm_cursor + duration >= cursor - 1e-6 else "0s"
        bgm.add_fade(fade_in, fade_out)
        bgm_cursor += duration

    project.save()
    manifest = {
        "schema": "evomind.product_story.jianying_draft.v2",
        "project_name": args.project_name,
        "draft_folder": str(project.draft_dir),
        "sources": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in sources.items()
        },
        "edit_plan": str(plan_path),
        "edit_plan_sha256": file_sha256(plan_path),
        "edit_plan_revision": plan.get("revision", "unspecified"),
        "motion_keyframes": sum(len(clip.get("zoom_keyframes") or []) for clip in plan["clips"]),
        "timeline_duration": round(cursor, 3),
        "video_segments": len(video_segments),
        "narration_segments": len(narration_clips),
        "subtitle_path": str(srt),
        "bgm_id": BGM_ID,
        "bgm_name": BGM_NAME,
        "bgm_path": str(bgm_path),
        "bgm_sha256": file_sha256(bgm_path),
        "narration_volume": 1.05,
        "bgm_volume": 0.10,
        "bgm_source_start": BGM_SOURCE_START,
        "bgm_loop_duration": BGM_LOOP_DURATION,
        "subtitle_style": {
            "font_size": 2.2,
            "transform_y": -0.83,
            "max_line_width": 0.68,
            "border_width": 14.0,
        },
        "promo_safe_zone": "top-safe-zone",
        "public_story": "single-user-request-to-reproducible-result",
    }
    manifest_path = bundle / "final" / "jianying-draft-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
