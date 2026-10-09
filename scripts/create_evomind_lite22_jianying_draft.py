from __future__ import annotations

import os
import sys
from pathlib import Path


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_ROOT = os.getenv("JY_SKILL_ROOT", "").strip()
SKILL_CANDIDATES = [
    ENV_ROOT,
    os.path.join(CURRENT_DIR, ".agent", "skills", "jianying-editor"),
    os.path.join(CURRENT_DIR, ".trae", "skills", "jianying-editor"),
    os.path.join(CURRENT_DIR, ".claude", "skills", "jianying-editor"),
    os.path.join(CURRENT_DIR, "skills", "jianying-editor"),
    os.path.abspath(".agent/skills/jianying-editor"),
    os.path.dirname(CURRENT_DIR),
]

SCRIPTS_PATH = None
ATTEMPTED: list[str] = []
for candidate in SKILL_CANDIDATES:
    if not candidate:
        continue
    candidate = os.path.abspath(candidate)
    ATTEMPTED.append(candidate)
    if os.path.exists(os.path.join(candidate, "scripts", "jy_wrapper.py")):
        SCRIPTS_PATH = os.path.join(candidate, "scripts")
        break

if not SCRIPTS_PATH:
    raise ImportError(
        "Could not find jianying-editor/scripts/jy_wrapper.py\nTried:\n- "
        + "\n- ".join(ATTEMPTED)
    )
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from jy_wrapper import JyProject  # noqa: E402


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bundle = project_root / "video-production" / "evomind-lite22-training-20260808"
    source_video = bundle / "sources" / "real-training-chrome-cdp.mp4"
    subtitle_path = bundle / "final" / "evomind-lite22-demo.zh-CN.srt"
    narration_dir = bundle / "final" / "narration-clips"
    starts = (0.0, 8.0, 17.0, 27.0, 37.0, 50.0, 61.0, 73.0)

    required = [source_video, subtitle_path]
    required.extend(narration_dir / f"narration-{index:02d}.mp3" for index in range(1, 9))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing video draft inputs:\n" + "\n".join(missing))

    project = JyProject(
        project_name="EvoMind_Lite22_Novice_Training_Demo_20260808",
        overwrite=True,
        width=1920,
        height=1080,
    )
    project.add_media_safe(
        str(source_video),
        start_time="0s",
        duration="81.5s",
        track_name="ChromeProductCapture",
    )
    for index, start in enumerate(starts, 1):
        project.add_media_safe(
            str(narration_dir / f"narration-{index:02d}.mp3"),
            start_time=f"{start:g}s",
            track_name="ChineseNarration",
        )
    project.script.import_srt(str(subtitle_path), track_name="ChineseSubtitles")
    project.save()
    print(f"draft_name={project.name}")
    print(f"draft_folder={project.draft_dir}")
    print("video_segments=1")
    print("narration_segments=8")
    print("subtitle_source=evomind-lite22-demo.zh-CN.srt")


if __name__ == "__main__":
    main()
