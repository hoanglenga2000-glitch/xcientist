from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import re
import textwrap
from pathlib import Path
from typing import Any

from research_agent_workstation.server.core.gpu_credentials import ALLOWED_GPU_REMOTE_ROOT

PERSISTENT_COMPETITION_DATA_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/competition_data"
MLEBENCH_PREPARED_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_official_data"

COMPETITION_CATALOG: dict[str, dict[str, Any]] = {
    "cure_bench": {
        "title": "CURE-Bench",
        "source": "https://www.kaggle.com/competitions/cure-bench",
        "access": "kaggle_rules_and_controlled_token",
        "aliases": ("cure", "curebench", "cure-bench"),
    },
    "e2lmc": {
        "title": "E2LMC",
        "source": "https://e2lmc.github.io/starter_kit.html",
        "access": "public_google_drive_notebooks",
        "aliases": ("e2lm", "e2lmc", "early training"),
    },
    "mindgames": {
        "title": "MindGames",
        "source": "https://github.com/mind-games-challenge/mindgames-starter-kit",
        "access": "public_git_and_pypi",
        "aliases": ("mindgames", "mind games", "mind-games"),
    },
    "ariel_2025": {
        "title": "Ariel Data Challenge 2025",
        "source": "https://www.kaggle.com/competitions/ariel-data-challenge-2025",
        "access": "kaggle_rules_and_controlled_token",
        "aliases": ("ariel", "ariel r2", "ariel 2025"),
    },
    "weather4cast": {
        "title": "Weather4cast 2025",
        "source": "https://weather4cast.net/neurips2025/",
        "access": "controlled_sftp_secret",
        "aliases": ("weather4cast", "weather 4 cast", "w4c"),
    },
    "open_polymer": {
        "title": "Open Polymer Challenge",
        "source": "https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025",
        "access": "kaggle_rules_and_controlled_token",
        "aliases": ("open polymer", "polymer", "neurips open polymer"),
    },
}

# Sources whose official archive is already prepared on the shared GPU data
# root (MLE-bench layout).  They are deliberately kept OUT of
# COMPETITION_CATALOG: the shipped six-competition contract (and its pinned
# catalogue hash) must stay stable.  These sources resolve through the same
# competition_data_status / competition_data_prepare tools, but preparation is
# read-only verification of the prepared tree plus a persistent pointer, so a
# research Run can train from public/ without the managed Kaggle connector.
LOCAL_PREPARED_SOURCES: dict[str, dict[str, Any]] = {
    "histopathologic_cancer": {
        "title": "Histopathologic Cancer Detection (MLE-bench prepared)",
        "source": "https://www.kaggle.com/c/histopathologic-cancer-detection",
        "access": "local_prepared_mlebench",
        "aliases": (
            "histopathologic",
            "histopathologic cancer",
            "histopathologic-cancer-detection",
            "histopathologic cancer detection",
            "pathology",
        ),
        "mlebench_slug": "histopathologic-cancer-detection",
    },
}

_LOCAL_PREPARED_COMPETITIONS = frozenset(LOCAL_PREPARED_SOURCES)

_SAFE_COMPETITION = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_E2LMC_NOTEBOOKS = (
    ("01_how_to_interact_with_model.ipynb", "1pmRaGgVulB391Jb26ixI8g9guBgFP836"),
    ("02_how_to_evaluate_a_model.ipynb", "11WLb8Wqh4ASQ-Qejs8HTFLZagVFaK01z"),
    ("03_reproduce_baseline_results.ipynb", "1pMyQUEOi0Ng1Fm1RBhbQUOBPtE9iJGGV"),
    ("04_1_how_to_contribute.ipynb", "1gnkTry6OOuuPlDWm6cMinrzm58LMMxkV"),
    ("04_2_how_to_contribute_advanced.ipynb", "1WYY-YRJXySeIQijKIeBn6HQb_xNL_4eR"),
    ("05_scoring.ipynb", "1sH0Pe-HS2zJyFt0yxec7dyMk0KY2ZRlz"),
    ("06_submission.ipynb", "1fY-hkS13sf5FAJuwW2AhnLsPPV_SeX8q"),
    ("07_scientific_alignment.ipynb", "127EYY-edGPwivnZwkDiSXyneFdj9AObO"),
)
_MINDGAMES_ENVIRONMENTS = (
    ("SecretMafia-v0", 6),
    ("Codenames-v0", 4),
    ("ColonelBlotto-v0", 2),
    ("ThreePlayerIPD-v0", 3),
)
_MINDGAMES_TEXTARENA_VERSION = "0.7.4"
_MINDGAMES_NLTK_DEPENDENCIES = (
    (
        "corpora/words.zip",
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/words.zip",
        "54ed02917d6771dcc3e8141218960d020947f7f2ccfd9ac9b320979349746015",
    ),
    (
        "taggers/averaged_perceptron_tagger_eng.zip",
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip",
        "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b",
    ),
)
_MINDGAMES_RUNTIME_DEPENDENCIES = {
    relative: {"source": source, "sha256": sha256}
    for relative, source, sha256 in _MINDGAMES_NLTK_DEPENDENCIES
}
_KAGGLE_COMPETITIONS = {
    "cure_bench": "cure-bench",
    "ariel_2025": "ariel-data-challenge-2025",
    "open_polymer": "neurips-open-polymer-prediction-2025",
}


def normalize_competition(value: str) -> str:
    folded = " ".join(str(value or "").strip().casefold().replace("_", " ").split())
    for name, item in {**COMPETITION_CATALOG, **LOCAL_PREPARED_SOURCES}.items():
        if folded == name.replace("_", " ") or folded in item["aliases"]:
            return name
    raise ValueError("unsupported competition data source")


def persistent_root(competition: str) -> str:
    name = normalize_competition(competition)
    if not _SAFE_COMPETITION.fullmatch(name):
        raise ValueError("invalid competition name")
    return f"{PERSISTENT_COMPETITION_DATA_ROOT}/{name}"


def required_secret_purpose(competition: str) -> str:
    name = normalize_competition(competition)
    if name == "weather4cast":
        return "weather4cast_sftp"
    if name in _KAGGLE_COMPETITIONS:
        return "kaggle_api"
    return ""


def _script_header(competition: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        COMPETITION={competition!r}
        ROOT="$EVOMIND_COMPETITION_DATA_ROOT"
        RECEIPT="$EVOMIND_COMPETITION_RECEIPT"
        mkdir -p "$ROOT" "$(dirname "$RECEIPT")" "$ROOT/.evomind"
        write_receipt() {{
          python3 - "$RECEIPT" "$COMPETITION" "$ROOT" "$1" "$2" <<'PY'
        import json, os, re, sys
        path, competition, root, status, detail = sys.argv[1:]
        payload = {{
          "schema": "evomind.competition_data_receipt.v1",
          "competition": competition,
          "status": status,
          "detail": detail,
          "persistent_root": root,
          "files": 0,
          "bytes": 0,
          "manifest_sha256": "",
          "loader_smoke": {{}},
          "secret_values_logged": False,
        }}
        manifest = os.path.join(root, ".evomind", "data-manifest.json")
        manifest_data = {{}}
        if os.path.isfile(manifest):
          try:
            with open(manifest, "rb") as handle:
              raw = handle.read()
            data = json.loads(raw)
            manifest_data = data if isinstance(data, dict) else {{}}
            payload["files"] = int(data.get("file_count", 0))
            payload["bytes"] = int(data.get("total_bytes", 0))
            import hashlib
            payload["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            payload["loader_smoke"] = data.get("loader_smoke", {{}})
            payload["duplicate_files"] = int(data.get("duplicate_files", 0))
            for key in (
              "expected_environment_count", "textarena_version", "runtime_dependencies",
              "valid_unique_notebooks", "archive_sha256", "archive_zip_ok",
              "unsafe_member_count", "expected_file_count", "attempt_id",
              "skipped_directory_count", "remote_file_set_sha256",
            ):
              if key in data:
                payload[key] = data[key]
          except Exception as exc:
            payload["manifest_error"] = type(exc).__name__
        job_root = os.path.join(root, ".evomind", "download-job")
        state_path = os.path.join(job_root, "state.json")
        pid_path = os.path.join(job_root, "pid")
        exit_code_path = os.path.join(job_root, "exit-code")
        selector_path = os.path.join(job_root, "current-attempt.json")
        selector_present = os.path.isfile(selector_path)
        selector_valid = False
        selector_attempt_id = ""
        if selector_present:
          try:
            selector = json.load(open(selector_path, encoding="utf-8"))
            attempt_id = str(selector.get("attempt_id") or "")
            if re.fullmatch(r"[a-f0-9]{{32}}", attempt_id) is None:
              raise ValueError("invalid attempt id")
            selector_valid = True
            selector_attempt_id = attempt_id
            attempt_root = os.path.join(job_root, "attempts", attempt_id)
            state_path = os.path.join(attempt_root, "state.json")
            pid_path = os.path.join(attempt_root, "pid")
            exit_code_path = os.path.join(attempt_root, "exit-code")
            payload["attempt_id"] = attempt_id
          except Exception as exc:
            payload["selector_error"] = type(exc).__name__
        manifest_matches_current = (
          not selector_present
          or (
            selector_valid
            and manifest_data.get("attempt_id") == selector_attempt_id
          )
        )
        if not manifest_matches_current:
          payload["files"] = 0
          payload["bytes"] = 0
          payload["manifest_sha256"] = ""
          payload["loader_smoke"] = {{}}
          payload["duplicate_files"] = 0
          for key in (
            "expected_environment_count", "textarena_version", "runtime_dependencies",
              "valid_unique_notebooks", "archive_sha256", "archive_zip_ok",
              "unsafe_member_count", "expected_file_count",
              "skipped_directory_count", "remote_file_set_sha256",
          ):
            payload.pop(key, None)
        if os.path.isfile(state_path):
          try:
            state = json.load(open(state_path, encoding="utf-8"))
            if selector_valid and state.get("attempt_id") != selector_attempt_id:
              raise ValueError("state attempt mismatch")
            for key in (
              "phase", "failure_code", "official_url", "source_competition_slug",
              "rules_accepted", "attempt_id", "adapter_listing_status",
              "adapter_listing_skipped_directories", "adapter_listing_timeout_exhaustions",
            ):
              if key in state:
                payload[key] = state[key]
          except Exception as exc:
            payload["state_error"] = type(exc).__name__
        if status == "FULL_DATA_READY":
          payload["phase"] = "complete"
          payload["failure_code"] = ""
        partial_path = os.path.join(job_root, "archive.partial")
        payload["partial_archive_bytes"] = os.path.getsize(partial_path) if os.path.isfile(partial_path) else 0
        worker_alive = False
        if os.path.isfile(pid_path):
          try:
            worker_pid = int(open(pid_path, encoding="utf-8").read().strip())
            os.kill(worker_pid, 0)
            worker_alive = True
          except (OSError, TypeError, ValueError):
            worker_alive = False
        payload["worker_alive"] = worker_alive
        if os.path.isfile(exit_code_path):
          try:
            payload["worker_exit_code"] = int(open(exit_code_path, encoding="utf-8").read().strip())
          except (OSError, TypeError, ValueError):
            payload["worker_exit_code_error"] = True
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
          json.dump(payload, handle, sort_keys=True)
          handle.write("\\n")
        os.replace(temporary, path)
        PY
        }}
        """
    )


def _inventory_python(*, loader_expression: str, extra: str = "") -> str:
    return textwrap.dedent(
        f"""\
        python3 - "$ROOT" <<'PY'
        import hashlib, json, os, pathlib, sys
        root = pathlib.Path(sys.argv[1])
        rows = []
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
          if not path.is_file() or ".evomind" in path.parts or ".git" in path.parts or ".runtime" in path.parts:
            continue
          digest = hashlib.sha256()
          with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
              digest.update(chunk)
          rows.append({{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}})
        loader_smoke = {loader_expression}
        payload = {{
          "schema": "evomind.competition_data_manifest.v1",
          "competition": root.name,
          "file_count": len(rows),
          "total_bytes": sum(item["bytes"] for item in rows),
          "files": rows,
          "loader_smoke": loader_smoke,
          "duplicate_files": len(rows) - len({{item["sha256"] for item in rows}}),
        }}
        {extra}
        target = root / ".evomind" / "data-manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
        os.replace(temporary, target)
        PY
        """
    )


def _status_script(competition: str) -> str:
    header = _script_header(competition)
    if competition in _LOCAL_PREPARED_COMPETITIONS:
        check = textwrap.dedent(
            """\
            if test -f "$ROOT/.evomind/data-manifest.json"; then
              state=$(python3 - "$ROOT/.evomind/data-manifest.json" "$COMPETITION" <<'PY'
            import json, re, sys
            data = json.load(open(sys.argv[1], encoding="utf-8"))
            valid = (
              data.get("schema") == "evomind.competition_data_manifest.v1"
              and data.get("competition") == sys.argv[2]
              and data.get("access") == "local_prepared_mlebench"
              and data.get("secret_values_logged") is False
              and data.get("rules_accepted") is True
              and isinstance(data.get("file_count"), int) and data.get("file_count") > 0
              and isinstance(data.get("train_file_count"), int) and data.get("train_file_count") > 0
              and isinstance(data.get("test_file_count"), int) and data.get("test_file_count") > 0
              and isinstance(data.get("label_rows"), int) and data.get("label_rows") > 0
              and re.fullmatch(r"[a-f0-9]{64}", str(data.get("prepared_file_set_sha256") or "")) is not None
            )
            print("FULL_DATA_READY" if valid else "PARTIAL")
            PY
            )
            else state=NOT_STARTED; fi
            write_receipt "$state" "MLE-bench prepared layout on the shared GPU data root"
            """
        )
        return header + check
    if competition == "mindgames":
        expected_environments = json.dumps([name for name, _ in _MINDGAMES_ENVIRONMENTS])
        expected_dependencies = json.dumps(_MINDGAMES_RUNTIME_DEPENDENCIES, sort_keys=True)
        check = textwrap.dedent(
            f"""\
            if test -f "$ROOT/.evomind/data-manifest.json"; then
              state=$(python3 - "$ROOT/.evomind/data-manifest.json" <<'PY'
            import json, sys
            data=json.load(open(sys.argv[1], encoding="utf-8"))
            smoke=data.get("loader_smoke", {{}})
            expected_environments=set({expected_environments})
            expected_dependencies={expected_dependencies}
            valid=(
              data.get("expected_environment_count")==len(expected_environments)
              and data.get("textarena_version")=={_MINDGAMES_TEXTARENA_VERSION!r}
              and data.get("runtime_dependencies")==expected_dependencies
              and set(smoke)==expected_environments
              and all(item.get("ok") is True for item in smoke.values())
            )
            print("FULL_DATA_READY" if valid else "PARTIAL")
            PY
            )
            else state=NOT_STARTED; fi
            write_receipt "$state" "four distinct environment loaders are required"
            """
        )
    elif competition == "e2lmc":
        check = textwrap.dedent(
            """\
            if test -f "$ROOT/.evomind/data-manifest.json"; then
              state=$(python3 - "$ROOT/.evomind/data-manifest.json" <<'PY'
            import json, sys
            data=json.load(open(sys.argv[1], encoding="utf-8"))
            print("FULL_DATA_READY" if data.get("valid_unique_notebooks")==8 else "PARTIAL")
            PY
            )
            else state=NOT_STARTED; fi
            write_receipt "$state" "eight unique valid official notebooks are required"
            """
        )
    elif competition in _KAGGLE_COMPETITIONS:
        slug = _KAGGLE_COMPETITIONS[competition]
        check = textwrap.dedent(
            f"""\
            state=NOT_STARTED
            detail="official Kaggle archive has not started"
            if test -f "$ROOT/.evomind/data-manifest.json" && python3 - "$ROOT/.evomind/data-manifest.json" {slug!r} <<'PY'
            import json, re, sys
            data=json.load(open(sys.argv[1], encoding="utf-8")); smoke=data.get("loader_smoke", {{}})
            valid=(
              data.get("schema")=="evomind.competition_data_manifest.v1"
              and data.get("source_competition_slug")==sys.argv[2]
              and data.get("rules_accepted") is True
              and data.get("archive_zip_ok") is True
              and data.get("unsafe_member_count")==0
              and isinstance(data.get("file_count"), int) and data.get("file_count")>0
              and len(smoke)==data.get("file_count")
              and all(item.get("ok") is True for item in smoke.values())
              and re.fullmatch(r"[a-f0-9]{{64}}", str(data.get("archive_sha256") or "")) is not None
            )
            raise SystemExit(0 if valid else 1)
            PY
            then
              state=FULL_DATA_READY
              detail="official Kaggle archive, safe extraction, SHA inventory and real loaders verified"
            elif test -f "$ROOT/.evomind/download-job/state.json"; then
              state=$(python3 - "$ROOT/.evomind/download-job/state.json" "$ROOT/.evomind/download-job/pid" <<'PY'
            import json, os, sys
            value=str(json.load(open(sys.argv[1], encoding="utf-8")).get("status") or "RUNNING_OR_PARTIAL")
            if value == "RUNNING":
              try:
                pid=int(open(sys.argv[2], encoding="utf-8").read().strip())
                os.kill(pid, 0)
              except (OSError, TypeError, ValueError):
                value="DOWNLOAD_FAILED"
            print(value if value in {{"RUNNING","RUNNING_OR_PARTIAL","HUMAN_GATE_REQUIRED","AUTH_FAILED","DOWNLOAD_FAILED","VALIDATION_FAILED","SOURCE_UNAVAILABLE"}} else "RUNNING_OR_PARTIAL")
            PY
              )
              if test "$state" = "DOWNLOAD_FAILED"; then detail="persistent Kaggle worker is no longer alive"; else detail="persistent Kaggle worker state inspected"; fi
            elif test -s "$ROOT/.evomind/download-job/archive.partial"; then
              state=RUNNING_OR_PARTIAL
              detail="resumable official archive bytes are present"
            fi
            write_receipt "$state" "$detail"
            """
        )
    elif competition == "weather4cast":
        check = textwrap.dedent(
            """\
            evaluation=$(python3 - "$ROOT" <<'PY'
            import hashlib, json, os, pathlib, re, sys

            root = pathlib.Path(sys.argv[1])
            job = root / ".evomind" / "download-job"
            selector_path = job / "current-attempt.json"
            manifest_path = root / ".evomind" / "data-manifest.json"
            attempt_id = ""
            attempt_root = job
            selector_valid = False
            selector_present = selector_path.is_file()
            if selector_present:
              try:
                selector = json.loads(selector_path.read_text(encoding="utf-8"))
                attempt_id = str(selector.get("attempt_id") or "")
                selector_valid = re.fullmatch(r"[a-f0-9]{32}", attempt_id) is not None
                if selector_valid:
                  attempt_root = job / "attempts" / attempt_id
              except Exception:
                selector_valid = False

            state = {}
            state_path = attempt_root / "state.json"
            if state_path.is_file():
              try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                  state = loaded
              except Exception:
                state = {}

            pid_path = attempt_root / "pid"
            worker_alive = False
            if pid_path.is_file():
              try:
                os.kill(int(pid_path.read_text(encoding="utf-8").strip()), 0)
                worker_alive = True
              except (OSError, TypeError, ValueError):
                worker_alive = False

            exit_code = None
            exit_path = attempt_root / "exit-code"
            if exit_path.is_file():
              try:
                exit_code = int(exit_path.read_text(encoding="utf-8").strip())
              except (OSError, TypeError, ValueError):
                exit_code = None

            files = []
            data_root = root / "official"
            if data_root.is_dir():
              files = [path for path in data_root.rglob("*") if path.is_file() and not path.is_symlink()]
            observed_bytes = sum(path.stat().st_size for path in files)
            actual_files = {path.relative_to(root).as_posix(): path for path in files}

            def file_sha256(path):
              value = hashlib.sha256()
              with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                  value.update(chunk)
              return value.hexdigest()

            valid_manifest = False
            if manifest_path.is_file():
              try:
                raw = manifest_path.read_bytes()
                manifest = json.loads(raw)
                rows = manifest.get("files")
                smoke = manifest.get("loader_smoke")
                paths = [item.get("path") for item in rows] if isinstance(rows, list) else []
                remote_paths = [
                  value[len("official/"):]
                  for value in paths
                  if isinstance(value, str) and value.startswith("official/")
                ]
                observed_remote_file_set_sha256 = (
                  hashlib.sha256(
                    ("\\n".join(sorted(remote_paths, key=str.casefold)) + "\\n").encode("utf-8")
                  ).hexdigest()
                  if len(remote_paths) == 108
                  else ""
                )
                skipped_directory_count = manifest.get("skipped_directory_count")
                skipped_count_valid = (
                  isinstance(skipped_directory_count, int)
                  and not isinstance(skipped_directory_count, bool)
                  and 0 <= skipped_directory_count <= 1000
                )
                safe_segment = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+()@=,~ \\-]{0,254}")
                safe_paths = (
                  isinstance(rows, list)
                  and all(
                    isinstance(item, dict)
                    and isinstance(item.get("path"), str)
                    and "\\\\" not in item.get("path", "")
                    and "\\r" not in item.get("path", "")
                    and "\\n" not in item.get("path", "")
                    and not pathlib.PurePosixPath(item.get("path", "")).is_absolute()
                    and ".." not in pathlib.PurePosixPath(item.get("path", "")).parts
                    and pathlib.PurePosixPath(item.get("path", "")).parts[:1] == ("official",)
                    and item.get("path") == pathlib.PurePosixPath(item.get("path", "")).as_posix()
                    and 2 <= len(pathlib.PurePosixPath(item.get("path", "")).parts) <= 17
                    and all(
                      safe_segment.fullmatch(part) is not None
                      for part in pathlib.PurePosixPath(item.get("path", "")).parts[1:]
                    )
                    for item in rows
                  )
                )
                rows_valid = (
                  isinstance(rows, list)
                  and len(rows) == 108
                  and len(set(paths)) == 108
                  and len({value.casefold() for value in paths if isinstance(value, str)}) == 108
                  and safe_paths
                  and all(
                    isinstance(item, dict)
                    and isinstance(item.get("path"), str)
                    and item.get("path", "").startswith("official/")
                    and isinstance(item.get("bytes"), int)
                    and item.get("bytes") > 0
                    and re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256") or "")) is not None
                    for item in rows
                  )
                )
                disk_valid = (
                  rows_valid
                  and set(actual_files) == set(paths)
                  and all(
                    actual_files[item["path"]].stat().st_size == item["bytes"]
                    and file_sha256(actual_files[item["path"]]) == item["sha256"]
                    for item in rows
                  )
                )
                smoke_valid = (
                  isinstance(smoke, dict)
                  and set(smoke) == set(paths)
                  and all(isinstance(item, dict) and item.get("ok") is True for item in smoke.values())
                )
                identity_valid = (
                  manifest.get("schema") == "evomind.competition_data_manifest.v1"
                  and manifest.get("competition") == "weather4cast"
                  and manifest.get("expected_file_count") == 108
                  and manifest.get("file_count") == 108
                  and manifest.get("total_bytes") == observed_bytes
                  and manifest.get("total_bytes") == sum(item["bytes"] for item in rows)
                  and manifest.get("duplicate_files") == len(rows) - len({item["sha256"] for item in rows})
                  and skipped_count_valid
                  and skipped_directory_count == 0
                  and manifest.get("remote_file_set_sha256") == observed_remote_file_set_sha256
                  and re.fullmatch(
                    r"[a-f0-9]{64}", str(manifest.get("remote_file_set_sha256") or "")
                  ) is not None
                  and manifest.get("rules_accepted") is False
                  and manifest.get("secret_values_logged") is False
                  and rows_valid
                  and disk_valid
                  and smoke_valid
                )
                if selector_valid:
                  identity_valid = (
                    identity_valid
                    and manifest.get("attempt_id") == attempt_id
                    and state.get("attempt_id") == attempt_id
                    and state.get("status") == "FULL_DATA_READY"
                    and state.get("phase") == "complete"
                    and not state.get("failure_code")
                    and state.get("adapter_listing_status") == "ok"
                    and isinstance(state.get("adapter_listing_skipped_directories"), int)
                    and not isinstance(state.get("adapter_listing_skipped_directories"), bool)
                    and state.get("adapter_listing_skipped_directories") == skipped_directory_count
                    and isinstance(state.get("adapter_listing_timeout_exhaustions"), int)
                    and not isinstance(state.get("adapter_listing_timeout_exhaustions"), bool)
                    and state.get("adapter_listing_timeout_exhaustions") == skipped_directory_count
                    and state.get("manifest_sha256") == hashlib.sha256(raw).hexdigest()
                  )
                elif selector_present:
                  identity_valid = False
                valid_manifest = identity_valid
              except Exception:
                valid_manifest = False

            if valid_manifest:
              status, detail = "FULL_DATA_READY", "current manifest, 108 file hashes and 108 real loaders verified"
            else:
              state_status = str(state.get("status") or "")
              failure_code = str(state.get("failure_code") or "")
              if selector_present and not selector_valid:
                status, detail = "VALIDATION_FAILED", "current attempt selector is invalid"
              elif state_status == "RUNNING" and worker_alive:
                status, detail = "RUNNING", "current attempt worker is alive"
              elif state_status in {"DOWNLOAD_FAILED", "VALIDATION_FAILED", "START_FAILED"}:
                status, detail = state_status, "current attempt reached a terminal failure state"
              elif failure_code.startswith("SFTP_"):
                status, detail = "DOWNLOAD_FAILED", "current attempt recorded a redacted SFTP failure"
              elif exit_code not in (None, 0):
                status, detail = "DOWNLOAD_FAILED", "current attempt worker exited nonzero"
              elif worker_alive and observed_bytes > 0:
                status, detail = "RUNNING_OR_PARTIAL", "current attempt worker is alive with partial files"
              elif worker_alive:
                status, detail = "RUNNING", "current attempt worker is alive"
              elif selector_valid or observed_bytes > 0:
                status, detail = "DOWNLOAD_FAILED", "current attempt is not alive and has no valid completion manifest"
              else:
                status, detail = "NOT_STARTED", "no current Weather4cast attempt exists"
            print(status)
            print(detail)
            PY
            )
            state=${evaluation%%$'\\n'*}
            detail=${evaluation#*$'\\n'}
            write_receipt "$state" "$detail"
            """
        )
    else:
        check = textwrap.dedent(
            """\
            if test -f "$ROOT/.evomind/data-manifest.json"; then state=PARTIAL; else state=NOT_STARTED; fi
            write_receipt "$state" "official source is catalogued; full-data readiness needs its source-specific gate"
            """
        )
    return header + check


def _mindgames_prepare_script() -> str:
    smoke_rows = json.dumps(_MINDGAMES_ENVIRONMENTS)
    dependency_rows = json.dumps(_MINDGAMES_NLTK_DEPENDENCIES)
    dependency_manifest = json.dumps(_MINDGAMES_RUNTIME_DEPENDENCIES, sort_keys=True)
    inventory = _inventory_python(
        loader_expression="smoke",
        extra=(
            "payload['expected_environment_count'] = 4; "
            f"payload['textarena_version'] = {_MINDGAMES_TEXTARENA_VERSION!r}; "
            f"payload['runtime_dependencies'] = json.loads({dependency_manifest!r})"
        ),
    )
    body = textwrap.dedent(
        f"""\
        adapter_failed() {{
          rc=$?
          trap - ERR
          write_receipt START_FAILED "MindGames adapter command failed before loader validation"
          exit "$rc"
        }}
        trap adapter_failed ERR
        SOURCE="$ROOT/starter-kit"
        STAGE="$ROOT/.evomind/starter-kit.next"
        if test -d "$SOURCE/.git"; then
          git -C "$SOURCE" fetch --depth=1 origin main
          git -C "$SOURCE" merge --ff-only origin/main
        elif test -e "$SOURCE"; then
          write_receipt SOURCE_CONFLICT "starter-kit exists without official git metadata"
          exit 4
        else
          rm -rf "$STAGE"
          git clone --depth=1 https://github.com/mind-games-challenge/mindgames-starter-kit.git "$STAGE"
          mv "$STAGE" "$SOURCE"
        fi
        RUNTIME="$EVOMIND_COMPETITION_DATA_ROOT/../.runtime/mindgames"
        python3 -m venv "$RUNTIME"
        "$RUNTIME/bin/python" -m pip install --disable-pip-version-check --no-input 'textarena=={_MINDGAMES_TEXTARENA_VERSION}'
        NLTK_ROOT="$RUNTIME/nltk_data"
        mkdir -p "$NLTK_ROOT/corpora" "$NLTK_ROOT/taggers"
        "$RUNTIME/bin/python" - "$NLTK_ROOT" <<'PY'
        import hashlib, os, pathlib, sys, time, urllib.request
        root = pathlib.Path(sys.argv[1])
        dependencies = {dependency_rows}
        for relative, source, expected_sha256 in dependencies:
          target = root / relative
          target.parent.mkdir(parents=True, exist_ok=True)
          if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
            continue
          temporary = target.with_suffix(target.suffix + ".tmp")
          error = None
          for attempt in range(3):
            try:
              with urllib.request.urlopen(source, timeout=120) as response, temporary.open("wb") as handle:
                while True:
                  chunk = response.read(1024 * 1024)
                  if not chunk:
                    break
                  handle.write(chunk)
              if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_sha256:
                raise ValueError("dependency hash mismatch")
              os.replace(temporary, target)
              error = None
              break
            except Exception as exc:
              error = exc
              temporary.unlink(missing_ok=True)
              if attempt < 2:
                time.sleep(2 ** attempt)
          if error is not None:
            raise SystemExit(6)
        PY
        NLTK_DATA="$NLTK_ROOT" "$RUNTIME/bin/python" - "$ROOT/.evomind/loader-smoke.json" <<'PY'
        import json, sys
        import textarena as ta
        environments = {smoke_rows}
        results = {{}}
        for env_id, players in environments:
          try:
            env = ta.make(env_id=env_id)
            env.reset(num_players=players)
            player_id, observation = env.get_observation()
            results[env_id] = {{"ok": isinstance(observation, str), "player_id": int(player_id), "environment_class": type(env).__name__}}
            env.close()
          except Exception as exc:
            results[env_id] = {{"ok": False, "error_class": type(exc).__name__}}
        with open(sys.argv[1], "w", encoding="utf-8") as handle:
          json.dump(results, handle, sort_keys=True)
          handle.write("\\n")
        PY
        __EVOMIND_MINDGAMES_INVENTORY__
        if "$RUNTIME/bin/python" - "$ROOT/.evomind/loader-smoke.json" <<'PY'
        import json, sys
        results = json.load(open(sys.argv[1], encoding="utf-8"))
        raise SystemExit(0 if len(results) == 4 and all(item.get("ok") is True for item in results.values()) else 1)
        PY
        then
          write_receipt FULL_DATA_READY "official starter kit plus four distinct environment loaders verified"
        else
          write_receipt VALIDATION_FAILED "one or more official MindGames environment loaders failed"
        fi
        trap - ERR
        """
    )
    return _script_header("mindgames") + body.replace(
        "__EVOMIND_MINDGAMES_INVENTORY__",
        inventory.replace(
            "loader_smoke = smoke",
            'loader_smoke = json.loads((root / ".evomind" / "loader-smoke.json").read_text(encoding="utf-8"))',
        ),
    )


def _e2lmc_prepare_script() -> str:
    rows = "\n".join(
        f"download {name!r} {file_id!r}"
        for name, file_id in _E2LMC_NOTEBOOKS
    )
    inventory = _inventory_python(
        loader_expression="{item['path']: {'ok': True, 'kind': 'jupyter_notebook'} for item in rows}",
        extra="payload['valid_unique_notebooks'] = len({item['sha256'] for item in rows if item['path'].endswith('.ipynb')})",
    )
    adapter = """mkdir -p "$ROOT/notebooks" "$ROOT/.evomind/downloads"
download() {
  name="$1"; file_id="$2"; target="$ROOT/notebooks/$name"; partial="$ROOT/.evomind/downloads/$name.part"
  curl --fail --location --retry 3 --retry-delay 2 --continue-at - \
    "https://drive.usercontent.google.com/download?id=$file_id&export=download&confirm=t" -o "$partial"
  python3 - "$partial" <<'PY'
import json, sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
assert isinstance(data.get("cells"), list) and isinstance(data.get("nbformat"), int)
PY
  mv "$partial" "$target"
}
"""
    return (
        _script_header("e2lmc")
        + adapter
        + rows
        + "\n"
        + inventory
        + 'write_receipt FULL_DATA_READY "eight unique official notebooks validated as Jupyter documents"\n'
    )


def _kaggle_prepare_script(competition: str) -> str:
    slug = _KAGGLE_COMPETITIONS[competition]
    official_url = COMPETITION_CATALOG[competition]["source"]
    body = r'''\
SECRET="${EVOMIND_SECRET_KAGGLE_FILE:-}"
if test -z "$SECRET" || test ! -s "$SECRET"; then
  write_receipt SECRET_REQUIRED "use the controlled Kaggle credential field; never put an API token in the prompt"
  exit 6
fi
JOB="$ROOT/.evomind/download-job"
mkdir -p "$JOB"
if test -s "$JOB/pid" && kill -0 "$(cat "$JOB/pid")" 2>/dev/null; then
  write_receipt RUNNING "attached to existing resumable official Kaggle download"
  exit 0
fi
cat > "$JOB/worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"; SECRET="$2"; INTERNAL="$3"; SLUG="$4"; OFFICIAL_URL="$5"
JOB="$ROOT/.evomind/download-job"
exec 9<"$SECRET"
rm -f "$SECRET"
: > "$JOB/secret-consumed"
set +e
python3 - /dev/fd/9 "$ROOT" "$INTERNAL" "$SLUG" "$OFFICIAL_URL" <<'PY' > /dev/null 2>&1
import base64, csv, fcntl, gzip, hashlib, importlib, io, json, os, pathlib, re, shutil, subprocess, sys, tarfile, time, urllib.error, urllib.request, zipfile

secret_path, root_text, internal, slug, official_url = sys.argv[1:]
root = pathlib.Path(root_text)
job = root / ".evomind" / "download-job"
state_path = job / "state.json"
partial = job / "archive.partial"
archive_dir = root / "official"
archive_path = archive_dir / f"{slug}.zip"
data_dir = root / "data"
stage_dir = job / "data.next"
sha_re = re.compile(r"^[a-f0-9]{64}$")

def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def state(status, phase, failure_code="", rules_accepted=False):
    atomic_json(state_path, {
        "schema": "evomind.competition_download_state.v1",
        "status": status,
        "phase": phase,
        "failure_code": failure_code,
        "official_url": official_url,
        "source_competition_slug": slug,
        "rules_accepted": rules_accepted,
        "secret_values_logged": False,
    })

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

class WorkerStop(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code

def fail_http(code):
    if code == 401:
        raise WorkerStop("AUTH_FAILED", "kaggle_auth_failed")
    if code == 403:
        raise WorkerStop("HUMAN_GATE_REQUIRED", "kaggle_rules_or_access_required")
    if code == 404:
        raise WorkerStop("SOURCE_UNAVAILABLE", "official_competition_not_found")

def download(headers):
    archive_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file() and archive_path.stat().st_size > 0 and zipfile.is_zipfile(archive_path):
        return
    endpoint = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{slug}"
    delays = (1, 2, 4, 8, 16)
    for attempt, delay in enumerate(delays, 1):
        current = partial.stat().st_size if partial.is_file() else 0
        request_headers = {"User-Agent": "EvoMind/0.4", **headers}
        if current:
            request_headers["Range"] = f"bytes={current}-"
        request = urllib.request.Request(endpoint, headers=request_headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status_code = int(getattr(response, "status", 200) or 200)
                content_length_text = str(response.headers.get("Content-Length") or "").strip()
                if content_length_text and not content_length_text.isdigit():
                    raise WorkerStop("DOWNLOAD_FAILED", "kaggle_range_contract_invalid")
                content_length = int(content_length_text) if content_length_text.isdigit() else None
                expected_final = None
                if current:
                    if status_code != 206:
                        raise WorkerStop("DOWNLOAD_FAILED", "kaggle_range_not_honored")
                    matched = re.fullmatch(
                        r"bytes\s+(\d+)-(\d+)/(\d+|\*)",
                        str(response.headers.get("Content-Range") or "").strip(),
                    )
                    if matched is None:
                        raise WorkerStop("DOWNLOAD_FAILED", "kaggle_range_contract_invalid")
                    start, end = int(matched.group(1)), int(matched.group(2))
                    total = None if matched.group(3) == "*" else int(matched.group(3))
                    if (
                        start != current
                        or end < start
                        or (total is not None and total <= end)
                        or (content_length is not None and content_length != end - start + 1)
                    ):
                        raise WorkerStop("DOWNLOAD_FAILED", "kaggle_range_contract_invalid")
                    expected_final = total if total is not None else (current + content_length if content_length is not None else None)
                    mode = "ab"
                else:
                    if status_code == 206:
                        matched = re.fullmatch(
                            r"bytes\s+(\d+)-(\d+)/(\d+|\*)",
                            str(response.headers.get("Content-Range") or "").strip(),
                        )
                        if matched is None:
                            raise WorkerStop("DOWNLOAD_FAILED", "kaggle_range_contract_invalid")
                        start, end = int(matched.group(1)), int(matched.group(2))
                        total = None if matched.group(3) == "*" else int(matched.group(3))
                        if (
                            start != 0
                            or end < start
                            or (total is not None and total <= end)
                            or (content_length is not None and content_length != end - start + 1)
                        ):
                            raise WorkerStop("DOWNLOAD_FAILED", "kaggle_range_contract_invalid")
                        expected_final = total if total is not None else content_length
                    elif status_code != 200:
                        raise WorkerStop("DOWNLOAD_FAILED", f"kaggle_http_{status_code}")
                    else:
                        expected_final = content_length
                    mode = "wb"
                written = 0
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
            observed = partial.stat().st_size if partial.is_file() else 0
            base = current if mode == "ab" else 0
            response_complete = content_length is None or written == content_length
            total_complete = expected_final in (None, 0) or observed == expected_final
            if expected_final not in (None, 0) and observed > expected_final:
                raise WorkerStop("DOWNLOAD_FAILED", "kaggle_range_contract_invalid")
            if observed > 0 and response_complete and total_complete and zipfile.is_zipfile(partial):
                os.replace(partial, archive_path)
                return
            if attempt == len(delays):
                code = (
                    "kaggle_archive_no_progress" if observed <= base
                    else "kaggle_response_truncated" if not response_complete
                    else "kaggle_archive_incomplete" if not total_complete
                    else "official_archive_invalid_after_resume"
                )
                raise WorkerStop("DOWNLOAD_FAILED", code)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                matched = re.fullmatch(
                    r"bytes\s+\*/(\d+)",
                    str(exc.headers.get("Content-Range") or "").strip(),
                )
                if (
                    matched is not None
                    and partial.is_file()
                    and partial.stat().st_size == int(matched.group(1))
                    and zipfile.is_zipfile(partial)
                ):
                    os.replace(partial, archive_path)
                    return
                raise WorkerStop("DOWNLOAD_FAILED", "official_archive_corrupt")
            fail_http(exc.code)
            if exc.code not in {429, 500, 502, 503, 504} or attempt == len(delays):
                raise WorkerStop("DOWNLOAD_FAILED", f"kaggle_http_{exc.code}")
        except WorkerStop:
            raise
        except Exception:
            if attempt == len(delays):
                raise WorkerStop("DOWNLOAD_FAILED", "kaggle_network_exhausted")
        time.sleep(delay)

def safe_extract():
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True)
    members = 0
    with zipfile.ZipFile(archive_path) as source:
        for info in source.infolist():
            raw = info.filename.replace("\\", "/")
            relative = pathlib.PurePosixPath(raw)
            unix_type = (info.external_attr >> 16) & 0o170000
            if (not raw or raw.startswith("/") or "\x00" in raw or ":" in raw
                    or ".." in relative.parts or unix_type == 0o120000):
                raise WorkerStop("VALIDATION_FAILED", "unsafe_archive_member")
            target = stage_dir.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            members += 1
            if members > 250000:
                raise WorkerStop("VALIDATION_FAILED", "archive_file_count_exceeded")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer, 4 * 1024 * 1024)
    if members < 1:
        raise WorkerStop("VALIDATION_FAILED", "official_archive_empty")
    previous = job / "data.previous"
    shutil.rmtree(previous, ignore_errors=True)
    if data_dir.exists():
        os.replace(data_dir, previous)
    try:
        os.replace(stage_dir, data_dir)
    except Exception:
        if previous.exists() and not data_dir.exists():
            os.replace(previous, data_dir)
        raise
    shutil.rmtree(previous, ignore_errors=True)

vendor = root.parent / ".runtime" / "kaggle-loaders"
vendor.mkdir(parents=True, exist_ok=True)
if str(vendor) not in sys.path:
    sys.path.insert(0, str(vendor))

def dependency(module, specification):
    try:
        return importlib.import_module(module)
    except ImportError:
        lock_path = vendor.parent / "kaggle-loaders.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                importlib.invalidate_caches()
                try:
                    return importlib.import_module(module)
                except ImportError:
                    completed = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--target", str(vendor), specification],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200, check=False,
                    )
                    if completed.returncode != 0:
                        raise WorkerStop("VALIDATION_FAILED", "loader_dependency_install_failed")
                    importlib.invalidate_caches()
                    return importlib.import_module(module)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

def load(path):
    suffix = path.suffix.casefold()
    name = path.name.casefold()
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("empty file")
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            rows = csv.reader(handle, delimiter="\t" if suffix == ".tsv" else ",")
            first = next(rows)
            if not first:
                raise ValueError("empty header")
            next(rows, None)
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        parsed = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line); parsed += 1
                    if parsed == 3:
                        break
        if not parsed:
            raise ValueError("empty json lines")
        return "json_lines"
    if suffix in {".json", ".ipynb"}:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        return "json"
    if suffix == ".parquet":
        parquet = dependency("pyarrow.parquet", "pyarrow>=15,<24")
        reader = parquet.ParquetFile(path)
        reader.schema
        next(reader.iter_batches(batch_size=1), None)
        return "pyarrow.parquet"
    if suffix in {".npy", ".npz"}:
        numpy = dependency("numpy", "numpy>=1.26,<3")
        value = numpy.load(path, mmap_mode="r" if suffix == ".npy" else None, allow_pickle=False)
        if hasattr(value, "files"):
            list(value.files); value.close()
        return "numpy"
    if suffix in {".h5", ".hdf5"}:
        h5py = dependency("h5py", "h5py>=3.10,<4")
        with h5py.File(path, "r") as handle:
            list(handle.keys())
        return "h5py"
    if suffix in {".fits", ".fit", ".fts"}:
        fits = dependency("astropy.io.fits", "astropy>=6,<8")
        with fits.open(path, memmap=True) as handle:
            len(handle)
        return "astropy.fits"
    if suffix == ".zip":
        with zipfile.ZipFile(path) as handle:
            if handle.testzip() is not None:
                raise ValueError("nested zip CRC failure")
        return "zipfile"
    if suffix in {".tar", ".tgz"} or name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(path) as handle:
            handle.getmembers()
        return "tarfile"
    if suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            if not handle.read(1):
                raise ValueError("empty gzip payload")
        return "gzip"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        image = dependency("PIL.Image", "Pillow>=10,<13")
        with image.open(path) as handle:
            handle.verify()
        return "Pillow"
    if suffix in {".pkl", ".pickle", ".joblib", ".pt", ".pth"}:
        raise ValueError("unsafe serialized payload is not auto-loaded")
    if suffix in {".txt", ".md", ".yaml", ".yml", ".xml", ".smi", ".smiles", ".fasta", ".fa"}:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            if not handle.read(4096):
                raise ValueError("empty text")
        return "text"
    with path.open("rb") as handle:
        if not handle.read(16):
            raise ValueError("empty binary")
    return "binary_signature"

def manifest():
    rows, smoke = [], {}
    for path in sorted(data_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        item = {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}
        rows.append(item)
        try:
            smoke[relative] = {"ok": True, "loader": load(path)}
        except Exception as exc:
            smoke[relative] = {"ok": False, "error_class": type(exc).__name__}
    payload = {
        "schema": "evomind.competition_data_manifest.v1",
        "competition": internal,
        "source_competition_slug": slug,
        "official_url": official_url,
        "rules_accepted": True,
        "archive_path": archive_path.relative_to(root).as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": digest(archive_path),
        "archive_zip_ok": zipfile.is_zipfile(archive_path),
        "unsafe_member_count": 0,
        "file_count": len(rows),
        "total_bytes": sum(item["bytes"] for item in rows),
        "files": rows,
        "loader_smoke": smoke,
        "duplicate_files": len(rows) - len({item["sha256"] for item in rows}),
        "secret_values_logged": False,
    }
    atomic_json(root / ".evomind" / "data-manifest.json", payload)
    if not rows or len(smoke) != len(rows) or not all(item["ok"] for item in smoke.values()) or not sha_re.fullmatch(payload["archive_sha256"]):
        raise WorkerStop("VALIDATION_FAILED", "one_or_more_real_loaders_failed")

try:
    payload = json.load(open(secret_path, encoding="utf-8"))
    username = str(payload.get("username") or "").strip()
    token = str(payload.get("token") or "").strip()
    if (payload.get("schema") != "evomind.run_secret_payload.v1" or payload.get("purpose") != "kaggle_api"
            or not token or len(token) > 4096 or (username and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", username) is None)):
        raise WorkerStop("AUTH_FAILED", "controlled_kaggle_payload_invalid")
    if username:
        auth = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
        headers = {"Authorization": f"Basic {auth}"}
    else:
        headers = {"Authorization": f"Bearer {token}"}
    payload = None; token = ""; auth = ""
    state("RUNNING", "downloading", rules_accepted=False)
    download(headers)
    headers.clear()
    state("RUNNING", "extracting", rules_accepted=True)
    safe_extract()
    state("RUNNING", "validating", rules_accepted=True)
    manifest()
    state("FULL_DATA_READY", "complete", rules_accepted=True)
except WorkerStop as exc:
    state(exc.status, "blocked" if exc.status in {"HUMAN_GATE_REQUIRED", "AUTH_FAILED"} else "failed", exc.code)
    raise SystemExit(10)
except Exception as exc:
    state("VALIDATION_FAILED", "failed", type(exc).__name__)
    raise SystemExit(11)
PY
rc=$?
set -e
printf '%s\n' "$rc" > "$JOB/exit-code.tmp"; mv "$JOB/exit-code.tmp" "$JOB/exit-code"
exit "$rc"
WORKER
chmod 700 "$JOB/worker.sh"
rm -f "$JOB/secret-consumed" "$JOB/exit-code"
nohup setsid bash "$JOB/worker.sh" "$ROOT" "$SECRET" '__COMPETITION__' '__SLUG__' '__OFFICIAL_URL__' > /dev/null 2>&1 < /dev/null &
printf '%s\n' "$!" > "$JOB/pid.tmp"; mv "$JOB/pid.tmp" "$JOB/pid"
for attempt in $(seq 1 100); do test -f "$JOB/secret-consumed" && break; sleep 0.1; done
if test ! -f "$JOB/secret-consumed"; then
  write_receipt START_FAILED "background Kaggle worker did not consume the ephemeral credential"
  exit 8
fi
write_receipt RUNNING "official Kaggle download started; use competition_data_status for resumable progress"
'''
    return (
        _script_header(competition)
        + body.replace("__COMPETITION__", competition).replace("__SLUG__", slug).replace("__OFFICIAL_URL__", official_url)
    )


def _weather4cast_sftp_supports_short_option(usage_text: str, option: str) -> bool:
    """Recognize an OpenSSH short option in both plain and bracketed usage text."""

    if option not in {"-B", "-R"}:
        raise ValueError("unsupported Weather4cast SFTP tuning option")
    return re.search(rf"(?:^|[\s\[])({re.escape(option)})(?:[\s\]]|$)", str(usage_text or "")) is not None


def _weather4cast_listing_file_category(value: str) -> str:
    """Return one fixed, suffix-only category without exposing the supplied value."""

    folded = str(value or "").casefold()
    archive_suffixes = (
        ".tar", ".tar.gz", ".tgz", ".zip", ".gz", ".bz2", ".xz", ".7z",
    )
    scientific_suffixes = (
        ".nc", ".npy", ".npz", ".h5", ".hdf5", ".grib", ".grb",
    )
    metadata_suffixes = (
        ".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml",
        ".sha256", ".html",
    )
    if folded.endswith(archive_suffixes):
        return "archive"
    if folded.endswith(scientific_suffixes):
        return "scientific"
    if folded.endswith(metadata_suffixes):
        return "metadata"
    return "other"


def _weather4cast_listing_profile(
    remote_files: list[str],
    local_directories: set[str],
) -> dict[str, int]:
    """Build aggregate-only listing telemetry for the bounded worker state."""

    categories = {
        "archive": 0,
        "scientific": 0,
        "metadata": 0,
        "other": 0,
    }
    maximum_depth = 0
    for relative in remote_files:
        categories[_weather4cast_listing_file_category(relative)] += 1
        maximum_depth = max(maximum_depth, len(pathlib.PurePosixPath(relative).parts))
    for relative in local_directories:
        maximum_depth = max(maximum_depth, len(pathlib.PurePosixPath(relative).parts))
    return {
        "adapter_listing_entries": len(remote_files) + len(local_directories),
        "adapter_listing_files": len(remote_files),
        "adapter_listing_directories": len(local_directories),
        "adapter_listing_archive_files": categories["archive"],
        "adapter_listing_scientific_files": categories["scientific"],
        "adapter_listing_metadata_files": categories["metadata"],
        "adapter_listing_other_files": categories["other"],
        "adapter_listing_max_depth": maximum_depth,
    }


def _weather4cast_nonroot_listing_timeout_is_skippable(
    listing_failure: str,
    current_directory: str,
) -> bool:
    """Allow only an exhausted non-root listing timeout to keep the BFS alive."""

    return listing_failure == "SFTP_TIMEOUT" and bool(current_directory)


def _weather4cast_remote_file_set_sha256(remote_files: list[str]) -> str:
    """Hash the exact 108-file remote set or reject incomplete listing evidence."""

    ordered = sorted(remote_files, key=str.casefold)
    if len(ordered) != 108 or len({value.casefold() for value in ordered}) != 108:
        raise ValueError("Weather4cast remote file set is not exactly 108 unique files")
    return hashlib.sha256(("\n".join(ordered) + "\n").encode("utf-8")).hexdigest()


def _weather4cast_local_file_set_matches_remote(
    local_files: list[str],
    remote_files: list[str],
) -> bool:
    """Require the downloaded relative paths to equal the current remote listing."""

    return len(local_files) == len(remote_files) and set(local_files) == set(remote_files)


def _parse_weather4cast_sftp_long_listing(
    stdout: str,
    remote_root: str,
    listing_root: str = "",
) -> tuple[list[tuple[str, str]], dict[str, int | str]]:
    """Parse one bounded OpenSSH long listing without returning paths in telemetry."""

    entries: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    canonical_parent_prefix: tuple[str, ...] | None = None
    diagnostics: dict[str, int | str] = {
        "raw_lines": 0,
        "control_lines": 0,
        "rejected_lines": 0,
        "files": 0,
        "directories": 0,
    }

    def reject(reason: str) -> None:
        diagnostics["rejected_lines"] = int(diagnostics["rejected_lines"]) + 1
        diagnostics.setdefault("reject_reason", reason)

    controlled_prefix = str(remote_root or "").rstrip("/") + "/"
    if controlled_prefix == "//":
        controlled_prefix = "/"
    listing_prefix = str(listing_root or "").strip("/")
    if listing_prefix:
        listing_prefix += "/"
    safe_leaf = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+()@=,~\- ]{0,254}")
    control_prefixes = (
        "connected to ",
        "changing to: ",
        "remote working directory: ",
    )
    for raw_line in str(stdout or "").splitlines():
        value = raw_line.strip()
        if not value:
            continue
        diagnostics["raw_lines"] += 1
        folded = value.casefold()
        if value.startswith("sftp>") or folded.startswith(control_prefixes):
            diagnostics["control_lines"] += 1
            continue
        if re.fullmatch(r"total [0-9]+", value) is not None:
            # Some SFTP servers include the POSIX aggregate block-count line
            # before a long directory listing.  It carries no path entry and
            # is safe to ignore only in this exact numeric form.
            diagnostics["control_lines"] += 1
            continue
        if folded == "total" or folded.startswith("total "):
            reject("total_like_format")
            continue
        if value.endswith(":"):
            header = value[:-1]
            while header.startswith("./"):
                header = header[2:]
            header_parts = pathlib.PurePosixPath(header).parts
            if header_parts and all(part not in {"", ".", ".."} and "\\" not in part for part in header_parts):
                diagnostics["control_lines"] += 1
                continue
        fields = value.split(None, 8)
        # SFTP ``longname`` metadata is server-defined.  Some compliant
        # servers ignore ``-n`` and emit opaque mode/owner/group/link/size
        # tokens.  None of those fields are used for authorization or path
        # construction here: accept only regular-file or directory rows by
        # their leading type marker, then apply the strict direct-leaf and
        # canonical-parent checks below.  Symlinks and special files retain
        # distinct leading markers and remain fail-closed.
        if not fields or not fields[0]:
            reject("field_count")
            continue
        entry_type = fields[0][0]
        if entry_type == "l":
            # Never follow or download remote symlinks.  They are not part of
            # the regular-file manifest; the final exact file-count and hash
            # gates still prove completeness of the official payload.
            diagnostics["control_lines"] += 1
            continue
        if entry_type in {"b", "c", "p", "s"}:
            reject("special_type")
            continue
        if entry_type not in {"-", "d"}:
            reject("unknown_type")
            continue
        if len(fields) != 9:
            reject("field_count")
            continue
        kind = "file" if entry_type == "-" else "directory"
        leaf = fields[8].strip()
        while leaf.startswith("./"):
            leaf = leaf[2:]
        if (
            controlled_prefix == "/"
            and not listing_prefix
            and leaf.startswith("/")
            and "/" not in leaf[1:]
        ):
            leaf = leaf[1:]
        elif controlled_prefix != "/" and controlled_prefix and leaf.startswith(controlled_prefix):
            leaf = leaf[len(controlled_prefix):]
        if listing_prefix and leaf.startswith(listing_prefix):
            leaf = leaf[len(listing_prefix):]
        elif listing_prefix and leaf.startswith("/") and "\\" not in leaf:
            # OpenSSH may echo a match after ``cd``/``ls .`` with the server's
            # canonical absolute root rather than the path supplied to
            # ``cd``.  The canonical prefix is intentionally
            # opaque.  Accept only a direct child of the exact bounded
            # listing root; the worker still reconstructs and downloads
            # ``listing_root/leaf`` instead of trusting the echoed prefix.
            canonical_parts = tuple(
                part for part in pathlib.PurePosixPath(leaf).parts
                if part not in {"", "/"}
            )
            listing_parts = pathlib.PurePosixPath(listing_root.strip("/")).parts
            if (
                listing_parts
                and all(part not in {".", ".."} and "\\" not in part for part in canonical_parts)
                and len(canonical_parts) > len(listing_parts)
                and canonical_parts[-len(listing_parts) - 1:-1] == listing_parts
            ):
                candidate_parent = canonical_parts[:-len(listing_parts) - 1]
                if canonical_parent_prefix is None:
                    canonical_parent_prefix = candidate_parent
                if candidate_parent == canonical_parent_prefix:
                    leaf = canonical_parts[-1]
        if not leaf or leaf in {".", ".."} or "/" in leaf or "\\" in leaf:
            reject("path")
            continue
        if leaf.startswith("-"):
            reject("leading_dash")
            continue
        if ":" in leaf:
            reject("colon")
            continue
        if "[" in leaf or "]" in leaf:
            reject("bracket")
            continue
        if "%" in leaf:
            reject("percent")
            continue
        if "&" in leaf:
            reject("ampersand")
            continue
        if "#" in leaf:
            reject("hash")
            continue
        if safe_leaf.fullmatch(leaf) is None:
            if len(leaf) > 255:
                reject("leaf_length")
            elif "'" in leaf:
                reject("apostrophe")
            elif '"' in leaf:
                reject("double_quote")
            elif ";" in leaf:
                reject("semicolon")
            elif "!" in leaf:
                reject("exclamation")
            elif "$" in leaf:
                reject("dollar")
            elif "`" in leaf:
                reject("backtick")
            elif "{" in leaf or "}" in leaf:
                reject("brace")
            elif "*" in leaf:
                reject("asterisk")
            elif "?" in leaf:
                reject("question")
            elif any(ord(character) < 32 or ord(character) == 127 for character in leaf):
                reject("control_character")
            elif any(ord(character) > 127 for character in leaf):
                reject("unicode_character")
            else:
                reject("other_ascii")
            continue
        folded_leaf = leaf.casefold()
        if folded_leaf in seen:
            reject("duplicate")
            continue
        seen[folded_leaf] = kind
        entries.append((kind, leaf))
        diagnostics["files" if kind == "file" else "directories"] += 1
    entries.sort(key=lambda item: item[1].casefold())
    return entries, diagnostics


def _weather4cast_prepare_script(*, accelerate: bool = False) -> str:
    raw = _script_header("weather4cast") + textwrap.dedent(
        """\
        ACCELERATE=__EVOMIND_ACCELERATE__
        SECRET="${EVOMIND_SECRET_WEATHER4CAST_FILE:-}"
        if test -z "$SECRET" || test ! -s "$SECRET"; then
          write_receipt SECRET_REQUIRED "use the controlled Weather4cast credential panel; do not put credentials in the prompt"
          exit 6
        fi
        if ! command -v flock >/dev/null 2>&1 || ! command -v setsid >/dev/null 2>&1; then
          write_receipt DEPENDENCY_MISSING "flock and setsid are required for a single attempt-scoped worker"
          exit 7
        fi
        if ! command -v sftp >/dev/null 2>&1; then
          write_receipt DEPENDENCY_MISSING "OpenSSH sftp is required for exact remote file-set verification"
          exit 7
        fi
        if test "$ACCELERATE" = "1"; then
          exec 7<"$SECRET"
          if ! python3 - /dev/fd/7 <<'PY'
        import json, re, sys
        try:
          data = json.load(open(sys.argv[1], encoding="utf-8"))
          valid = (
            data.get("schema") == "evomind.run_secret_payload.v1"
            and data.get("purpose") == "weather4cast_sftp"
            and re.fullmatch(r"[A-Za-z0-9.-]{1,253}", str(data.get("host") or "")) is not None
            and re.fullmatch(r"[A-Za-z0-9_.@+-]{1,256}", str(data.get("username") or "")) is not None
            and 0 < len(str(data.get("password") or "")) <= 4096
            and 1 <= int(data.get("port") or 22) <= 65535
            and str(data.get("remote_path") or "/").startswith("/")
            and str(data.get("target_subdir") or "official") == "official"
          )
        except Exception:
          valid = False
        raise SystemExit(0 if valid else 1)
        PY
          then
            exec 7<&-
            write_receipt START_FAILED "controlled Weather4cast acceleration credential failed preflight"
            exit 12
          fi
          exec 7<&-
        fi
        JOB="$ROOT/.evomind/download-job"
        mkdir -p "$JOB/attempts"
        if test "$ACCELERATE" = "1"; then
          exec 6>"$JOB/accelerate.lock"
          if ! flock -n 6; then
            write_receipt RUNNING "another Weather4cast acceleration owns the cutover lock"
            exit 0
          fi
          ACCELERATE_PID=$(python3 - "$JOB" <<'PY'
        import json, pathlib, re, sys
        job = pathlib.Path(sys.argv[1]); selector = job / "current-attempt.json"; pid_path = job / "pid"
        if selector.is_file():
          try:
            value = json.loads(selector.read_text(encoding="utf-8")); attempt_id = str(value.get("attempt_id") or "")
            if re.fullmatch(r"[a-f0-9]{32}", attempt_id):
              pid_path = job / "attempts" / attempt_id / "pid"
          except Exception:
            pass
        if pid_path.is_file():
          try:
            print(int(pid_path.read_text(encoding="utf-8").strip()))
          except (OSError, TypeError, ValueError):
            pass
        PY
          )
          if test -n "$ACCELERATE_PID" && kill -0 "$ACCELERATE_PID" 2>/dev/null; then
            if ! python3 - "$JOB" "$ACCELERATE_PID" <<'PY'
        import json, pathlib, re, sys
        job = pathlib.Path(sys.argv[1]); pid = int(sys.argv[2])
        try:
          selector = json.loads((job / "current-attempt.json").read_text(encoding="utf-8")); attempt_id = str(selector.get("attempt_id") or "")
          pid_text = (job / "attempts" / attempt_id / "pid").read_text(encoding="utf-8").strip()
          raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8"); fields = raw[raw.rfind(")") + 2:].split()
          command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\\0")
          valid = (
            selector.get("schema") == "evomind.weather4cast_current_attempt.v1"
            and re.fullmatch(r"[a-f0-9]{32}", attempt_id) is not None
            and int(selector.get("pid") or 0) == pid and int(pid_text) == pid
            and int(fields[2]) == pid and int(fields[3]) == pid
            and any(b"worker.sh" in item for item in command) and attempt_id.encode("ascii") in command
          )
        except Exception:
          valid = False
        raise SystemExit(0 if valid else 1)
        PY
            then
              write_receipt START_FAILED "existing Weather4cast worker identity could not be verified before lock recovery"
              exit 14
            fi
            if ! kill -TERM -- "-$ACCELERATE_PID" 2>/dev/null; then
              write_receipt START_FAILED "existing Weather4cast worker could not be stopped before lock recovery"
              exit 15
            fi
            for attempt in $(seq 1 100); do
              if ! kill -0 "$ACCELERATE_PID" 2>/dev/null; then break; fi
              sleep 0.2
            done
            if kill -0 "$ACCELERATE_PID" 2>/dev/null; then
              write_receipt START_FAILED "existing Weather4cast worker did not release the inherited lock"
              exit 16
            fi
          fi
        fi
        exec 8>"$JOB/start.lock"
        if ! flock -n 8; then
          write_receipt RUNNING "another Weather4cast prepare owns the single-worker lock"
          exit 0
        fi
        EXISTING_PID=$(python3 - "$JOB" <<'PY'
        import json, pathlib, re, sys
        job = pathlib.Path(sys.argv[1])
        selector = job / "current-attempt.json"
        pid_path = job / "pid"
        if selector.is_file():
          try:
            value = json.loads(selector.read_text(encoding="utf-8"))
            attempt_id = str(value.get("attempt_id") or "")
            if re.fullmatch(r"[a-f0-9]{32}", attempt_id):
              pid_path = job / "attempts" / attempt_id / "pid"
          except Exception:
            pass
        if pid_path.is_file():
          try:
            print(int(pid_path.read_text(encoding="utf-8").strip()))
          except (OSError, TypeError, ValueError):
            pass
        PY
        )
        if test -n "$EXISTING_PID" && kill -0 "$EXISTING_PID" 2>/dev/null; then
          if test "$ACCELERATE" != "1"; then
            write_receipt RUNNING "attached to existing resumable Weather4cast mirror"
            exit 0
          fi
          if ! python3 - "$JOB" "$EXISTING_PID" <<'PY'
        import json, pathlib, re, sys
        job = pathlib.Path(sys.argv[1]); pid = int(sys.argv[2])
        try:
          selector = json.loads((job / "current-attempt.json").read_text(encoding="utf-8"))
          attempt_id = str(selector.get("attempt_id") or "")
          pid_text = (job / "attempts" / attempt_id / "pid").read_text(encoding="utf-8").strip()
          raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
          fields = raw[raw.rfind(")") + 2:].split()
          command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\\0")
          valid = (
            selector.get("schema") == "evomind.weather4cast_current_attempt.v1"
            and re.fullmatch(r"[a-f0-9]{32}", attempt_id) is not None
            and int(selector.get("pid") or 0) == pid
            and int(pid_text) == pid
            and int(fields[2]) == pid
            and int(fields[3]) == pid
            and any(b"worker.sh" in item for item in command)
            and attempt_id.encode("ascii") in command
          )
        except Exception:
          valid = False
        raise SystemExit(0 if valid else 1)
        PY
          then
            write_receipt START_FAILED "existing Weather4cast worker identity could not be verified"
            exit 14
          fi
          if ! kill -TERM -- "-$EXISTING_PID" 2>/dev/null; then
            write_receipt START_FAILED "existing Weather4cast worker could not be stopped safely"
            exit 15
          fi
          for attempt in $(seq 1 100); do
            if ! kill -0 "$EXISTING_PID" 2>/dev/null; then break; fi
            sleep 0.2
          done
          if kill -0 "$EXISTING_PID" 2>/dev/null; then
            write_receipt START_FAILED "existing Weather4cast worker did not stop before acceleration"
            exit 16
          fi
        fi
        ATTEMPT_ID=$(python3 - <<'PY'
        import uuid
        print(uuid.uuid4().hex)
        PY
        )
        ATTEMPT="$JOB/attempts/$ATTEMPT_ID"
        mkdir -p "$ATTEMPT"
        python3 - "$JOB/current-attempt.json" "$ATTEMPT/state.json" "$ATTEMPT_ID" <<'PY'
        import json, os, pathlib, sys
        selector_path = pathlib.Path(sys.argv[1])
        state_path = pathlib.Path(sys.argv[2])
        attempt_id = sys.argv[3]
        def atomic_json(path, payload):
          temporary = path.with_suffix(path.suffix + ".tmp")
          temporary.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
          os.replace(temporary, path)
        atomic_json(state_path, {
          "schema": "evomind.weather4cast_download_state.v2",
          "attempt_id": attempt_id,
          "status": "RUNNING",
          "phase": "starting",
          "failure_code": "",
          "rules_accepted": False,
          "secret_values_logged": False,
        })
        atomic_json(selector_path, {
          "schema": "evomind.weather4cast_current_attempt.v1",
          "attempt_id": attempt_id,
          "pid": 0,
        })
        PY
        if test "$ACCELERATE" = "1"; then
          cat > "$JOB/worker.sh" <<'WORKER'
        #!/usr/bin/env bash
        set -euo pipefail
        umask 077
        ROOT="$1"; SECRET="$2"; ATTEMPT_ID="$3"; JOB="$ROOT/.evomind/download-job"
        ATTEMPT="$JOB/attempts/$ATTEMPT_ID"
        exec 9<"$SECRET"
        rm -f "$SECRET"
        : > "$ATTEMPT/secret-consumed"
        set +e
        python3 - /dev/fd/9 "$ROOT" "$ATTEMPT_ID" <<'PY'
        import concurrent.futures, hashlib, json, os, pathlib, re, shutil, subprocess, sys, tarfile, time, zipfile

        secret_path, root_text, attempt_id = sys.argv[1:]
        root = pathlib.Path(root_text)
        job = root / ".evomind" / "download-job"
        attempt = job / "attempts" / attempt_id
        state_path = attempt / "state.json"
        manifest_path = root / ".evomind" / "data-manifest.json"
        askpass = attempt / "askpass.sh"
        password_file = attempt / "password"
        batch = attempt / "sftp-recursive-transfer.batch"
        adapter_telemetry = {
          "adapter_transfer_mode": "openssh_recursive_get",
          "adapter_requested_parallelism": 1,
          "adapter_listing_status": "ok",
          "adapter_listing_entries": 0,
          "adapter_listing_files": 0,
          "adapter_listing_directories": 0,
          "adapter_listing_archive_files": 0,
          "adapter_listing_scientific_files": 0,
          "adapter_listing_metadata_files": 0,
          "adapter_listing_other_files": 0,
          "adapter_listing_max_depth": 0,
          "adapter_listing_raw_lines": 0,
          "adapter_listing_control_lines": 0,
          "adapter_listing_rejected_lines": 0,
          "adapter_listing_skipped_directories": 0,
          "adapter_listing_timeout_exhaustions": 0,
          "adapter_parallel_failures": 0,
          "adapter_retry_rounds": 0,
          "adapter_pending_batches": 1,
          "adapter_fallback_reason": "",
          "adapter_sftp_buffer_bytes": 0,
          "adapter_sftp_num_requests": 0,
          "adapter_tuning_requested": False,
        }

        def atomic_json(path, payload):
          temporary = path.with_suffix(path.suffix + ".tmp")
          temporary.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
          os.replace(temporary, path)

        def state(status, phase, failure_code="", manifest_sha256=""):
          payload = {
            "schema": "evomind.weather4cast_download_state.v2",
            "attempt_id": attempt_id,
            "status": status,
            "phase": phase,
            "failure_code": failure_code,
            "manifest_sha256": manifest_sha256,
            "rules_accepted": False,
            "secret_values_logged": False,
          }
          payload.update(adapter_telemetry)
          atomic_json(state_path, payload)

        def digest(path):
          value = hashlib.sha256()
          with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
              value.update(chunk)
          return value.hexdigest()

        def classify(stderr):
          error = str(stderr or "").casefold()
          if "permission denied" in error or "authentication" in error or "login failed" in error:
            return "SFTP_AUTH_FAILED"
          if "timed out" in error:
            return "SFTP_TIMEOUT"
          if "connection refused" in error or "could not resolve" in error or "no route" in error or "connection closed" in error:
            return "SFTP_CONNECTION_FAILED"
          if "askpass" in error or "no tty" in error or "cannot open" in error:
            return "SFTP_ASKPASS_FAILED"
          return "SFTP_FAILED"

        def lquote(value):
          return json.dumps(str(value))

        class WorkerStop(Exception):
          def __init__(self, status, code, exit_code):
            self.status, self.code, self.exit_code = status, code, exit_code

        def smoke(path):
          if path.stat().st_size <= 0:
            return False, "empty"
          try:
            suffix = path.suffix.casefold()
            if suffix == ".zip":
              with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                  return False, "zip_crc"
                archive.infolist()
              return True, "zipfile"
            if suffix in {".tar", ".tgz", ".gz"}:
              with tarfile.open(path) as archive:
                archive.getmembers()
              return True, "tarfile"
            if suffix in {".h5", ".hdf5"}:
              import h5py
              with h5py.File(path, "r") as handle:
                list(handle.keys())
              return True, "h5py"
            if suffix == ".nc":
              import xarray as xr
              with xr.open_dataset(path) as dataset:
                list(dataset.variables)
              return True, "xarray"
          except Exception as exc:
            return False, type(exc).__name__
          return True, "nonempty_binary"

        try:
          data = json.load(open("/dev/fd/9", encoding="utf-8"))
          host = str(data.get("host") or "").strip()
          user = str(data.get("username") or "").strip()
          password = str(data.get("password") or "")
          port = int(data.get("port") or 22)
          remote = str(data.get("remote_path") or "/").strip()
          target = str(data.get("target_subdir") or "official").strip()
          valid = (
            data.get("schema") == "evomind.run_secret_payload.v1"
            and data.get("purpose") == "weather4cast_sftp"
            and re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host) is not None
            and re.fullmatch(r"[A-Za-z0-9_.@+-]{1,256}", user) is not None
            and 0 < len(password) <= 4096
            and "\\n" not in password and "\\r" not in password
            and 1 <= port <= 65535
            and remote.startswith("/") and "\\n" not in remote and "\\r" not in remote
            and target == "official"
          )
          if not valid:
            raise WorkerStop("DOWNLOAD_FAILED", "SFTP_CONTROLLED_PAYLOAD_INVALID", 12)
          data_root = root / target
          if root.is_symlink() or data_root.is_symlink():
            raise WorkerStop("DOWNLOAD_FAILED", "WEATHER_FILE_SET_INVALID", 17)
          data_root.mkdir(parents=True, exist_ok=True)
          if data_root.is_symlink() or not data_root.is_dir():
            raise WorkerStop("DOWNLOAD_FAILED", "WEATHER_FILE_SET_INVALID", 17)
          state("RUNNING", "transferring")
          sftp = shutil.which("sftp")
          setsid = shutil.which("setsid")
          if not sftp or not setsid:
            raise WorkerStop("DOWNLOAD_FAILED", "SFTP_BINARY_MISSING", 13)
          password_fd = os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
          with os.fdopen(password_fd, "w", encoding="utf-8") as handle:
            handle.write(password + "\\n")
          askpass.write_text('#!/bin/sh\\ncat "$EVOMIND_PASSWORD_FILE"\\n', encoding="utf-8")
          askpass.chmod(0o700)
          env = os.environ.copy()
          env.update({
            "DISPLAY": "evomind",
            "SSH_ASKPASS": str(askpass),
            "SSH_ASKPASS_REQUIRE": "force",
            "EVOMIND_PASSWORD_FILE": str(password_file),
          })
          usage = subprocess.run([sftp, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20, check=False)
          usage_text = str(usage.stdout or "") + "\\n" + str(usage.stderr or "")
          tuning = []
          if re.search(r"(?:^|[\\s,])(?:-B|--buffer)(?:[=\\s])", usage_text):
            tuning += ["-B", "262144"]
            adapter_telemetry["adapter_sftp_buffer_bytes"] = 262144
          if re.search(r"(?:^|[\\s,])(?:-R|--requests)(?:[=\\s])", usage_text):
            tuning += ["-R", "256"]
            adapter_telemetry["adapter_sftp_num_requests"] = 256
          adapter_telemetry["adapter_tuning_requested"] = bool(
            adapter_telemetry["adapter_sftp_buffer_bytes"] and adapter_telemetry["adapter_sftp_num_requests"]
          )
          options = [
            "-oBatchMode=no", "-oPreferredAuthentications=password",
            "-oPubkeyAuthentication=no", "-oStrictHostKeyChecking=accept-new",
            "-oConnectTimeout=60", "-oServerAliveInterval=30", "-oServerAliveCountMax=3",
            "-oCompression=no", "-oIPQoS=throughput", "-oTCPKeepAlive=yes",
            f"-oUserKnownHostsFile={job / 'known_hosts'}",
          ]
          command = [sftp, *tuning, *options, "-b", str(batch), "-P", str(port), f"{user}@{host}"]
          batch.write_text(
            "\\n".join([f"cd {lquote(remote)}", f"lcd {lquote(str(data_root))}", "get -aR .", "quit"]) + "\\n",
            encoding="utf-8",
          )
          completed = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=86400, check=False)
          if completed.returncode != 0:
            raise WorkerStop("DOWNLOAD_FAILED", classify(completed.stderr), completed.returncode or 15)
          state("RUNNING", "validating")
          rows = []
          smoke_rows = {}
          local_paths = [
            path
            for path in sorted(data_root.rglob("*"), key=lambda item: item.as_posix().casefold())
            if path.is_file() and not path.is_symlink()
          ]
          remote_files = [path.relative_to(data_root).as_posix() for path in local_paths]
          adapter_telemetry.update({
            "adapter_listing_entries": len(remote_files),
            "adapter_listing_files": len(remote_files),
            "adapter_listing_max_depth": max((len(pathlib.PurePosixPath(item).parts) for item in remote_files), default=0),
            "adapter_pending_batches": 0,
          })
          remote_files.sort(key=str.casefold)
          remote_file_set_sha256 = hashlib.sha256(
            ("\\n".join(remote_files) + "\\n").encode("utf-8")
          ).hexdigest()
          for path in local_paths:
            relative = path.relative_to(data_root).as_posix()
            ok, loader = smoke(path)
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
            smoke_rows[relative] = {"ok": ok, "loader": loader}
          payload = {
            "schema": "evomind.competition_data_manifest.v1",
            "competition": "weather4cast",
            "attempt_id": attempt_id,
            "file_count": len(rows),
            "total_bytes": sum(item["bytes"] for item in rows),
            "files": rows,
            "loader_smoke": smoke_rows,
            "duplicate_files": len(rows) - len({item["sha256"] for item in rows}),
            "expected_file_count": 108,
            "skipped_directory_count": 0,
            "remote_file_set_sha256": remote_file_set_sha256,
            "rules_accepted": False,
            "secret_values_logged": False,
          }
          if len(rows) != 108 or len(smoke_rows) != 108 or not all(item["ok"] for item in smoke_rows.values()):
            raise WorkerStop("VALIDATION_FAILED", "WEATHER_FILE_SET_INVALID", 9)
          atomic_json(manifest_path, payload)
          state("FULL_DATA_READY", "complete", manifest_sha256=digest(manifest_path))
          exit_code = 0
        except subprocess.TimeoutExpired:
          state("DOWNLOAD_FAILED", "failed", "SFTP_TIMEOUT")
          exit_code = 124
        except WorkerStop as exc:
          state(exc.status, "failed", exc.code)
          exit_code = int(exc.exit_code)
        except Exception as exc:
          state("VALIDATION_FAILED", "failed", type(exc).__name__)
          exit_code = 11
        finally:
          for temporary in (askpass, password_file, batch):
            try:
              temporary.unlink()
            except OSError:
              pass
        raise SystemExit(exit_code)
        PY
        rc=$?
        set -e
        printf '%s\\n' "$rc" > "$ATTEMPT/exit-code.tmp"; mv "$ATTEMPT/exit-code.tmp" "$ATTEMPT/exit-code"
        exit "$rc"
        WORKER
          chmod 700 "$JOB/worker.sh"
          nohup setsid bash "$JOB/worker.sh" "$ROOT" "$SECRET" "$ATTEMPT_ID" 6>&- 8>&- > /dev/null 2>&1 < /dev/null &
          WORKER_PID=$!
          printf '%s\\n' "$WORKER_PID" > "$ATTEMPT/pid.tmp"; mv "$ATTEMPT/pid.tmp" "$ATTEMPT/pid"
          python3 - "$JOB/current-attempt.json" "$ATTEMPT_ID" "$WORKER_PID" <<'PY'
        import json, os, pathlib, sys
        path = pathlib.Path(sys.argv[1]); attempt_id = sys.argv[2]; pid = int(sys.argv[3])
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("attempt_id") != attempt_id:
          raise SystemExit("current Weather4cast attempt changed during startup")
        data["pid"] = pid
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, sort_keys=True) + "\\n", encoding="utf-8")
        os.replace(temporary, path)
        PY
          for attempt in $(seq 1 100); do test -f "$ATTEMPT/secret-consumed" && break; sleep 0.1; done
          if test ! -f "$ATTEMPT/secret-consumed"; then
            write_receipt START_FAILED "background worker did not consume the ephemeral secret"
            exit 8
          fi
          write_receipt RUNNING "resumable Weather4cast recursive mirror started; poll competition_data_status"
          exit 0
        fi
        cat > "$JOB/worker.sh" <<'WORKER'
        #!/usr/bin/env bash
        set -euo pipefail
        umask 077
        ROOT="$1"; SECRET="$2"; ATTEMPT_ID="$3"; JOB="$ROOT/.evomind/download-job"
        ATTEMPT="$JOB/attempts/$ATTEMPT_ID"
        exec 9<"$SECRET"
        rm -f "$SECRET"
        : > "$ATTEMPT/secret-consumed"
        set +e
        python3 - /dev/fd/9 "$ROOT" "$ATTEMPT_ID" <<'PY'
        import concurrent.futures, hashlib, json, os, pathlib, re, shutil, subprocess, sys, tarfile, time, zipfile

        secret_path, root_text, attempt_id = sys.argv[1:]
        root = pathlib.Path(root_text)
        job = root / ".evomind" / "download-job"
        attempt = job / "attempts" / attempt_id
        state_path = attempt / "state.json"
        manifest_path = root / ".evomind" / "data-manifest.json"
        askpass = attempt / "askpass.sh"
        password_file = attempt / "password"
        batch = attempt / "sftp.batch"
        parallel_batches = []
        known = job / "known_hosts"
        adapter_telemetry = {
          "adapter_transfer_mode": "starting",
          "adapter_requested_parallelism": 0,
          "adapter_listing_status": "not_run",
          "adapter_listing_entries": 0,
          "adapter_listing_files": 0,
          "adapter_listing_directories": 0,
          "adapter_listing_archive_files": 0,
          "adapter_listing_scientific_files": 0,
          "adapter_listing_metadata_files": 0,
          "adapter_listing_other_files": 0,
          "adapter_listing_max_depth": 0,
          "adapter_listing_raw_lines": 0,
          "adapter_listing_control_lines": 0,
          "adapter_listing_rejected_lines": 0,
          "adapter_listing_skipped_directories": 0,
          "adapter_listing_timeout_exhaustions": 0,
          "adapter_parallel_failures": 0,
          "adapter_retry_rounds": 0,
          "adapter_pending_batches": 0,
          "adapter_fallback_reason": "",
          "adapter_sftp_buffer_bytes": 0,
          "adapter_sftp_num_requests": 0,
          "adapter_tuning_requested": False,
        }

        def atomic_json(path, payload):
          temporary = path.with_suffix(path.suffix + ".tmp")
          temporary.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
          os.replace(temporary, path)

        def state(status, phase, failure_code="", manifest_sha256=""):
          payload = {
            "schema": "evomind.weather4cast_download_state.v2",
            "attempt_id": attempt_id,
            "status": status,
            "phase": phase,
            "failure_code": failure_code,
            "manifest_sha256": manifest_sha256,
            "rules_accepted": False,
            "secret_values_logged": False,
          }
          payload.update(adapter_telemetry)
          atomic_json(state_path, payload)

        def digest(path):
          value = hashlib.sha256()
          with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
              value.update(chunk)
          return value.hexdigest()

        def lftp_quote(value):
          return json.dumps(str(value))

        __EVOMIND_WEATHER_SFTP_HELPERS__

        def classify(stderr):
          error = str(stderr or "").casefold()
          if "permission denied" in error or "authentication" in error or "login failed" in error:
            return "SFTP_AUTH_FAILED"
          if "timed out" in error:
            return "SFTP_TIMEOUT"
          if "connection refused" in error or "could not resolve" in error or "no route" in error or "connection closed" in error:
            return "SFTP_CONNECTION_FAILED"
          if "askpass" in error or "no tty" in error or "cannot open" in error:
            return "SFTP_ASKPASS_FAILED"
          if "batchfile" in error or "invalid command" in error:
            return "SFTP_BATCH_FAILED"
          return "SFTP_FAILED"

        class WorkerStop(Exception):
          def __init__(self, status, code, exit_code):
            self.status, self.code, self.exit_code = status, code, exit_code

        try:
          data = json.load(open("/dev/fd/9", encoding="utf-8"))
          host = str(data.get("host") or "").strip()
          user = str(data.get("username") or "").strip()
          password = str(data.get("password") or "")
          port = int(data.get("port") or 22)
          remote = str(data.get("remote_path") or "/").strip()
          target = str(data.get("target_subdir") or "official").strip()
          valid = (
            data.get("schema") == "evomind.run_secret_payload.v1"
            and data.get("purpose") == "weather4cast_sftp"
            and re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host) is not None
            and re.fullmatch(r"[A-Za-z0-9_.@+-]{1,256}", user) is not None
            and 0 < len(password) <= 4096
            and "\\n" not in password and "\\r" not in password
            and 1 <= port <= 65535
            and remote.startswith("/") and "\\n" not in remote and "\\r" not in remote
            and target == "official"
          )
          if not valid:
            raise WorkerStop("DOWNLOAD_FAILED", "SFTP_CONTROLLED_PAYLOAD_INVALID", 12)

          data_root = root / target
          if root.is_symlink() or data_root.is_symlink():
            raise WorkerStop("DOWNLOAD_FAILED", "WEATHER_FILE_SET_INVALID", 17)
          data_root.mkdir(parents=True, exist_ok=True)
          if data_root.is_symlink() or not data_root.is_dir():
            raise WorkerStop("DOWNLOAD_FAILED", "WEATHER_FILE_SET_INVALID", 17)
          state("RUNNING", "downloading")
          sftp = shutil.which("sftp")
          if not sftp:
            raise WorkerStop("DOWNLOAD_FAILED", "SFTP_BINARY_MISSING", 13)
          if sftp:
            password_fd = os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(password_fd, "w", encoding="utf-8") as handle:
              handle.write(password + "\\n")
            askpass.write_text('#!/bin/sh\\ncat "$EVOMIND_PASSWORD_FILE"\\n', encoding="utf-8")
            askpass.chmod(0o700)
            env = os.environ.copy()
            env.update({
              "DISPLAY": "evomind",
              "SSH_ASKPASS": str(askpass),
              "SSH_ASKPASS_REQUIRE": "force",
              "EVOMIND_PASSWORD_FILE": str(password_file),
            })
            usage = subprocess.run(
              [sftp, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
              text=True, timeout=20, check=False,
            )
            usage_text = str(usage.stdout or "") + "\\n" + str(usage.stderr or "")
            transfer_tuning = []
            if _weather4cast_sftp_supports_short_option(usage_text, "-B"):
              transfer_tuning += ["-B", "262144"]
              adapter_telemetry["adapter_sftp_buffer_bytes"] = 262144
            if _weather4cast_sftp_supports_short_option(usage_text, "-R"):
              transfer_tuning += ["-R", "256"]
              adapter_telemetry["adapter_sftp_num_requests"] = 256
            adapter_telemetry["adapter_tuning_requested"] = (
              adapter_telemetry["adapter_sftp_buffer_bytes"] == 262144
              and adapter_telemetry["adapter_sftp_num_requests"] == 256
            )
            base_options = [
              "-oBatchMode=no", "-oPreferredAuthentications=password",
              "-oPubkeyAuthentication=no", "-oStrictHostKeyChecking=accept-new",
              "-oConnectTimeout=60", "-oServerAliveInterval=30", "-oServerAliveCountMax=3",
              "-oCompression=no", "-oIPQoS=throughput", "-oTCPKeepAlive=yes",
              f"-oUserKnownHostsFile={known}",
            ]
            sshpass = shutil.which("sshpass")
            setsid = shutil.which("setsid")
            if not sshpass and not setsid:
              raise WorkerStop("DOWNLOAD_FAILED", "SFTP_BINARY_MISSING", 13)

            def sftp_command(batch_path, tuned=True):
              command = [
                sftp, *(transfer_tuning if tuned else []), *base_options,
                "-b", str(batch_path), "-P", str(port), f"{user}@{host}",
              ]
              return [sshpass, "-f", str(password_file), *command] if sshpass else [setsid, *command]

            def run_batch(batch_path, timeout=86400, capture_stdout=False, tuned=True):
              return subprocess.run(
                sftp_command(batch_path, tuned=tuned), env=env,
                stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, timeout=timeout, check=False,
              )

            adapter_telemetry["adapter_transfer_mode"] = "openssh_listing"
            state("RUNNING", "downloading")
            pending_directories = [""]
            seen_directories = set()
            remote_files = []
            remote_file_keys = set()
            local_directories = set()
            listing_index = 0
            listing_totals = {
              "raw_lines": 0, "control_lines": 0, "rejected_lines": 0,
              "files": 0, "directories": 0,
            }
            def refresh_listing_profile():
              adapter_telemetry.update(
                _weather4cast_listing_profile(remote_files, local_directories)
              )
            def reject_listing_structure(reason):
              refresh_listing_profile()
              adapter_telemetry.update({
                "adapter_listing_status": "parse_rejected",
                "adapter_fallback_reason": f"listing_structure_{reason}",
              })
              state("DOWNLOAD_FAILED", "failed", "SFTP_LISTING_REJECTED")
              raise WorkerStop("DOWNLOAD_FAILED", "SFTP_LISTING_REJECTED", 15)
            while pending_directories:
              current_directory = pending_directories.pop(0)
              directory_key = current_directory.casefold()
              if directory_key in seen_directories:
                reject_listing_structure("duplicate_directory")
              seen_directories.add(directory_key)
              if len(pathlib.PurePosixPath(current_directory).parts) > 16:
                reject_listing_structure("depth")
              list_batch = attempt / f"sftp-list-{listing_index:04d}.batch"
              listing_index += 1
              parallel_batches.append(list_batch)
              list_commands = [f"cd {lftp_quote(remote)}"]
              if current_directory:
                list_commands.append(f"cd {lftp_quote(current_directory)}")
              list_commands.extend(["ls -ln .", "quit"])
              list_batch.write_text("\\n".join(list_commands) + "\\n", encoding="utf-8")
              listing = None
              listing_failure = "SFTP_LISTING_FAILED"
              for listing_retry in range(4):
                try:
                  listing = run_batch(list_batch, timeout=300, capture_stdout=True, tuned=False)
                except subprocess.TimeoutExpired:
                  listing_failure = "SFTP_TIMEOUT"
                  listing = None
                else:
                  if listing.returncode == 0:
                    break
                  listing_failure = classify(listing.stderr)
                retryable_listing_failure = listing_failure in {
                  "SFTP_CONNECTION_FAILED", "SFTP_TIMEOUT", "SFTP_FAILED",
                }
                if not retryable_listing_failure or listing_retry >= 3:
                  break
                adapter_telemetry["adapter_retry_rounds"] = max(
                  int(adapter_telemetry.get("adapter_retry_rounds") or 0),
                  listing_retry + 1,
                )
                state("RUNNING", "downloading")
                time.sleep(min(4, 2 ** listing_retry))
              if listing is None or listing.returncode != 0:
                if _weather4cast_nonroot_listing_timeout_is_skippable(
                  listing_failure, current_directory
                ):
                  adapter_telemetry["adapter_listing_skipped_directories"] += 1
                  adapter_telemetry["adapter_listing_timeout_exhaustions"] += 1
                  state("RUNNING", "downloading")
                  continue
                adapter_telemetry.update({
                  "adapter_listing_status": "timeout" if listing_failure == "SFTP_TIMEOUT" else "nonzero",
                  "adapter_fallback_reason": "listing_timeout" if listing_failure == "SFTP_TIMEOUT" else "listing_nonzero",
                })
                state("DOWNLOAD_FAILED", "failed", listing_failure)
                raise WorkerStop(
                  "DOWNLOAD_FAILED", listing_failure,
                  124 if listing_failure == "SFTP_TIMEOUT" else (listing.returncode if listing else 15) or 15,
                )
              entries, diagnostics = _parse_weather4cast_sftp_long_listing(
                str(listing.stdout or ""), remote, current_directory
              )
              for key in listing_totals:
                listing_totals[key] += int(diagnostics[key])
              adapter_telemetry.update({
                "adapter_listing_raw_lines": listing_totals["raw_lines"],
                "adapter_listing_control_lines": listing_totals["control_lines"],
                "adapter_listing_rejected_lines": listing_totals["rejected_lines"],
              })
              state("RUNNING", "downloading")
              if diagnostics["rejected_lines"]:
                reject_reason = str(diagnostics.get("reject_reason", "unknown_type"))
                adapter_telemetry.update({
                  "adapter_listing_status": "parse_rejected",
                  "adapter_fallback_reason": f"listing_parse_rejected_{reject_reason}",
                })
                state("DOWNLOAD_FAILED", "failed", "SFTP_LISTING_REJECTED")
                raise WorkerStop("DOWNLOAD_FAILED", "SFTP_LISTING_REJECTED", 15)
              for kind, leaf in entries:
                relative = (pathlib.PurePosixPath(current_directory) / leaf).as_posix()
                parts = pathlib.PurePosixPath(relative).parts
                if not parts or len(parts) > 16 or any(part in {"", ".", ".."} for part in parts):
                  reject_listing_structure("invalid_relative")
                if kind == "directory":
                  pending_directories.append(relative)
                  local_directories.add(relative)
                else:
                  key = relative.casefold()
                  if key in remote_file_keys:
                    reject_listing_structure("duplicate_file")
                  remote_file_keys.add(key)
                  remote_files.append(relative)
              refresh_listing_profile()
              state("RUNNING", "downloading")
              if len(remote_files) + len(local_directories) > 1000:
                reject_listing_structure("entry_cap")
            adapter_telemetry.update({
              "adapter_listing_status": (
                "ok_with_skips"
                if adapter_telemetry["adapter_listing_skipped_directories"]
                else "ok"
              ),
              "adapter_listing_raw_lines": listing_totals["raw_lines"],
              "adapter_listing_control_lines": listing_totals["control_lines"],
              "adapter_listing_rejected_lines": listing_totals["rejected_lines"],
            })
            state("RUNNING", "downloading")
            if adapter_telemetry["adapter_listing_skipped_directories"]:
              state("DOWNLOAD_FAILED", "failed", "WEATHER_LISTING_INCOMPLETE")
              raise WorkerStop("DOWNLOAD_FAILED", "WEATHER_LISTING_INCOMPLETE", 18)
            try:
              remote_file_set_sha256 = _weather4cast_remote_file_set_sha256(remote_files)
            except ValueError:
              state("DOWNLOAD_FAILED", "failed", "WEATHER_FILE_SET_INVALID")
              raise WorkerStop("DOWNLOAD_FAILED", "WEATHER_FILE_SET_INVALID", 17)
            for relative in sorted(local_directories, key=str.casefold):
              destination = data_root
              for part in pathlib.PurePosixPath(relative).parts:
                destination = destination / part
                if destination.is_symlink():
                  raise WorkerStop("DOWNLOAD_FAILED", "WEATHER_FILE_SET_INVALID", 17)
                destination.mkdir(exist_ok=True)
            parallelism = min(16, len(remote_files))
            groups = [remote_files[index::parallelism] for index in range(parallelism)]
            if remote_files:
              transfer_batches = []
              for index, names in enumerate(groups):
                current_batch = attempt / f"sftp-transfer-{index:02d}.batch"
                commands = [f"cd {lftp_quote(remote)}", f"lcd {lftp_quote(str(data_root))}"]
                commands.extend(
                  f"reget {lftp_quote(name)} {lftp_quote(name)}"
                  for name in names
                )
                commands.append("quit")
                current_batch.write_text("\\n".join(commands) + "\\n", encoding="utf-8")
                transfer_batches.append(current_batch)
                parallel_batches.append(current_batch)
              adapter_telemetry.update({
                "adapter_transfer_mode": "openssh_parallel_files",
                "adapter_requested_parallelism": parallelism,
                "adapter_parallel_failures": 0,
                "adapter_retry_rounds": 0,
                "adapter_pending_batches": len(transfer_batches),
                "adapter_fallback_reason": "",
              })
              levels = []
              level = parallelism
              while level not in levels:
                levels.append(level)
                if level == 1:
                  break
                level = max(1, level // 2)
              pending_batches = list(transfer_batches)
              maximum_failures = 0
              fatal_codes = {"SFTP_AUTH_FAILED", "SFTP_ASKPASS_FAILED", "SFTP_BATCH_FAILED"}
              for retry_round, requested_parallelism in enumerate(levels):
                adapter_telemetry.update({
                  "adapter_requested_parallelism": min(requested_parallelism, len(pending_batches)),
                  "adapter_retry_rounds": retry_round,
                  "adapter_pending_batches": len(pending_batches),
                })
                state("RUNNING", "downloading")
                with concurrent.futures.ThreadPoolExecutor(
                  max_workers=max(1, min(requested_parallelism, len(pending_batches)))
                ) as pool:
                  results = list(pool.map(run_batch, pending_batches))
                failed_batches = []
                for failed_batch, result in zip(pending_batches, results):
                  if result.returncode == 0:
                    continue
                  failure_class = classify(result.stderr)
                  if failure_class in fatal_codes:
                    adapter_telemetry.update({
                      "adapter_parallel_failures": max(maximum_failures, 1),
                      "adapter_pending_batches": 1,
                      "adapter_fallback_reason": "fatal_parallel_failure",
                    })
                    state("DOWNLOAD_FAILED", "failed", failure_class)
                    raise WorkerStop("DOWNLOAD_FAILED", failure_class, result.returncode or 16)
                  failed_batches.append(failed_batch)
                maximum_failures = max(maximum_failures, len(failed_batches))
                adapter_telemetry["adapter_parallel_failures"] = maximum_failures
                pending_batches = failed_batches
                adapter_telemetry["adapter_pending_batches"] = len(pending_batches)
                if not pending_batches:
                  break
              if pending_batches:
                adapter_telemetry["adapter_fallback_reason"] = "parallel_retry_exhausted"
                state("DOWNLOAD_FAILED", "failed", "SFTP_PARALLEL_RETRY_EXHAUSTED")
                raise WorkerStop("DOWNLOAD_FAILED", "SFTP_PARALLEL_RETRY_EXHAUSTED", 16)
              adapter_telemetry["adapter_fallback_reason"] = ""
              completed = subprocess.CompletedProcess([], 0, "", "")
            else:
              state("DOWNLOAD_FAILED", "failed", "WEATHER_FILE_SET_INVALID")
              raise WorkerStop("DOWNLOAD_FAILED", "WEATHER_FILE_SET_INVALID", 17)
          if completed.returncode != 0:
            raise WorkerStop("DOWNLOAD_FAILED", classify(completed.stderr), completed.returncode or 14)

          state("RUNNING", "validating")
          rows = []
          smoke = {}
          local_paths = [
            path for path in sorted(
              data_root.rglob("*"), key=lambda item: item.as_posix().casefold()
            )
            if path.is_file() and not path.is_symlink()
          ]
          local_remote_files = [path.relative_to(data_root).as_posix() for path in local_paths]
          if not _weather4cast_local_file_set_matches_remote(local_remote_files, remote_files):
            raise WorkerStop("VALIDATION_FAILED", "WEATHER_FILE_SET_INVALID", 9)
          for path in local_paths:
            relative = path.relative_to(root).as_posix()
            ok = path.stat().st_size > 0
            loader = "nonempty_binary"
            try:
              suffix = path.suffix.casefold()
              if suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                  if archive.testzip() is not None:
                    raise ValueError("zip CRC failure")
                  archive.infolist()
                loader = "zipfile"
              elif suffix in {".tar", ".tgz", ".gz"}:
                with tarfile.open(path) as archive:
                  archive.getmembers()
                loader = "tarfile"
              elif suffix in {".h5", ".hdf5"}:
                import h5py
                with h5py.File(path, "r") as handle:
                  list(handle.keys())
                loader = "h5py"
              elif suffix == ".nc":
                import xarray as xr
                with xr.open_dataset(path) as dataset:
                  list(dataset.variables)
                loader = "xarray"
            except Exception as exc:
              ok = False
              loader = type(exc).__name__
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
            smoke[relative] = {"ok": ok, "loader": loader}

          payload = {
            "schema": "evomind.competition_data_manifest.v1",
            "competition": "weather4cast",
            "attempt_id": attempt_id,
            "file_count": len(rows),
            "total_bytes": sum(item["bytes"] for item in rows),
            "files": rows,
            "loader_smoke": smoke,
            "duplicate_files": len(rows) - len({item["sha256"] for item in rows}),
            "expected_file_count": 108,
            "skipped_directory_count": adapter_telemetry["adapter_listing_skipped_directories"],
            "remote_file_set_sha256": remote_file_set_sha256,
            "rules_accepted": False,
            "secret_values_logged": False,
          }
          if len(rows) != 108 or len(smoke) != 108 or not all(item["ok"] for item in smoke.values()):
            raise WorkerStop("VALIDATION_FAILED", "WEATHER_FILE_SET_INVALID", 9)
          atomic_json(manifest_path, payload)
          manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
          state("FULL_DATA_READY", "complete", manifest_sha256=manifest_sha256)
          exit_code = 0
        except subprocess.TimeoutExpired:
          state("DOWNLOAD_FAILED", "failed", "SFTP_TIMEOUT")
          exit_code = 124
        except FileNotFoundError:
          state("DOWNLOAD_FAILED", "failed", "SFTP_BINARY_MISSING")
          exit_code = 127
        except WorkerStop as exc:
          state(exc.status, "failed", exc.code)
          exit_code = int(exc.exit_code)
        except Exception as exc:
          state("VALIDATION_FAILED", "failed", type(exc).__name__)
          exit_code = 11
        finally:
          for temporary in (askpass, password_file, batch, *parallel_batches):
            try:
              temporary.unlink()
            except OSError:
              pass
        raise SystemExit(exit_code)
        PY
        rc=$?
        set -e
        printf '%s\n' "$rc" > "$ATTEMPT/exit-code.tmp"; mv "$ATTEMPT/exit-code.tmp" "$ATTEMPT/exit-code"
        exit "$rc"
        WORKER
        chmod 700 "$JOB/worker.sh"
        nohup setsid bash "$JOB/worker.sh" "$ROOT" "$SECRET" "$ATTEMPT_ID" 6>&- 8>&- > /dev/null 2>&1 < /dev/null &
        WORKER_PID=$!
        printf '%s\n' "$WORKER_PID" > "$ATTEMPT/pid.tmp"; mv "$ATTEMPT/pid.tmp" "$ATTEMPT/pid"
        python3 - "$JOB/current-attempt.json" "$ATTEMPT_ID" "$WORKER_PID" <<'PY'
        import json, os, pathlib, sys
        path = pathlib.Path(sys.argv[1]); attempt_id = sys.argv[2]; pid = int(sys.argv[3])
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("attempt_id") != attempt_id:
          raise SystemExit("current Weather4cast attempt changed during startup")
        data["pid"] = pid
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, sort_keys=True) + "\\n", encoding="utf-8")
        os.replace(temporary, path)
        PY
        for attempt in $(seq 1 100); do test -f "$ATTEMPT/secret-consumed" && break; sleep 0.1; done
        if test ! -f "$ATTEMPT/secret-consumed"; then
          python3 - "$ATTEMPT/state.json" "$ATTEMPT_ID" <<'PY'
        import json, os, pathlib, sys
        path = pathlib.Path(sys.argv[1]); attempt_id = sys.argv[2]
        payload = {
          "schema": "evomind.weather4cast_download_state.v2",
          "attempt_id": attempt_id,
          "status": "START_FAILED",
          "phase": "failed",
          "failure_code": "WORKER_START_FAILED",
          "rules_accepted": False,
          "secret_values_logged": False,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
        os.replace(temporary, path)
        PY
          write_receipt START_FAILED "background worker did not consume the ephemeral secret"
          exit 8
        fi
        write_receipt RUNNING "resumable Weather4cast mirror started; poll competition_data_status"
        """
    )
    # The nested worker heredoc is embedded inside a Python triple-quoted
    # string. Remove only the outer eight-space indentation after the shell
    # payload starts; this restores column-zero WORKER/PY terminators while
    # preserving the Python code relative indentation.
    helper_source = "\n\n".join(
        textwrap.dedent(inspect.getsource(helper)).strip()
        for helper in (
            _weather4cast_sftp_supports_short_option,
            _weather4cast_listing_file_category,
            _weather4cast_listing_profile,
            _weather4cast_nonroot_listing_timeout_is_skippable,
            _weather4cast_remote_file_set_sha256,
            _weather4cast_local_file_set_matches_remote,
            _parse_weather4cast_sftp_long_listing,
        )
    )
    raw = raw.replace("__EVOMIND_ACCELERATE__", "1" if accelerate else "0")
    lines = raw.splitlines(keepends=True)
    payload_started = False
    normalized: list[str] = []
    for line in lines:
        if line.lstrip().startswith("ACCELERATE="):
            payload_started = True
        if payload_started and line.startswith("        "):
            line = line[8:]
        normalized.append(line)
    return "".join(normalized).replace("__EVOMIND_WEATHER_SFTP_HELPERS__", helper_source)


def _local_prepared_prepare_script(competition: str) -> str:
    """Verify an already-prepared MLE-bench dataset and publish a pointer.

    The official archive is staged on the shared GPU data root, so preparation
    is read-only: verify the public layout, publish ``$ROOT/prepared`` as a
    symlink to it, and write the standard competition-data manifest that the
    status adapter validates (counts, label rows, listing hash).
    """
    slug = str(LOCAL_PREPARED_SOURCES[competition]["mlebench_slug"])
    official_url = str(LOCAL_PREPARED_SOURCES[competition]["source"])
    source_root = f"{MLEBENCH_PREPARED_ROOT}/{slug}/prepared"
    header = _script_header(competition)
    body = textwrap.dedent(
        """\
        SOURCE="__SOURCE_ROOT__"
        if [ ! -d "$SOURCE/public/train" ] || [ ! -d "$SOURCE/public/test" ] \
           || [ ! -f "$SOURCE/public/train_labels.csv" ] || [ ! -f "$SOURCE/public/sample_submission.csv" ]; then
          write_receipt SOURCE_UNAVAILABLE "MLE-bench prepared layout is incomplete under $SOURCE"
          exit 0
        fi
        ln -sfn "$SOURCE" "$ROOT/prepared"
        python3 - "$SOURCE" "$ROOT" "$COMPETITION" "__OFFICIAL_URL__" <<'PY'
        import datetime, hashlib, json, os, sys

        source, root, competition, official_url = sys.argv[1:5]
        evomind = os.path.join(root, ".evomind")
        os.makedirs(evomind, exist_ok=True)
        digest = hashlib.sha256()
        counts = {}
        total_files = 0
        total_bytes = 0
        for part in ("train", "test"):
            files = 0
            part_bytes = 0
            base = os.path.join(source, "public", part)
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames.sort()
                for entry in sorted(filenames):
                    path = os.path.join(dirpath, entry)
                    relative = os.path.relpath(path, source)
                    size = os.path.getsize(path)
                    files += 1
                    part_bytes += size
                    digest.update(("%s|%d\\n" % (relative, size)).encode("utf-8"))
            counts[part] = {"files": files, "bytes": part_bytes}
            total_files += files
            total_bytes += part_bytes
        for entry in ("train_labels.csv", "sample_submission.csv"):
            path = os.path.join(source, "public", entry)
            size = os.path.getsize(path)
            digest.update(("%s|%d\\n" % (entry, size)).encode("utf-8"))
            total_files += 1
            total_bytes += size
        label_rows = 0
        with open(os.path.join(source, "public", "train_labels.csv"), encoding="utf-8", errors="replace") as handle:
            for label_rows, _line in enumerate(handle, start=0):
                pass
        manifest = {
            "schema": "evomind.competition_data_manifest.v1",
            "competition": competition,
            "access": "local_prepared_mlebench",
            "official_url": official_url,
            "source_prepared_root": source,
            "file_count": total_files,
            "total_bytes": total_bytes,
            "train_file_count": counts["train"]["files"],
            "train_bytes": counts["train"]["bytes"],
            "test_file_count": counts["test"]["files"],
            "test_bytes": counts["test"]["bytes"],
            "label_rows": label_rows,
            "prepared_file_set_sha256": digest.hexdigest(),
            "rules_accepted": True,
            "secret_values_logged": False,
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        target = os.path.join(evomind, "data-manifest.json")
        temporary = target + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
            handle.write("\\n")
        os.replace(temporary, target)
        print(json.dumps({"competition": competition, "file_count": total_files,
                          "label_rows": label_rows}, sort_keys=True))
        PY
        write_receipt FULL_DATA_READY "MLE-bench prepared layout verified; train/test counts and listing hash recorded"
        """
    )
    return header + body.replace("__SOURCE_ROOT__", source_root).replace("__OFFICIAL_URL__", official_url)


def build_adapter_script(competition: str, action: str) -> str:
    name = normalize_competition(competition)
    if action not in {"prepare", "status", "accelerate"}:
        raise ValueError("invalid competition data action")
    if action == "accelerate":
        if name != "weather4cast":
            raise ValueError("competition acceleration is available only for Weather4cast")
        return _weather4cast_prepare_script(accelerate=True)
    if action == "status":
        return _status_script(name)
    if name in _LOCAL_PREPARED_COMPETITIONS:
        return _local_prepared_prepare_script(name)
    if name == "mindgames":
        return _mindgames_prepare_script()
    if name == "e2lmc":
        return _e2lmc_prepare_script()
    if name in _KAGGLE_COMPETITIONS:
        return _kaggle_prepare_script(name)
    if name == "weather4cast":
        return _weather4cast_prepare_script()
    detail = COMPETITION_CATALOG[name]["access"]
    return _script_header(name) + textwrap.dedent(
        f"""\
        write_receipt HUMAN_GATE_OR_AUTH {detail!r}
        """
    )


def write_adapter_script(task_root: Path, competition: str, action: str) -> Path:
    name = normalize_competition(competition)
    target = task_root / "work" / "competition_data" / f"{name}-{action}.sh"
    target.parent.mkdir(parents=True, exist_ok=True)
    data = build_adapter_script(name, action).encode("utf-8")
    temporary = target.with_suffix(".sh.tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    return target


def validate_receipt(competition: str, receipt: Any) -> dict[str, Any]:
    name = normalize_competition(competition)
    if not isinstance(receipt, dict):
        raise ValueError("competition data receipt is missing")
    if receipt.get("schema") != "evomind.competition_data_receipt.v1" or receipt.get("competition") != name:
        raise ValueError("competition data receipt identity mismatch")
    if receipt.get("persistent_root") != persistent_root(name):
        raise ValueError("competition data receipt persistent root mismatch")
    status = str(receipt.get("status") or "")
    if status not in {
        "NOT_STARTED", "RUNNING", "RUNNING_OR_PARTIAL", "PARTIAL", "FULL_DATA_READY",
        "SECRET_REQUIRED", "DEPENDENCY_MISSING", "START_FAILED", "SOURCE_CONFLICT", "HUMAN_GATE_OR_AUTH",
        "HUMAN_GATE_REQUIRED", "AUTH_FAILED", "DOWNLOAD_FAILED", "VALIDATION_FAILED", "SOURCE_UNAVAILABLE",
    }:
        raise ValueError("competition data receipt status is invalid")
    if receipt.get("secret_values_logged") is not False:
        raise ValueError("competition data receipt lacks the no-secret-output invariant")
    if name == "mindgames" and status == "FULL_DATA_READY":
        smoke = receipt.get("loader_smoke")
        expected_environments = {environment for environment, _ in _MINDGAMES_ENVIRONMENTS}
        if (
            receipt.get("expected_environment_count") != len(expected_environments)
            or receipt.get("textarena_version") != _MINDGAMES_TEXTARENA_VERSION
            or receipt.get("runtime_dependencies") != _MINDGAMES_RUNTIME_DEPENDENCIES
            or not isinstance(smoke, dict)
            or set(smoke) != expected_environments
            or not all(isinstance(item, dict) and item.get("ok") is True for item in smoke.values())
        ):
            raise ValueError("MindGames FULL_DATA_READY receipt lacks dependency or loader evidence")
    if name == "weather4cast" and status == "FULL_DATA_READY":
        smoke = receipt.get("loader_smoke")
        skipped_directory_count = receipt.get("skipped_directory_count")
        if (
            re.fullmatch(r"[a-f0-9]{32}", str(receipt.get("attempt_id") or "")) is None
            or receipt.get("phase") != "complete"
            or receipt.get("failure_code") not in {"", None}
            or receipt.get("files") != 108
            or receipt.get("expected_file_count") != 108
            or re.fullmatch(r"[a-f0-9]{64}", str(receipt.get("manifest_sha256") or "")) is None
            or re.fullmatch(r"[a-f0-9]{64}", str(receipt.get("remote_file_set_sha256") or "")) is None
            or isinstance(skipped_directory_count, bool)
            or skipped_directory_count != 0
            or receipt.get("adapter_listing_status") != "ok"
            or isinstance(receipt.get("adapter_listing_skipped_directories"), bool)
            or not isinstance(receipt.get("adapter_listing_skipped_directories"), int)
            or receipt.get("adapter_listing_skipped_directories") != 0
            or isinstance(receipt.get("adapter_listing_timeout_exhaustions"), bool)
            or not isinstance(receipt.get("adapter_listing_timeout_exhaustions"), int)
            or receipt.get("adapter_listing_timeout_exhaustions") != 0
            or receipt.get("worker_exit_code") != 0
            or not isinstance(smoke, dict)
            or len(smoke) != 108
            or not all(isinstance(item, dict) and item.get("ok") is True for item in smoke.values())
        ):
            raise ValueError("Weather4cast FULL_DATA_READY receipt lacks attempt-bound manifest or loader evidence")
    return dict(receipt)


def catalog_projection() -> list[dict[str, Any]]:
    return [
        {
            "id": name,
            "title": item["title"],
            "official_source": item["source"],
            "access": item["access"],
            "persistent_root": persistent_root(name),
            "required_secret_purpose": required_secret_purpose(name),
        }
        for name, item in COMPETITION_CATALOG.items()
    ]


def kaggle_catalog_projection() -> list[dict[str, Any]]:
    """Return the fixed Kaggle subset without making a live listing request.

    These slugs are part of the shipped six-competition contract.  The live
    Kaggle listing is useful discovery evidence, but it is not an availability
    dependency for adapters whose official URLs and slugs are already pinned.
    """
    return [
        {
            "catalog_id": name,
            "slug": slug,
            "title": COMPETITION_CATALOG[name]["title"],
            "deadline": "",
            "metric": "",
            "url": COMPETITION_CATALOG[name]["source"],
        }
        for name, slug in _KAGGLE_COMPETITIONS.items()
    ]


def source_catalog_sha256() -> str:
    payload = json.dumps(catalog_projection(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "COMPETITION_CATALOG",
    "LOCAL_PREPARED_SOURCES",
    "MLEBENCH_PREPARED_ROOT",
    "PERSISTENT_COMPETITION_DATA_ROOT",
    "build_adapter_script",
    "catalog_projection",
    "kaggle_catalog_projection",
    "normalize_competition",
    "persistent_root",
    "required_secret_purpose",
    "source_catalog_sha256",
    "validate_receipt",
    "write_adapter_script",
]
