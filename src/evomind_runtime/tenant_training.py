"""t238 C7c-4: minimal tenant GPU training — ``managed_tensor_train`` only, tenant-scoped.

* Offered to tenant (personal) sessions only while the GPU time-share is active and its
  ``gpu_pool`` block sets ``tenant_training: true``. Every other HPC tool stays refused.
* Input comes only from this Run's ``inputs/`` (the bound-clone copies of the tenant's own
  uploads); the data file's SHA-256 is the ``protocol_id`` and its path is inside the
  approved configuration, so approval binds both.
* Dispatch uses the service-level principal: the operator's ``hpc-binding.json`` for job
  98270 only (X6). Anything else fails closed with a Chinese message.
* Nothing HPC-side (paths, hosts, job ids, profiles, GPU UUIDs) reaches the tenant: raw
  runtime files stay outside the Run workspace and every result, artifact and log passes
  ``scrub`` at the boundary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import zipfile
import zlib
from array import array
from pathlib import Path
from typing import Any

from . import gpu_admission

MAX_DATA_BYTES = 256 * 1024 * 1024
MAX_ROWS = 2_000_000
MAX_COLUMNS = 8192
MAX_CELLS = 256 * 1024 * 1024  # 1 GiB as float32: what one training set may occupy in host memory
MAX_NPZ_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_CSV_RECORD_CHARS = 4 * 1024 * 1024  # far above MAX_COLUMNS numeric cells; checked before csv parses the record
_PROFILE_INSTANCE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE)
_BINDING_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_OWNER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TENANT_CONFIG_FIELDS = {"hidden_sizes", "epochs", "batch_size", "learning_rate", "seed", "data_path", "task", "target_column", "model_family"}


class TenantTrainingError(ValueError):
    """``str(error)`` is a stable code; ``message`` is the tenant-facing Chinese sentence."""

    MESSAGES = {
        "binding_invalid": gpu_admission.MESSAGES["binding_invalid"],
        "identity_gate_not_ready": "训练暂时不可用，请稍后再试。",
        "environment_not_ready": "这次训练还没有开始：训练环境没有准备好。点「重新准备」就行。",
        "tenant_training_disabled": "现在还不能在这里训练模型。",
        "data_path_rejected": gpu_admission.MESSAGES["path_rejected"],
        "data_invalid": "训练数据读不出来。请上传带表头的数字表格（CSV），或含 x 和 y 的 NPZ 文件。",
        "data_too_large": "训练数据超过训练上限（256\u00a0MiB）。先缩小或预处理，再继续。",
        "data_too_many_cells": "训练数据解压后的大小或行列数超过训练上限。先缩小或预处理，再继续。",
        "data_changed": "训练数据和批准时不一致，请重新确认后再训练。",
        "config_invalid": "训练设置无效，请检查后再试。",
        "config_invalid_repeated": "训练设置连续多次无效，已停止自动重试。请检查数据文件和训练要求后重新发起。",
    }

    @property
    def message(self) -> str:
        return self.MESSAGES.get(str(self), self.MESSAGES["config_invalid"])


# -- scrubber (tenant boundary) --

_SCRUB_PATTERNS = (
    re.compile(r"/hpc2hdd(?:/[^\s\"'<>|,;)\]}]*)?", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])/(?:home|tmp|scratch|data|mnt|opt|var|root)(?:/[^\s\"'<>|,;)\]}]+)+"),
    re.compile(r"\btenant_[a-f0-9]{24}_job\d+_g\d+\b", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])job[_-]?\d{4,}(?!\d)", re.IGNORECASE),
    re.compile(r"\b(?:jinghw|aimslab)[A-Za-z0-9_.-]*", re.IGNORECASE),
    re.compile(r"\bGPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.IGNORECASE),
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE),
    re.compile(r"\b10\.120\.\d{1,3}\.\d{1,3}(?::\d+)?\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?\b"),
    re.compile(r"(?<![0-9A-Za-z])98270(?![0-9A-Za-z])"),
    re.compile(r"(?<![0-9A-Za-z])" + str(int(gpu_admission.ALLOWED_HPC_JOB_ID)) + r"(?![0-9A-Za-z])"),
    # 2026-11-05: every card of the pool, not only the primary.
    *(re.compile(r"(?<![0-9A-Za-z])" + str(int(job)) + r"(?![0-9A-Za-z])") for job in sorted(gpu_admission.POOL_HPC_JOB_IDS)),
)
HIDDEN = "[已隐藏]"
# t238 C7c-FIX4: the model relays tool results; tell it how a training refusal may be worded (failure replies quoted the error
# code, the budget, account binding and hashes). The web view also strips any code that still slips through.
REPLY_RULE = ("Quote the sentence in user_message word for word, in plain Chinese, then at most one short next step (try again later or "
              "adjust the data). Never quote error codes or snake_case identifiers, and never mention budgets, minutes, whether anything was or was not used up, account "
              "or identity binding, hashes, hardware, shells, drivers or internal tools unless user_message does. If user_message says the training was cancelled, do "
              "not start or retry it yourself: wait until the person asks again.")
# The same for a finished training: say in Chinese what was made and where, quote the numbers, never narrate how it was checked.
SUCCESS_RULE = ("Reply in plain Chinese only. Say that the training finished, where the model and predictions are (the task's files), and the headline numbers "
                "from managed_training.metrics. If managed_training.model_selection is present, add one plain sentence: the platform compared "
                "model_selection.trials settings automatically and kept the best one; when model_selection.default_holdout exists, give the main number of "
                "the default setting next to the chosen one. "
                "Never mention hashes, checksums, protocols, receipts, adapters, hardware, queues, budgets or how results were verified, "
                "and write no English narration, not even between tool calls.")
CODE_PATTERN = re.compile(r"`?tenant_training_[a-z0-9_]+`?")
FALLBACK_MESSAGE = "训练暂时不可用，请稍后再试。"


_FENCE = re.compile(r"^\s*(?:```|~~~)")
_LIST_ITEM = re.compile(r"^(\s*(?:\d{1,3}[.)]|[-*+\u2022])\s+)(.*)$")


def _tidy_line(line: str) -> str:
    """Leftovers of a removed code or sentence on ONE edited line: 「，`，」, 「。，」, empty brackets, doubled commas, one unpaired backtick."""
    line = re.sub(r"[，,、]\s*`+\s*(?=[，,。；;])", "", line)
    if line.count("`") % 2:
        line = re.sub(r"(?<![`\w])`+(?![`\w])", "", line)
    line = re.sub(r"[（(]\s*[）)]", "", line)
    line = re.sub(r"([。！？；])\s*[，,、]+", r"\1", line)
    line = re.sub(r"[，,]{2,}", "，", line)
    line = re.sub(r"(?<=\S)[ \t]{2,}", " ", line)
    return re.sub(r"^([ \t]*)[，,、；;]+\s*", r"\1", line).rstrip()


def _edit_prose_lines(text: str, edit) -> str:
    """Apply ``edit`` to every line outside a fenced code block and tidy ONLY the lines it changed: code, indentation and markdown stay as written."""
    out: list[str] = []
    fenced = False
    for line in str(text).split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
            out.append(line)
            continue
        if fenced:
            out.append(line)
            continue
        new = edit(line)
        out.append(_tidy_line(new) if new != line else line)
    return "\n".join(out)


def tidy_punctuation(text: str) -> str:
    """Tidy the leftovers of removed codes, line by line, outside fenced code."""
    return _edit_prose_lines(text, lambda line: line).strip()


def _residue(line: str) -> str:
    """Tenant wording on one prose line: 「您」 → 「你」 and a bracketed code name like （xxx_yyy） is dropped (inline code spans stay as written)."""
    line = line.replace("您", "你")
    line = re.sub(r"（\s*`?[A-Za-z]+(?:_[A-Za-z0-9]+)+`?\s*）", "", line)
    return "".join(part if index % 2 else re.sub(r"\(\s*[a-z]+(?:_[a-z0-9]+)+\s*\)", "", part)
                   for index, part in enumerate(re.split(r"(`[^`\n]*`)", line)))


def plain_reply(text: str) -> str:
    """Replace a leaked ``tenant_training_*`` code in assistant text with its tenant sentence (code blocks are left alone)."""
    def swap(match: re.Match) -> str:
        code = match.group(0).strip("`")[len("tenant_training_"):]
        return TenantTrainingError.MESSAGES.get(code, FALLBACK_MESSAGE)

    def edit(line: str) -> str:
        line = re.sub(r"[（(]?`?remote_dispatched`?\s*[:：=]\s*(?:true|false)[）)]?[，,]?", "", line)
        return _residue(CODE_PATTERN.sub(swap, line))
    return _edit_prose_lines(text, edit).strip()


# t238 C7c-FIX5 (Critiquito B2): the tenant must never read the machinery in the assistant's own prose. Words are matched per sentence and
# the sentence is dropped, in any language (the model narrated "Configuration written (sha256 …) … through the managed adapter" in English).
# FIX5h (ACCEPT11 H-1): word-bounded, so ordinary ML talk (shape, SHAP, shared, Shanghai) is never dropped.
INTERNAL_TERMS = re.compile(
    r"哈希|校验和|(?<![a-z])sha[-_ ]?\d{3}(?!\d)|_sha\d*|(?<![a-z])sha(?![a-z])|\bhash(?:ed|es|ing)?\b|\bprotocols?\b|\bdata[_-]protocol\b|"
    r"\bmanaged\b|\b[0-9a-f]{40,64}\b|受管|适配器|\badapters?\b|\bGPU[s]?\b|\bCUDA\b|显存|显卡|\bVRAM\b|vLLM|A800|\bA40\b|AIMSLAB|Workbench|"
    r"训练身份|\bidentity[\s_-]*(?:gate|binding)|租约|\blease\b|remote_dispatched|tenant_training|managed_tensor_train|\bshell\b|训练资源|\bdispatch(?:ed|es|ing)?\b|\bcheck-?sums?\b|"
    r"[A-Za-z]:\\ProgramData|/hpc2hdd", re.IGNORECASE)
OMITTED_STEP = "这部分说明已省略。"
# split AFTER the sentence end (the period stays with its sentence)
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])|(?<=[a-z0-9\)\]\"”]\.)(?=\s+[A-Z“\"(\u3400-\u9fff])")
_CJK = re.compile(r"[\u3400-\u9fff]")


_LATIN_FUNCTION_WORDS = frozenset(
    "a an the we i you he she it they is are was were be been am do does did done has have had will would can could should may might "
    "this that these those here there to of on in at for with by from as and or but not no so if then than our your my its their "
    "us me them what when where which who how all any each some more most very just also only again".split())


def _latin_narration(part: str) -> bool:
    """True for a whole English/Latin *prose* sentence (Critiquito F5-2). Code, JSON and CJK stay."""
    s = part.strip()
    if not s or _CJK.search(s):
        return False
    # narration starts with a capital (optionally after an opening quote); code/JSON keys do not
    open_q = "\u201c\u201d\"'"
    if s[0] in open_q:
        if len(s) < 2 or not ("A" <= s[1] <= "Z"):
            return False
    elif not ("A" <= s[0] <= "Z"):
        return False
    words = re.findall(r"[A-Za-z]+", s)
    if len(words) < 2:
        return False
    if sum(len(w) for w in words) < 8:
        return False
    # `Accuracy on test set: 0.91` — a key/value metric line ends in a bare number (ACCEPT15 M-1)
    if re.search(r"[:=]\s*[-+]?\d+(?:\.\d+)?%?\s*$", s):
        return False
    # FIX5c: `Best model: Random Forest` / `Model: XGBoost` — a name/value line with a plain display value is data (leaky values are screened in plain_prose)
    if _name_value_data(s):
        return False
    # A metric, file name or column name (no function words, carries a number or an identifier) is data, not narration (ACCEPT14 M-1).
    if not any(w.lower() in _LATIN_FUNCTION_WORDS for w in words) and (re.search(r"\d", s) or re.search(r"\w[_.]\w", s)):
        return False
    return True


# t238 C7c-FIX5c (Grok ruling): an English success reply keeps its model name and metrics. A `Best model: Random Forest` line is data, not narration,
# but a name/value line is also the easiest way for an internal value to slip through, so the value is checked before the line is kept.
_NV_LINE = re.compile(
    r"^\s*(?:(?:\d{1,3}[.)]|[-*+\u2022])\s+)?[*_`]*\s*([A-Za-z\u3400-\u9fff][A-Za-z \u3400-\u9fff]{0,38}?)\s*[*_`]*\s*[:\uff1a]\s*[*_`]*\s*(\S.*?)\s*$")
_NV_KEY_OK = frozenset(
    "model models algorithm algorithms classifier classifiers regressor estimator method methods metric metrics target label labels dataset data file files "
    "column columns feature features optimizer loss task type best selected final chosen baseline split sample samples rows row class classes result results "
    "output input name accuracy precision recall score".split())
_NV_KEY_ALSO_FUNCTION = frozenset("of on in at for per".split())
_NV_KEY_SENSITIVE = frozenset(
    "api token tokens secret secrets password passwd credential credentials host hostname node worker workers device devices path paths url uri endpoint "
    "provider backend cluster queue job jobs partition container server machine account tenant user email ip engine driver runtime id gpu".split())
_NV_KEY_MODELISH = frozenset("model models algorithm algorithms classifier classifiers regressor estimator engine provider backend llm".split())
# the assistant's own model/provider family: never shown in a tenant-facing name/value line
_FAMILY = re.compile(
    r"\b(?:gpt(?:[-_ ]?\d\w*)?|chatgpt|claude|anthropic|openai|opus|sonnet|haiku|gemini|grok|llama(?:[-_ ]?\d\w*)?|qwen\d*(?:\.\d+)?|deepseek|mistral|mixtral|"
    r"kimi|moonshot|glm[-_ ]?\d\w*|antigravity|lt4net|xai)\b", re.IGNORECASE)
_FILE_EXT = re.compile(r"\.(?:pt|pth|pkl|pickle|joblib|onnx|h5|ckpt|bin|safetensors|csv|tsv|json|jsonl|txt|md|png|jpg|xlsx|parquet|yaml|yml)\b", re.IGNORECASE)
_CLEAN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+]{0,24}(?:-[A-Za-z][A-Za-z0-9+]{0,24})*|\d{1,6}(?:\.\d+)?%?")
_RAW_CODE = re.compile(r"[A-Za-z0-9.]+(?:[_-][A-Za-z0-9.]+)+|\S*/\S*")


def _name_value(line: str):
    m = _NV_LINE.match(line)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def _key_words(key: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z]+", key)]


def _clean_value(value: str) -> bool:
    """A display name such as `Random Forest`, `XGBoost`, `Logistic Regression (L2)` or a short list of them: plain words, no codes, ids, paths or narration."""
    v = re.sub("[\u300c\u300d\"'`\u201c\u201d\u2018\u2019]", "", value).strip().rstrip(".;").strip()
    if not v or v[-1] in "!?:" or len(v) > 90:
        return False
    toks = [t for t in re.split(r"[\s,;]+", v.replace("(", " ").replace(")", " ")) if t]
    if not toks or len(toks) > 12:
        return False
    for t in toks:
        if t.lower() == "and" or t.lower() == "vs":
            continue
        if not _CLEAN_TOKEN.fullmatch(t) or t.lower() in _LATIN_FUNCTION_WORDS:
            return False
    return True


def _leaky_name_value(line: str) -> bool:
    """A `Name: value` line whose value is an internal token (model/provider id, host, device, path, secret, digest) never reaches the tenant."""
    nv = _name_value(line)
    if not nv:
        return False
    key, value = nv
    if _CJK.search(key) and "模型" not in key:
        return False  # other Chinese keys: the sentence filters decide
    words = _key_words(key)
    if any(w in _NV_KEY_SENSITIVE for w in words):
        return True
    if _FAMILY.search(value) or _FAMILY.search(key):
        return True
    if re.match(r"^(?:[/~\\]|[A-Za-z]:[\\/])", value) or "://" in value or "\\\\" in value:
        return True
    plain = _FILE_EXT.sub("", value)
    if re.search(r"\b(?=[0-9a-f]*\d)[0-9a-f]{16,}\b", plain, re.IGNORECASE) or re.search(r"\b(?:sk|pk|ghp|gho|xox[a-z]|AKIA)[-_]?[A-Za-z0-9]{8,}", plain):
        return True
    if any(len(t) >= 24 for t in re.findall(r"[A-Za-z0-9_-]+", plain)):
        return True
    if any(w in _NV_KEY_MODELISH for w in words) or "模型" in key:
        # a model name is a plain display name or one of the user's own files; `tenant_model_7f3a`, `qwen2.5-72b` and `provider/model` are codes
        for tok in value.split():
            tok = tok.strip(",;()")
            if _FILE_EXT.search(tok):
                continue
            if _RAW_CODE.fullmatch(tok) and not _CLEAN_TOKEN.fullmatch(tok) and not re.fullmatch(r"\d+(?:\.\d+)?%?", tok):
                return True
    return False


def _name_value_data(s: str) -> bool:
    """`Best model: Random Forest`, `Model: XGBoost`, `Target column: churned`: a short key from the data-science vocabulary and a plain display value."""
    nv = _name_value(s)
    if not nv:
        return False
    key, value = nv
    words = _key_words(key)
    if not words or len(words) > 4 or not any(w in _NV_KEY_OK for w in words):
        return False
    if any(w in _LATIN_FUNCTION_WORDS and w not in _NV_KEY_ALSO_FUNCTION and w not in _NV_KEY_OK for w in words):
        return False
    if any(w in _NV_KEY_SENSITIVE for w in words) or _leaky_name_value(s):
        return False
    return _clean_value(value) or bool(_FILE_EXT.search(value) and not any(t.lower() in _LATIN_FUNCTION_WORDS for t in re.findall(r"[A-Za-z]+", value)))


def _keep_sentence(part: str) -> bool:
    return (not INTERNAL_TERMS.search(part)) and (not _latin_narration(part)) and (not _leaky_name_value(part))


def _join_kept(parts: list[str]) -> str:
    """Join kept sentences; drop the Latin-period space that would sit after a Chinese 。！？."""
    out: list[str] = []
    for part in parts:
        if out and out[-1].rstrip()[-1:] in "。！？；" and part[:1] == " ":
            out.append(part.lstrip())
        else:
            out.append(part)
    return "".join(out)


_QUOTED = re.compile(r"「([^「」]*)」")


def _drop_quoted_narration(line: str) -> str:
    """A 「…」 that holds only whole English narrative sentences (or nothing) goes with its quotes; names and metrics inside 「」 stay."""
    def _repl(m: "re.Match[str]") -> str:
        inner = m.group(1)
        parts = [part for part in _SENTENCE_END.split(inner) if part and part.strip()]
        # ACCEPT15 H-1: only an empty 「」 or whole narrative sentences that end in terminal punctuation; a name such as 「Churn Label」 stays
        if not parts or all(_latin_narration(part) and part.rstrip().endswith((".", "!", "?")) for part in parts):
            return ""
        return m.group(0)
    return _QUOTED.sub(_repl, line)


def plain_prose(text: str) -> str:
    """Drop every sentence or line that names the machinery or is Latin narration (outside code blocks), then tidy only the lines that were edited."""
    kept: list[str] = []
    fenced = False
    table_dropped = False
    for line in str(text).split("\n"):
        if table_dropped and line.lstrip().startswith("|"):
            continue
        table_dropped = False
        if _FENCE.match(line):
            fenced = not fenced
            kept.append(line)
            continue
        if not fenced:
            tenant_line = _residue(line)
            line = _tidy_line(tenant_line) if tenant_line != line else line
            unquoted = _drop_quoted_narration(line)
            line = _tidy_line(unquoted) if unquoted != line else line
        if (not fenced) and _leaky_name_value(line):
            continue  # FIX5c: `Device: …`, `Model: <raw id>`, `Token: …` — an internal value in a name/value line goes whole
        machinery = bool(INTERNAL_TERMS.search(line))
        hit = machinery or ((not fenced) and _latin_narration(line))
        if not hit:
            item = _LIST_ITEM.match(line)
            prefix, body = (item.group(1), item.group(2)) if item else ("", line)
            parts = [part for part in _SENTENCE_END.split(body) if part and part.strip()]
            if len(parts) > 1 and any(_latin_narration(part) or _leaky_name_value(part) for part in parts):
                body = _join_kept([part for part in parts if _keep_sentence(part)])
                if not body.strip():
                    if not item and machinery:
                        kept.append("")
                    continue
                kept.append(_tidy_line(prefix + body.lstrip() if item else body))
            else:
                kept.append(line)
            continue
        if line.lstrip().startswith("|"):
            table_dropped = True
            continue
        if fenced:
            continue  # a line of code or JSON that names the machinery (*_sha256 …) goes; the rest of the block stays as written
        item = _LIST_ITEM.match(line)
        prefix, body = (item.group(1), item.group(2)) if item else ("", line)
        parts = [part for part in _SENTENCE_END.split(body) if part and part.strip()]
        if len(parts) > 1:
            body = _join_kept([part for part in parts if _keep_sentence(part)])
        else:
            body = "" if not _keep_sentence(body) else body
        if not body.strip():
            if not item and machinery:
                kept.append("")
            continue
        kept.append(_tidy_line(prefix + body.lstrip() if item else body))
    cleaned = re.sub(r"```[a-z]*\s*[{}\[\],\s]*```", "", "\n".join(kept))  # a block emptied (or reduced to brackets) by the line filter
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or OMITTED_STEP

_TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".csv", ".md", ".yaml", ".yml", ".py"}


def scrub_text(value: str) -> str:
    text = str(value)
    for pattern in _SCRUB_PATTERNS:
        text = pattern.sub(HIDDEN, text)
    return text


def scrub(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {scrub_text(key) if isinstance(key, str) else key: scrub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(item) for item in value]
    return value


def scrub_file(path: Path) -> None:
    """Rewrite a tenant-bound text artifact in place when it carries HPC details."""

    path = Path(path)
    if path.suffix.casefold() not in _TEXT_SUFFIXES or path.is_symlink() or not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return
    cleaned = scrub_text(text)
    if cleaned != text:
        path.write_text(cleaned, encoding="utf-8")


# -- service binding (X6) --

def enabled() -> dict[str, Any] | None:
    config = gpu_admission.active()
    return config if config and config.get("tenant_training") else None


def byoa_state_root() -> Path:
    return Path(os.environ.get("WORKSTATION_BYOA_STATE_ROOT") or r"C:\ProgramData\EvoMind\byoa")


def service_identity(config: dict[str, Any]) -> dict[str, Any]:
    """The operator's managed HPC identity for job 98270, read strictly from hpc-binding.json."""

    tenant = str(config.get("service_tenant_id") or "")
    path = byoa_state_root() / "tenants" / tenant / "hpc-binding.json"
    try:
        binding = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise TenantTrainingError("binding_invalid") from None
    job_id, generation = binding.get("job_id"), binding.get("allocation_generation")
    owner = str(binding.get("username") or binding.get("owner_principal_id") or "")
    if (not isinstance(binding, dict) or binding.get("schema") != "evomind.tenant_hpc_binding.v1"
            or binding.get("tenant_id") != tenant or binding.get("state") != "active"
            or isinstance(job_id, bool) or job_id != gpu_admission.ALLOWED_HPC_JOB_ID
            or isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0
            or binding.get("credential_profile") != f"{tenant}_job{job_id}_g{generation}"
            or not _PROFILE_INSTANCE.fullmatch(str(binding.get("profile_instance_id") or ""))
            or not _BINDING_ID.fullmatch(str(binding.get("allocation_binding_id") or ""))
            or not _OWNER.fullmatch(owner)):
        raise TenantTrainingError("binding_invalid")
    return {"tenant_id": tenant, "owner_principal_id": owner, "job_id": job_id,
            "credential_profile": binding["credential_profile"], "allocation_generation": generation,
            "profile_instance_id": binding["profile_instance_id"], "allocation_binding_id": binding["allocation_binding_id"]}


def pool_node_identity(config: dict[str, Any], job_id: int) -> dict[str, Any]:
    """2026-11-05: the managed HPC identity of an additional pool card, read strictly from ``pool-nodes/job<id>.json``.

    Only cards listed in gpu-pool.json ``nodes`` qualify; the record must be active and name this tenant's profile for
    exactly that job. Anything else raises ``binding_invalid`` (the card is then not schedulable)."""

    tenant = str(config.get("service_tenant_id") or "")
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id not in gpu_admission.POOL_HPC_JOB_IDS or job_id == gpu_admission.ALLOWED_HPC_JOB_ID:
        raise TenantTrainingError("binding_invalid")
    path = byoa_state_root() / "tenants" / tenant / "pool-nodes" / f"job{job_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise TenantTrainingError("binding_invalid") from None
    generation = record.get("allocation_generation") if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get("schema") != "evomind.tenant_hpc_pool_node.v1"
            or record.get("tenant_id") != tenant or record.get("state") != "active" or record.get("job_id") != job_id
            or isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0
            or record.get("credential_profile") != f"{tenant}_job{job_id}_g{generation}"
            or not _PROFILE_INSTANCE.fullmatch(str(record.get("profile_instance_id") or ""))
            or not _BINDING_ID.fullmatch(str(record.get("allocation_binding_id") or ""))):
        raise TenantTrainingError("binding_invalid")
    return {"tenant_id": tenant, "job_id": job_id, "credential_profile": record["credential_profile"],
            "allocation_generation": generation, "profile_instance_id": record["profile_instance_id"],
            "allocation_binding_id": record["allocation_binding_id"]}


# -- tenant data (inputs/ only) --

def resolve_input(workspace_root: Path, relative: str) -> Path:
    """A regular, non-symlink file under this Run's inputs/ — nothing else."""

    raw = str(relative or "")
    if not raw or "\x00" in raw or raw.startswith(("/", "\\", "~")) or re.match(r"^[A-Za-z]:", raw) or "$" in raw:
        raise TenantTrainingError("data_path_rejected")
    parts = raw.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts) or parts[0] != "inputs" or len(parts) < 2:
        raise TenantTrainingError("data_path_rejected")
    root = Path(workspace_root).resolve(strict=True)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise TenantTrainingError("data_path_rejected")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root / "inputs")
    except (OSError, ValueError):
        raise TenantTrainingError("data_path_rejected") from None
    if not resolved.is_file():
        raise TenantTrainingError("data_path_rejected")
    if resolved.stat().st_size > MAX_DATA_BYTES:
        raise TenantTrainingError("data_too_large")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_shape(shape: tuple[int, ...]) -> None:
    cells = 1
    for size in shape:
        cells *= int(size)
    if (shape and shape[0] > MAX_ROWS) or (len(shape) > 1 and shape[1] > MAX_COLUMNS) or cells > MAX_CELLS:
        raise TenantTrainingError("data_too_many_cells")


def _check_npz(path: Path) -> None:
    """Sizes and shapes come from the archive directory and the .npy headers, before anything is inflated."""

    import numpy as np

    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_NPZ_UNCOMPRESSED_BYTES:
            raise TenantTrainingError("data_too_many_cells")
        for name in ("x.npy", "y.npy"):
            with archive.open(name) as member:
                version = np.lib.format.read_magic(member)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _fortran, _dtype = reader(member)
            _check_shape(shape)


class _BoundedRecords:
    """Lines for ``csv.reader``, refused once one record passes the cap: csv builds a whole row before any
    cell cap can run, at about 20 times the text size. ``chars`` spans the lines of a quoted multi-line record."""

    def __init__(self, handle) -> None:
        self.handle = handle
        self.chars = 0

    def __iter__(self) -> "_BoundedRecords":
        return self

    def __next__(self) -> str:
        line = self.handle.readline(MAX_CSV_RECORD_CHARS - self.chars + 1)
        if not line:
            raise StopIteration
        self.chars += len(line)
        if self.chars > MAX_CSV_RECORD_CHARS:
            raise TenantTrainingError("data_too_many_cells")
        return line


def _read_csv(path: Path, target_column: str | None):
    """Streams into one flat array: cells cost 8 bytes each, and the caps stop the read as soon as they hit."""

    import numpy as np

    values = array("d")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        lines = _BoundedRecords(handle)
        reader = csv.reader(lines)
        header = next(reader)
        lines.chars = 0
        if len(header) > MAX_COLUMNS:
            raise TenantTrainingError("data_too_many_cells")
        target = header.index(target_column) if target_column else len(header) - 1
        rows = 0
        for row in reader:
            lines.chars = 0
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError("shape")
            rows += 1
            if rows > MAX_ROWS or rows * len(header) > MAX_CELLS:
                raise TenantTrainingError("data_too_many_cells")
            values.extend(float(cell) for cell in row)
    table = np.frombuffer(values, dtype=np.float64).reshape(rows, len(header))
    return np.delete(table, target, axis=1).astype(np.float32), table[:, target].copy()


def load_data(path: Path, *, task: str, target_column: str | None = None):
    import numpy as np

    try:
        if path.suffix.casefold() == ".npz":
            _check_npz(path)
            with np.load(path, allow_pickle=False) as data:
                x = np.asarray(data["x"], dtype=np.float32)
                y = np.asarray(data["y"]).reshape(-1)
        elif path.suffix.casefold() == ".csv":
            x, y = _read_csv(path, target_column)
        else:
            raise ValueError("suffix")
    except TenantTrainingError:
        raise
    except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError, StopIteration,
            csv.Error, zipfile.BadZipFile, zlib.error):
        raise TenantTrainingError("data_invalid") from None
    if x.ndim != 2 or x.shape[0] < 10 or x.shape[1] < 1 or len(y) != len(x) or not np.isfinite(x).all() or not np.isfinite(y.astype(np.float64)).all():
        raise TenantTrainingError("data_invalid")
    classes = 0
    if task == "classification":
        if np.any(y != np.round(y)) or np.any(y < 0):
            raise TenantTrainingError("data_invalid")
        y = y.astype(np.int64)
        classes = int(y.max()) + 1
        if not 2 <= classes <= 1000:
            raise TenantTrainingError("data_invalid")
    elif task == "regression":
        y = y.astype(np.float32)
    else:
        raise TenantTrainingError("config_invalid")
    return x, y, classes


def prepare(workspace_root: Path, config: dict[str, Any], protocol_id: str, *, vram_budget_mb: int | None = None,
            warm_start: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the approved configuration + tenant data; return the arrays and trainer config.

    ADDENDUM 2 §6: sizing fields the configuration leaves out are chosen by ``tenant_sizing.suggest``
    from the data and the VRAM budget; fields it sets are used as approved.

    Phase 3: ``model_family`` 'auto' (default) adds a cross-validated search over at most 6 candidates, the first
    being exactly the approved / suggested setting (``warm_start`` = the best matching past experience). 'mlp' trains
    that one setting only. A dataset too large for the search budget falls back to 'mlp'.
    """

    from .training_control import training_config
    from . import tenant_search
    import numpy as np

    if not isinstance(config, dict) or set(config) - TENANT_CONFIG_FIELDS or not isinstance(config.get("data_path"), str):
        raise TenantTrainingError("config_invalid")
    task = config.get("task")
    target = config.get("target_column")
    family = config.get("model_family", "auto")
    if task not in {"classification", "regression"} or (target is not None and not isinstance(target, str)) or family not in tenant_search.MODEL_FAMILIES:
        raise TenantTrainingError("config_invalid")
    path = resolve_input(workspace_root, config["data_path"])
    if sha256(path) != protocol_id:
        raise TenantTrainingError("data_changed")
    x, y, classes = load_data(path, task=task, target_column=target)
    seed = config.get("seed", 0) if type(config.get("seed", 0)) is int else 0
    order = np.random.default_rng(seed).permutation(len(x))
    holdout = max(1, len(x) // 5)
    train_index, eval_index = order[holdout:], order[:holdout]
    protocol = {"id": protocol_id, "contract": {"task": task, **({"classes": classes} if classes else {})},
                "training": {"x": x[train_index]}}
    sizing_keys = ("hidden_sizes", "epochs", "batch_size", "learning_rate")
    chosen = {key: config[key] for key in config if key in {*sizing_keys, "seed"}}
    sizing = None
    if any(key not in config for key in sizing_keys):
        from .tenant_sizing import suggest
        try:
            sizing = suggest(len(train_index), x.shape[1], classes, vram_budget_mb=int(vram_budget_mb or 4096), task=task)
        except ValueError:
            raise TenantTrainingError("config_invalid") from None
        chosen = {**{key: sizing[key] for key in sizing_keys}, **chosen}
    try:
        trainer = training_config(chosen, protocol)
    except ValueError:
        raise TenantTrainingError("config_invalid") from None
    trainer["class_weight"] = "none"
    applied = "mlp"
    if family == "auto" and tenant_search.fits_budget(len(train_index), trainer):
        seed_value = trainer["seed"]
        options = tenant_search.candidates(trainer, task=task, warm=warm_start,
                                           validate=lambda candidate: training_config({**candidate, "seed": seed_value}, protocol))
        if len(options) > 1:
            trainer["search"] = tenant_search.search_block(task, options)
            applied = "auto"
    return {"x": x[train_index], "y": y[train_index], "evaluation_x": x[eval_index], "evaluation_ids": eval_index,
            "evaluation_y": y[eval_index], "trainer_config": trainer, "task": task, "classes": classes, "data_name": path.name,
            "sizing": sizing, "model_family": applied}
