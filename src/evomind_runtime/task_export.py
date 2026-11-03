"""Conversation → Markdown export. Read-only; technical identifiers never leave the server."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from urllib.parse import quote

HIDDEN = '[已隐藏]'
_QUOTED = r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
# ASCII-only boundaries: in Python `\w` also matches CJK, which would let "路径/hpc2hdd/…" through.
_EDGE = r'(?<![A-Za-z0-9_.\-/\\@\]])'
# Stand-ins for a leading / trailing "\b" that also split ASCII from CJK ("密钥sk-…", "任务utask_…完成").
_WB = r'(?<![A-Za-z0-9_])'
_WE = r'(?![A-Za-z0-9_])'
# A path runs until whitespace, a quote, a bracket or CJK punctuation; a trailing "." or ":" stays prose.
_BODY = r'[^\s`\'"<>()\[\]{}，。；：！？、（）《》「」,;]*(?<![.:])'
_SECRET_NAME = r'(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|key|token|secret|password|passwd|pwd|authorization|cookie|credentials?)'
# Runs before everything else: whole blocks and URIs, then secret-named keys whose value may be quoted and contain spaces.
_KEYED = (
    (re.compile(r'-----BEGIN [A-Z0-9 ]+-----.*?(?:-----END [A-Z0-9 ]+-----|\Z)', re.S), HIDDEN),
    (re.compile(r'(?i)' + _WB + r'file://[^\s`\'"<>)\]]*'), HIDDEN),
    # Linear: the scheme is possessive and at most 32 chars, so each start costs O(32); with no anchor a scheme glued to
    # CJK, "_" or a long word still matches its last ≤32 chars and \1 keeps them. The userinfo runs are possessive; their
    # stop characters are excluded from the class, so nothing that used to match is lost. The second branch is an empty
    # username with a non-empty password ("redis://:pw@host"); it starts only where the first cannot, so both stay O(n).
    # A longer scheme (33-255 chars) is tried only at the start of a scheme-char run, so it adds one O(255) scan per run and
    # catches runs whose last 32 chars hold no letter ("a" + "1" * 40 + "://").
    (re.compile(r'(?i)((?:(?<![a-z0-9+.\-])[a-z][a-z0-9+.\-]{32,254}+|[a-z][a-z0-9+.\-]{0,31}+)://)'
                r'(?:[^\s/@:]++:[^\s/@]*+|:[^\s/@]++)@'), r'\1' + HIDDEN + '@'),
    (re.compile(r'(?i)(["\'])(' + _SECRET_NAME + r')\1(\s*[:=]\s*)(?:' + _QUOTED + r'|[^\s,;}\]]+)'), r'\1\2\1\3' + HIDDEN),
    # Unquoted names: a bare "key" needs a prefix (FOO_KEY, private_key) so prose like "key: idea" stays.
    (re.compile(r'(?i)(?<![A-Za-z0-9_\-])((?:[A-Za-z0-9]+[_-])+(?:api[_-]?key|key|token|secret|password|passwd|pwd|credentials?)'
                r'|api[_-]?key|token|secret|password|passwd|cookie|authorization)(\s*[:=]\s*|\s+(?=hf_|sk-|gh[pousr]_))'
                r'(?:' + _QUOTED + r'|[^\s,;]+)'), r'\1\2' + HIDDEN),
)
_SECRETS = (
    re.compile(r'(?i)' + _WB + r'(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}'),
    re.compile(_WB + r'Bearer\s+(?!(?:of|the|and)' + _WE + r')[A-Za-z0-9._~+/=-]{4,}'),
    re.compile(r'(?i)' + _WB + r'(?:set-cookie|cookie|authorization)' + _WE + r'\s*[:=]\s*[^\s,;]+(?:;\s*[^\s,;]+)*'),
    re.compile(_WB + r'(?:sk|pk|rk)-[A-Za-z0-9_-]{8,}'),
    re.compile(_WB + r'sk-(?!learn' + _WE + r')[A-Za-z0-9_-]{3,}'),
    re.compile(_WB + r'(?:gh[pousr]_|github_pat_|xox[abprs][-_]|hf_|glpat-)[A-Za-z0-9_-]{8,}'),
    re.compile(_WB + r'AIza[0-9A-Za-z_-]{20,}'),
    re.compile(_WB + r'AKIA[0-9A-Z]{16}' + _WE),
    re.compile(_WB + r'eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}'),
)
_TECHNICAL = (
    re.compile(r'(?i)' + _WB + r'error_class' + _WE + r'\s*[:=]?\s*[^\s,;]*'),
    re.compile(_WB + r'[A-Z][A-Za-z0-9_]*(?:Error|Exception)(?:/[A-Z][A-Za-z0-9_]*)*'),
    re.compile(_WB + r'(?:run|utask|conversation|attachment|artifact|project|tenant|session|upload|approval|turn|tool)_[A-Za-z0-9]{6,}' + _WE),
    re.compile(r'(?i)' + _WB + r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' + _WE),
    re.compile(r'(?i)' + _WB + r'[a-f0-9]{32,}' + _WE),
    re.compile(r'claude_lt4net_temp'),
)
_OCTET = r'(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)'
_PRIVATE_IP = (r'(?:10(?:\.' + _OCTET + r'){3}|127(?:\.' + _OCTET + r'){3}|192\.168(?:\.' + _OCTET + r'){2}'
               r'|172\.(?:1[6-9]|2\d|3[01])(?:\.' + _OCTET + r'){2}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.' + _OCTET + r'){2}'
               r'|169\.254(?:\.' + _OCTET + r'){2})')
# Internal hosts: private-use suffixes and HPC node / login names (node97323, login01, *.hpc.*).
_INTERNAL_HOST = (r'(?:[A-Za-z0-9-]+\.)+(?:internal|local|lan|localdomain|intranet|corp|home\.arpa)(?!\.?[A-Za-z0-9-])'
                  r'|(?:node\d{3,}|(?:login|gpu|cn|compute|mgmt)\d{2,})(?:\.[A-Za-z0-9-]+)*'
                  r'|(?:[A-Za-z0-9-]+\.)+hpc(?:\.[A-Za-z0-9-]+)+')
# Server paths in every form we have seen in chats. A URL keeps its path: the edge skips "host/" and "]/".
_PATHS = (
    # scp/ssh targets: user@host:/path, or host:/path when the host has a digit or a dot (not "log:/", not "C:/").
    re.compile(_EDGE + r'(?:[A-Za-z0-9_.\-]+@[A-Za-z0-9_.\-]+|(?=[A-Za-z0-9_\-]*[0-9.])[A-Za-z0-9_\-][A-Za-z0-9_.\-]+):/(?!/)' + _BODY),
    re.compile(_EDGE + r'~/' + _BODY),
    re.compile(r'(?:\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_][A-Za-z0-9_]*%)[\\/]' + _BODY),
    re.compile(r'(?<![A-Za-z0-9_\\])\\\\[A-Za-z0-9_.$\-]+\\' + _BODY),
    # Windows drive paths; a following word joins when it continues the path ("Program Files\…") or is a file name ("x y.txt").
    re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]' + _BODY + r'(?: [A-Za-z0-9_\-]+(?:[\\/]' + _BODY + r'|\.[A-Za-z0-9]{1,8}(?![A-Za-z0-9])))*'),
    re.compile(_EDGE + r'/[A-Za-z0-9_.~\-]' + _BODY),
    re.compile(_EDGE + r'(?:hpc2hdd|workspace|home|root|scratch|mnt|srv|opt)/' + _BODY),
    re.compile(r'(?<![A-Za-z0-9_.\-])(?:' + _PRIVATE_IP + r')(?::\d{1,5})?(?![0-9]|\.[0-9])'),
    re.compile(r'(?<![A-Za-z0-9_.\-])(?:' + _INTERNAL_HOST + r')(?::\d{1,5})?(?![A-Za-z0-9_\-])'),
)
_MODEL_PREFIX = re.compile(r'\*\*本轮实际模型\*\*：`[^`]*`（来自(?:受管 model_observed 事件|模型输出)）\s*')
_ILLEGAL_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')


def redact_export_text(text: str) -> str:
    value = _MODEL_PREFIX.sub('', str(text or ''))
    for pattern, replacement in _KEYED:
        value = pattern.sub(replacement, value)
    for pattern in (*_SECRETS, *_TECHNICAL, *_PATHS):
        value = pattern.sub(HIDDEN, value)
    return value


def export_filename(title: str, today: str) -> str:
    stem = _ILLEGAL_NAME.sub('-', title).strip(' .') or '对话'
    return f'{stem[:100]}-{today}.md'


def content_disposition(name: str, today: str) -> str:
    """RFC 5987: an ASCII fallback plus the exact UTF-8 name."""
    return f"attachment; filename=\"conversation-{today}.md\"; filename*=UTF-8''{quote(name, safe='')}"


def render_markdown(title: str, turns: list[tuple[str, str]], attachments: list[str], artifacts: list[str], today: str) -> str:
    """turns: (role, text) with role 'user' | 'assistant'. Every free-text field is redacted."""
    lines = [f'# {redact_export_text(title)}', '', f'导出日期：{today}', '']
    for role, text in turns:
        body = redact_export_text(text).strip()
        if not body:
            continue
        lines += ['**你**' if role == 'user' else '**DeepEvo**', '', body, '']
    for label, names in (('附件', attachments), ('产物', artifacts)):
        names = [redact_export_text(name) for name in dict.fromkeys(names) if name]
        if names:
            lines += [f'{label}：', '', *[f'- {name}' for name in names], '']
    return '\n'.join(lines).rstrip() + '\n'


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')
