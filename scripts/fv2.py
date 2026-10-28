"""fv2.py - shared session-feature code (version 2) for the lab, external and live paths.

Unit of analysis
    An attacker INPUT LINE: a non-empty cowrie.command.input event that is not a
    command-substitution artefact. Cowrie (checked: v2.5.0, v2.6.1, v2.9.16 and the four
    COW160 builds) logs every $(...) / `...` inside a line as an extra command.input event
    (HoneyPotShell.run_subshell_command -> lineReceived); collapse_subshell() removes them.
    Cowrie's sh/bash command also runs its piped stdin through lineReceived. In the COW160
    builds v2-v4, 'passwd' prints "Changing password for <user>. Current password:", so
    'echo ... | passwd | bash' logs that prompt as an input line; drop_passwd_echo() removes
    it (the only piped-output artefact found in the exploration days).

Decision point
    The k-th input line. Every feature uses only events and keystrokes up to that line,
    so offline training sees exactly what the live classifier sees.

JSON features (lab and external)
    log_first_latency  log10 of seconds from login.success to the first line
    log_gap_median     log10 of the median gap between lines 1..k
    gap_cv             std / mean of those gaps (0 when every gap is below 1 ms)
    log_max_rate       log10 of max over lines i<=k of len(line_i) / gap_before_i  [chars/s]
                       (gap_before_1 = first latency). A typed line needs at least one
                       keystroke per character, so this is a lower bound on typing speed.

TTY features (lab and live only; Model B)
    single_char_ratio  share of keyboard-input writes that carry exactly one byte
    log_key_gap_median log10 of the median gap between writes inside a line (a line that
                       arrives in one write has no such gap -> 1 ms floor)
    edit_keys_per_line (backspace, DEL, TAB, Ctrl-U, Ctrl-W, ESC sequences) / k
    Missing TTY input is NaN (missing), never 0.

Campaign labels (external proxy labels; never used as features)
    sequence_hash(lines)                 exact whitespace-normalised line sequence
    sequence_hash(lines, template=True)  the same after template_line(): URLs, IPv4
                                         addresses, long letter+digit tokens (random names,
                                         hashes, keys) and numbers become placeholders
    family_hash(lines)                   template of the FIRST line only (cluster unit)
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import struct

import numpy as np

K_CANDIDATES = (3, 4, 5)
FLOOR = 0.001                      # s; floor for gaps and latencies
TTY_SLACK = 0.1                    # s; keystrokes up to t_k + slack belong to the prefix
JSON_FEATURES = ["log_first_latency", "log_gap_median", "gap_cv", "log_max_rate"]
TTY_FEATURES = ["single_char_ratio", "log_key_gap_median", "edit_keys_per_line"]

_NORM = re.compile(r"[\s'\"\\]")
_TTY_HDR = "<iLiiLL"
_TTY_HSZ = struct.calcsize(_TTY_HDR)
OP_WRITE = 3
TYPE_INPUT, TYPE_OUTPUT, TYPE_INTERACT = 1, 2, 3
_EDIT = (b"\x7f", b"\x08", b"\t", b"\x15", b"\x17", b"\x1b")


def _norm(text: str) -> str:
    return _NORM.sub("", text)


def collapse_subshell(cmds, max_gap: float = 1.0):
    """cmds: [(t, text)] sorted by t, empty texts already removed.
    An event is a substitution artefact when the current top-level line contains $( or a
    backtick, the event text (no whitespace/quotes/backslashes) is a substring of that line,
    and it arrives <= max_gap s after the previous event. Returns (lines, n_derived)."""
    lines, derived, top, top_sub, prev = [], 0, "", False, None
    for t, text in cmds:
        n = _norm(text)
        if top_sub and prev is not None and t - prev <= max_gap and n and n in top:
            derived += 1
        else:
            lines.append((t, text))
            top, top_sub = n, ("$(" in text or "`" in text)
        prev = t
    return lines, derived


_PIPE_SHELL = re.compile(r"\|\s*(?:/bin/)?(?:ba)?sh\b")


def drop_passwd_echo(lines, max_gap: float = 1.0):
    """Remove Cowrie's own passwd prompt when it was piped into sh/bash and logged as a line:
    text starts with 'Changing password for ', the previous line runs passwd and pipes into
    sh/bash, and the event arrives <= max_gap s later. Returns (lines, n_dropped)."""
    out, dropped = [], 0
    for t, text in lines:
        if out and text.startswith("Changing password for ") and t - out[-1][0] <= max_gap \
                and "passwd" in out[-1][1] and _PIPE_SHELL.search(out[-1][1]):
            dropped += 1
            continue
        out.append((t, text))
    return out, dropped


def input_lines(cmd_events, detail: bool = False):
    """cmd_events: [(t, raw_input)] in any order -> (lines, n_artefacts, n_empty).
    With detail=True the middle item is {'substitution': a, 'passwd_echo': b}."""
    ev = sorted(((t, (x or "").strip()) for t, x in cmd_events), key=lambda e: e[0])
    nonempty = [e for e in ev if e[1]]
    lines, derived = collapse_subshell(nonempty)
    lines, echoed = drop_passwd_echo(lines)
    arte = {"substitution": derived, "passwd_echo": echoed} if detail else derived + echoed
    return lines, arte, len(ev) - len(nonempty)


_URL = re.compile(r"\b(?:https?|ftp|tftp)://\S+")
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b")
_LONG = re.compile(r"[A-Za-z0-9+/=_.\-]{8,}")
_DIGIT, _ALPHA = re.compile(r"\d"), re.compile(r"[A-Za-z]")
_NUM = re.compile(r"\b\d+\b")


def template_line(text: str) -> str:
    s = _URL.sub("<URL>", text)
    s = _IPV4.sub("<IP>", s)
    s = _LONG.sub(lambda m: "<TOK>" if _DIGIT.search(m.group()) and _ALPHA.search(m.group())
                  else m.group(), s)
    s = _NUM.sub("<N>", s)
    return " ".join(s.split())


def sequence_hash(lines, template: bool = False) -> str:
    f = template_line if template else (lambda x: " ".join(x.split()))
    return hashlib.sha1("\n".join(f(x) for _, x in lines).encode()).hexdigest()[:16]


def family_hash(lines) -> str:
    """Cluster unit for external statistics: the template of the first input line."""
    return hashlib.sha1(template_line(lines[0][1]).encode()).hexdigest()[:16]


def _log10(x: float) -> float:
    return math.log10(max(x, FLOOR))


def json_prefix_features(lines, login_ts, k: int):
    """Features at the k-th line, or None when the session never reaches it."""
    if len(lines) < k:
        return None
    t = [x[0] for x in lines[:k]]
    texts = [x[1] for x in lines[:k]]
    gaps = np.diff(t)
    first = (t[0] - login_ts) if login_ts is not None else float("nan")
    mean = float(gaps.mean())
    rates = [len(texts[i]) / max(float(gaps[i - 1]), FLOOR) for i in range(1, k)]
    if not math.isnan(first):
        rates.append(len(texts[0]) / max(first, FLOOR))
    return {
        "log_first_latency": _log10(first) if not math.isnan(first) else float("nan"),
        "log_gap_median": _log10(float(np.median(gaps))),
        "gap_cv": float(gaps.std() / mean) if mean >= FLOOR else 0.0,
        "log_max_rate": math.log10(max(rates)),
        "t_decision": t[-1],
        "decision_delay": (t[-1] - login_ts) if login_ts is not None else float("nan"),
    }


def read_tty(path):
    """All write records of a Cowrie TTY log: [(t, direction, payload)]."""
    try:
        data = open(path, "rb").read()
    except OSError:
        return None
    out, i = [], 0
    while i + _TTY_HSZ <= len(data):
        op, _tty, length, direction, sec, usec = struct.unpack(_TTY_HDR, data[i:i + _TTY_HSZ])
        i += _TTY_HSZ
        payload = data[i:i + length]
        i += length
        if op == OP_WRITE:
            out.append((sec + usec / 1e6, direction, payload))
    return out


def tty_mode(records):
    """'exec' when the log holds an exec command, 'interactive' otherwise."""
    if records is None:
        return "unknown"
    return "exec" if any(d == TYPE_INTERACT for _, d, _ in records) else "interactive"


def tty_prefix_features(records, t_cut: float, k: int):
    nan = {f: float("nan") for f in TTY_FEATURES}
    if not records:
        return {**nan, "tty_input_writes": 0}
    inp = [(t, p) for t, d, p in records if d == TYPE_INPUT and t <= t_cut + TTY_SLACK]
    if not inp:
        return {**nan, "tty_input_writes": 0}
    sizes = np.array([len(p) for _, p in inp])
    intra = [b[0] - a[0] for a, b in zip(inp, inp[1:])
             if b"\r" not in a[1] and b"\n" not in a[1]]
    edits = sum(p.count(s) for _, p in inp for s in _EDIT)
    return {
        "single_char_ratio": float((sizes == 1).mean()),
        # whole-line writes have no gap between their characters -> floor (1 ms)
        "log_key_gap_median": _log10(float(np.median(intra))) if intra else _log10(0.0),
        "edit_keys_per_line": edits / k,
        "tty_input_writes": len(inp),
    }


def tty_path_status(closed_event, tty_dir):
    """Resolve the TTY log that belongs to THIS session.
    Unpatched Cowrie deletes a later session's log when its input matches an earlier one
    (duplicate=true) and points ttylog at the earlier session's file -> unusable."""
    if not closed_event:
        return None, "no_log_closed"
    base = os.path.basename(closed_event.get("ttylog") or "")
    if not base:
        return None, "no_ttylog_field"
    dup = closed_event.get("duplicate") in (True, "true", "True", 1)
    if dup and "-" not in base:                 # pre-patch duplicate: file is another session's
        return None, "dedup_deleted"
    path = os.path.join(tty_dir, base)
    return (path, "ok") if os.path.exists(path) else (None, "file_missing")
