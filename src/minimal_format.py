"""XmYsZms-only minimal format parsing and SRT assembly."""

import re
from typing import Any, Dict, List, Optional, Tuple

# Strict bracketed minimal line: [<start>-<end>]: text
TIME_LINE_RE = re.compile(r'^\s*\[(.+?)\s*-\s*(.+?)\]\s*:\s*(.+?)\s*$')

# Strict token: optional minutes, optional seconds, optional milliseconds
TIME_TOKEN_RE = re.compile(
    r"^\s*(?:(\d+)\s*m)?\s*(?:(\d{1,2})\s*s)?\s*(?:(\d{1,3})\s*ms)?\s*$",
    re.IGNORECASE,
)

def parse_time_value_to_ms(value: Any) -> Optional[int]:
    """Parse XmYsZms-like token into milliseconds.

    Accepted examples:
    - 9m32s839ms
    - 1m2s
    - 32s
    - 839ms
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Treat as milliseconds if int, seconds if float
        return max(0, int(round(value)) if isinstance(value, int) else int(round(value * 1000)))
    if isinstance(value, str):
        v = value.strip()
        m = TIME_TOKEN_RE.match(v)
        if not m:
            return None
        mm = int(m.group(1) or 0)
        ss = int(m.group(2) or 0)
        ms_part = (m.group(3) or '0')
        ms = int(ms_part.ljust(3, '0'))
        return ((mm * 60) + ss) * 1000 + ms
    return None

def ms_to_hhmmssms(total_ms: int) -> str:
    """Convert milliseconds to a standard SRT timestamp string HH:MM:SS,ms."""
    if total_ms < 0:
        total_ms = 0
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _normalize_text_for_compare(s: str) -> str:
    """Normalize text for comparison by lowercasing and removing punctuation."""
    import unicodedata as _ud
    s2 = _ud.normalize("NFKC", s).lower()
    return " ".join("".join(c for c in s2 if not _ud.category(c).startswith(('P', 'M'))).split())

def parse_minimal_lines(text: str) -> List[Dict[str, Any]]:
    """Parse minimal lines strictly: [XmYsZms-XmYsZms]: text.

    - Tolerates surrounding quotes on the whole line.
    - Ignores lines that don't match or have invalid times.
    """
    items: List[Dict[str, Any]] = []
    if not text:
        return items

    for raw in text.strip().splitlines():
        line = raw.strip()
        # Tolerate lines wrapped in matching quotes (some models echo quotes from instructions)
        if (len(line) >= 2 and ((line[0] == line[-1]) and line[0] in ('"', "'", '“', '”'))):
            # Handle symmetric fancy quotes as well
            if (line[0] == '“' and line[-1] == '”') or (line[0] in ('"', "'") and line[-1] == line[0]):
                line = line[1:-1].strip()
        if not line:
            continue

        start_raw: Optional[str] = None
        end_raw: Optional[str] = None
        content: Optional[str] = None

        m = TIME_LINE_RE.match(line)
        if m:
            start_raw, end_raw, content = m.groups()

        if start_raw is None or end_raw is None or content is None:
            continue

        start_ms = parse_time_value_to_ms(start_raw)
        end_ms = parse_time_value_to_ms(end_raw)
        if start_ms is None or end_ms is None:
            continue
        text_val = content.strip()
        if not text_val or end_ms <= start_ms:
            continue
        items.append({"start_ms": start_ms, "end_ms": end_ms, "text": text_val})

    items.sort(key=lambda d: (d["start_ms"], d["end_ms"]))
    return items

def format_ms_xmys(total_ms: int) -> str:
    """Format milliseconds as XmYsZms, omitting zero components."""
    if total_ms < 0:
        total_ms = 0
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    m = total_sec // 60
    parts: List[str] = []
    if m:
        parts.append(f"{m}m")
    if s or m:
        parts.append(f"{s}s")
    if ms:
        parts.append(f"{ms}ms")
    # If everything zero, return 0s
    return "".join(parts) if parts else "0s"


def clean_minimal_text(text: str) -> str:
    """Validate and fix minimal transcript/translation lines.

    - Keeps only lines of the form [start-end]: text with XmYsZms times.
    - Strips surrounding quotes.
    - Ensures end > start; if swapped, swap back.
    - Re-emits in canonical formatting using XmYsZms.
    """
    items = parse_minimal_lines(text)
    cleaned: List[str] = []
    for it in items:
        s = it["start_ms"]
        e = it["end_ms"]
        t = it["text"].strip()
        if not t:
            continue
        if e <= s:
            s, e = e, s  # swap
        cleaned.append(f"[{format_ms_xmys(s)}-{format_ms_xmys(e)}]: {t}")
    return "\n".join(cleaned)


def assemble_srt_from_minimal_segments(
    segment_outputs: List[str],
    offsets_ms: List[int],
) -> str:
    """Assemble final SRT by strictly parsing minimal lines and offsetting.

    No rebasing, clamping, or legacy fallbacks.
    """
    entries: List[Tuple[int, int, str]] = []
    for idx, text in enumerate(segment_outputs):
        items = parse_minimal_lines(text)
        if not items:
            continue
        off = offsets_ms[idx]
        for it in items:
            s = it["start_ms"] + off
            e = it["end_ms"] + off
            if e <= s:
                continue
            entries.append((s, e, it["text"]))
    entries.sort(key=lambda t: (t[0], t[1]))
    srt_lines = [f"{i+1}\n{ms_to_hhmmssms(s)} --> {ms_to_hhmmssms(e)}\n{c}\n" for i, (s, e, c) in enumerate(entries)]
    return "\n".join(srt_lines)
