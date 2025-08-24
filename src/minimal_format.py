"""Functions for parsing and assembling SRT subtitle files."""

import re
from typing import Any, Dict, List, Optional, Tuple

TIME_RE_HHMMSS_MS = re.compile(r"^(\d{2}):(\d{2}):(\d{2})([.,](\d{1,3}))?$")
TIME_LINE_RE = re.compile(r'^\s*\[(.+?)\s*-\s*(.+?)\]:\s*(.+?)\s*$')

def parse_time_value_to_ms(value: Any) -> Optional[int]:
    """Parse a flexible time value into milliseconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0, int(round(value * 1000)) if isinstance(value, float) else int(value))
    if isinstance(value, str):
        v = value.strip()
        m = TIME_RE_HHMMSS_MS.match(v)
        if m:
            hh, mm, ss, _, ms_part = m.groups()
            ms = int((ms_part or '0').ljust(3, '0'))
            return ((int(hh) * 60 + int(mm)) * 60 + int(ss)) * 1000 + ms
        try:
            if '.' in v or ',' in v:
                return max(0, int(round(float(v.replace(',', '.')) * 1000)))
            return max(0, int(v))
        except ValueError:
            return None
    return None

def ms_to_hhmmssms(total_ms: int) -> str:
    """Convert milliseconds to an SRT timestamp string."""
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
    """Parse the simple timed-line format into a list of subtitle dictionaries."""
    items = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        m = TIME_LINE_RE.match(line.strip())
        if not m:
            continue
        start_raw, end_raw, text_val = m.groups()
        start_ms = parse_time_value_to_ms(start_raw)
        end_ms = parse_time_value_to_ms(end_raw)
        if start_ms is None or end_ms is None or not text_val.strip() or end_ms <= start_ms:
            continue
        items.append({"start_ms": start_ms, "end_ms": end_ms, "text": text_val.strip()})
    items.sort(key=lambda d: (d["start_ms"], d["end_ms"]))
    return items

def assemble_srt_from_minimal_segments(
    segment_outputs: List[str],
    offsets_ms: List[int],
    segment_durations_ms: Optional[List[int]] = None,
    total_duration_ms: Optional[int] = None,
) -> str:
    """Assemble a final SRT file from multiple timed-line segment outputs by appending and shifting."""
    entries = []
    current_offset = 0
    for idx, text in enumerate(segment_outputs):
        items = parse_minimal_lines(text)
        if not items:
            continue
        # Use provided offset if available, else cumulative
        offset = offsets_ms[idx] if idx < len(offsets_ms) else current_offset
        for it in items:
            start = it["start_ms"] + offset
            end = it["end_ms"] + offset
            entries.append((start, end, it["text"]))
        # Update cumulative for next (if durations provided)
        if segment_durations_ms and idx < len(segment_durations_ms):
            current_offset += segment_durations_ms[idx]
    entries.sort(key=lambda t: (t[0], t[1]))
    srt_lines = [f"{i+1}\n{ms_to_hhmmssms(s)} --> {ms_to_hhmmssms(e)}\n{c}\n" 
                 for i, (s, e, c) in enumerate(entries)]
    return "\n".join(srt_lines)
