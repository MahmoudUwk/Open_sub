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
    """Assemble a final SRT file from multiple timed-line segment outputs."""
    entries = []
    for idx, (text, offset) in enumerate(zip(segment_outputs, offsets_ms)):
        items = parse_minimal_lines(text)
        if not items:
            continue
        seg_len = segment_durations_ms[idx] if segment_durations_ms and idx < len(segment_durations_ms) else None
        assume_absolute = False
        if seg_len and seg_len > 0:
            votes_abs = sum(1 for it in items if it["end_ms"] > seg_len + 1500)
            assume_absolute = votes_abs > len(items) / 2
        for it in items:
            start = it["start_ms"] if assume_absolute else it["start_ms"] + offset
            end = it["end_ms"] if assume_absolute else it["end_ms"] + offset
            entries.append((start, end, it["text"]))
    entries.sort(key=lambda t: (t[0], t[1]))
    clamped = []
    total_max = total_duration_ms and int(total_duration_ms)
    for start, end, content in entries:
        if total_max is not None:
            start, end = min(start, total_max), min(end, total_max)
        if end - start >= 600:
            clamped.append((start, end, content))
    deduped = []
    last_entry: Optional[Tuple[int, int, str]] = None
    for start, end, content in clamped:
        norm = _normalize_text_for_compare(content)
        if last_entry and norm == _normalize_text_for_compare(last_entry[2]):
            ls, le, lt = last_entry
            if min(le, end) - max(ls, start) > 0 or abs(start - ls) <= 1000:
                last_entry = (ls, max(le, end), lt)
                deduped[-1] = last_entry
                continue
        deduped.append((start, end, content))
        last_entry = (start, end, content)
    srt_lines = [f"{i+1}\n{ms_to_hhmmssms(s)} --> {ms_to_hhmmssms(e)}\n{c}\n" 
                 for i, (s, e, c) in enumerate(deduped)]
    return "\n".join(srt_lines)
