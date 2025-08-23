import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


TIME_RE_HHMMSS_MS = re.compile(r"^(\d{2}):(\d{2}):(\d{2})([\.,](\d{1,3}))?$")
TIME_LINE_RE = re.compile(r'^\s*\[(.+?)\s*-\s*(.+?)\]:\s*(.+?)\s*


def assemble_srt_from_minimal_segments(
    segment_outputs: List[str],
    offsets_ms: List[int],
    segment_durations_ms: Optional[List[int]] = None,
    total_duration_ms: Optional[int] = None,
) -> str:
    """Combine multiple segment minimal outputs into a single SRT string with robustness.

    - segment_outputs: list of JSONL strings (one per segment)
    - offsets_ms: segment start offsets in ms (same length)
    - segment_durations_ms: optional per-segment durations, used to detect absolute times
    - total_duration_ms: optional overall length for clamping
    """
    entries: List[Tuple[int, int, str]] = []
    for idx, (text, offset) in enumerate(zip(segment_outputs, offsets_ms)):
        items = parse_minimal_lines(text)
        if not items:
            continue
        seg_len = None
        if segment_durations_ms is not None and idx < len(segment_durations_ms):
            seg_len = max(0, int(segment_durations_ms[idx]))

        # Detect absolute vs relative timing
        assume_absolute = False
        if seg_len is not None and seg_len > 0:
            votes_abs = 0
            votes_rel = 0
            tolerance = 1500  # 1.5s tolerance
            for it in items:
                if it["end_ms"] > seg_len + tolerance or it["start_ms"] > seg_len + tolerance:
                    votes_abs += 1
                else:
                    votes_rel += 1
            assume_absolute = votes_abs > votes_rel

        for it in items:
            if assume_absolute:
                start = it["start_ms"]
                end = it["end_ms"]
            else:
                start = it["start_ms"] + offset
                end = it["end_ms"] + offset
            entries.append((start, end, it["text"]))

    # Sort globally by time
    entries.sort(key=lambda t: (t[0], t[1]))

    # Post-process: clamp to total, drop too-short, and dedup near-duplicates across edges
    clamped: List[Tuple[int, int, str]] = []
    min_dur_ms = 600
    if total_duration_ms is not None:
        total_max = max(0, int(total_duration_ms))
    else:
        total_max = None

    for start, end, content in entries:
        if total_max is not None:
            start = min(max(0, start), total_max)
            end = min(max(0, end), total_max)
        if end <= start:
            continue
        if end - start < min_dur_ms:
            continue
        clamped.append((start, end, content))

    deduped: List[Tuple[int, int, str]] = []
    last_norm = None
    last_entry: Optional[Tuple[int, int, str]] = None
    for start, end, content in clamped:
        norm = _normalize_text_for_compare(content)
        if last_entry is not None and norm == last_norm:
            # If overlapping significantly, merge by extending end; otherwise keep both
            ls, le, lt = last_entry
            overlap = min(le, end) - max(ls, start)
            if overlap > 0 or abs(start - ls) <= 1000:
                last_entry = (ls, max(le, end), lt)
                deduped[-1] = last_entry
                continue
        deduped.append((start, end, content))
        last_entry = (start, end, content)
        last_norm = norm

    # Build SRT
    srt_lines: List[str] = []
    for idx, (start, end, content) in enumerate(deduped, start=1):
        srt_lines.append(str(idx))
        srt_lines.append(f"{ms_to_hhmmssms(start)} --> {ms_to_hhmmssms(end)}")
        srt_lines.append(content)
        srt_lines.append("")
    return ("\n".join(srt_lines)).strip() + "\n"


)


def parse_time_value_to_ms(value: Any) -> Optional[int]:
    """Parse flexible time value into milliseconds.

    Accepts:
    - integer milliseconds (e.g., 12345)
    - float seconds (e.g., 12.345)
    - string HH:MM:SS(.mmm or ,mmm) (e.g., "00:01:02.345" or "00:01:02,345")
    - string seconds or milliseconds (heuristic): "12.345" -> seconds, "12345" -> ms
    Returns None if cannot parse.
    """
    if value is None:
        return None
    # Numeric types
    if isinstance(value, (int,)):
        # assume milliseconds
        return max(0, int(value))
    if isinstance(value, float):
        # assume seconds
        return max(0, int(round(value * 1000)))
    # String types
    if isinstance(value, str):
        v = value.strip()
        # HH:MM:SS(.mmm|,mmm)
        m = TIME_RE_HHMMSS_MS.match(v)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2))
            ss = int(m.group(3))
            ms_part = m.group(5)
            ms = int(ms_part) if ms_part is not None else 0
            # Normalize ms width
            if ms_part is not None and len(ms_part) < 3:
                ms = int(ms_part.ljust(3, "0"))
            return ((hh * 60 + mm) * 60 + ss) * 1000 + ms
        # Bare number string -> decide seconds vs ms
        try:
            if "." in v or "," in v:
                v = v.replace(",", ".")
                f = float(v)
                return max(0, int(round(f * 1000)))
            else:
                i = int(v)
                # Heuristic: treat >= 3600000 as ms; otherwise if < 100000 treat as ms too
                # This avoids misclassifying typical small values.
                return max(0, i)
        except Exception:
            return None
    return None


def ms_to_hhmmssms(total_ms: int) -> str:
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
    """Lowercase, Unicode-normalize, strip punctuation/marks, collapse spaces."""
    import unicodedata as _ud
    s2 = _ud.normalize("NFKC", s).lower()
    cleaned_chars: List[str] = []
    for ch in s2:
        cat = _ud.category(ch)
        # Drop punctuation (P*) and marks/diacritics (M*)
        if cat.startswith("P") or cat.startswith("M"):
            continue
        cleaned_chars.append(ch)
    s2 = "".join(cleaned_chars)
    s2 = " ".join(s2.split())
    return s2


def parse_minimal_lines(text: str) -> List[Dict[str, Any]]:
    items = []
    if not text or not text.strip():
        return items
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = TIME_LINE_RE.match(line)
        if not m:
            continue
        start_raw, end_raw, text_val = m.groups()
        start_ms = parse_time_value_to_ms(start_raw)
        end_ms = parse_time_value_to_ms(end_raw)
        if start_ms is None or end_ms is None or not text_val.strip():
            continue
        items.append({"start_ms": start_ms, "end_ms": end_ms, "text": text_val.strip()})
    
    items.sort(key=lambda d: (d["start_ms"], d["end_ms"]))
    normalized = []
    for it in items:
        start_ms = max(0, int(it["start_ms"]))
        end_ms = max(0, int(it["end_ms"]))
        if end_ms <= start_ms:
            continue
        text_val = str(it["text"]).strip()
        if not text_val:
            continue
        normalized.append({"start_ms": start_ms, "end_ms": end_ms, "text": text_val})
    return normalized


def assemble_srt_from_minimal_segments(
    segment_outputs: List[str],
    offsets_ms: List[int],
    segment_durations_ms: Optional[List[int]] = None,
    total_duration_ms: Optional[int] = None,
) -> str:
    """Combine multiple segment minimal outputs into a single SRT string with robustness.

    - segment_outputs: list of JSONL strings (one per segment)
    - offsets_ms: segment start offsets in ms (same length)
    - segment_durations_ms: optional per-segment durations, used to detect absolute times
    - total_duration_ms: optional overall length for clamping
    """
    entries: List[Tuple[int, int, str]] = []
    for idx, (text, offset) in enumerate(zip(segment_outputs, offsets_ms)):
        items = parse_minimal_lines(text)
        if not items:
            continue
        seg_len = None
        if segment_durations_ms is not None and idx < len(segment_durations_ms):
            seg_len = max(0, int(segment_durations_ms[idx]))

        # Detect absolute vs relative timing
        assume_absolute = False
        if seg_len is not None and seg_len > 0:
            votes_abs = 0
            votes_rel = 0
            tolerance = 1500  # 1.5s tolerance
            for it in items:
                if it["end_ms"] > seg_len + tolerance or it["start_ms"] > seg_len + tolerance:
                    votes_abs += 1
                else:
                    votes_rel += 1
            assume_absolute = votes_abs > votes_rel

        for it in items:
            if assume_absolute:
                start = it["start_ms"]
                end = it["end_ms"]
            else:
                start = it["start_ms"] + offset
                end = it["end_ms"] + offset
            entries.append((start, end, it["text"]))

    # Sort globally by time
    entries.sort(key=lambda t: (t[0], t[1]))

    # Post-process: clamp to total, drop too-short, and dedup near-duplicates across edges
    clamped: List[Tuple[int, int, str]] = []
    min_dur_ms = 600
    if total_duration_ms is not None:
        total_max = max(0, int(total_duration_ms))
    else:
        total_max = None

    for start, end, content in entries:
        if total_max is not None:
            start = min(max(0, start), total_max)
            end = min(max(0, end), total_max)
        if end <= start:
            continue
        if end - start < min_dur_ms:
            continue
        clamped.append((start, end, content))

    deduped: List[Tuple[int, int, str]] = []
    last_norm = None
    last_entry: Optional[Tuple[int, int, str]] = None
    for start, end, content in clamped:
        norm = _normalize_text_for_compare(content)
        if last_entry is not None and norm == last_norm:
            # If overlapping significantly, merge by extending end; otherwise keep both
            ls, le, lt = last_entry
            overlap = min(le, end) - max(ls, start)
            if overlap > 0 or abs(start - ls) <= 1000:
                last_entry = (ls, max(le, end), lt)
                deduped[-1] = last_entry
                continue
        deduped.append((start, end, content))
        last_entry = (start, end, content)
        last_norm = norm

    # Build SRT
    srt_lines: List[str] = []
    for idx, (start, end, content) in enumerate(deduped, start=1):
        srt_lines.append(str(idx))
        srt_lines.append(f"{ms_to_hhmmssms(start)} --> {ms_to_hhmmssms(end)}")
        srt_lines.append(content)
        srt_lines.append("")
    return ("\n".join(srt_lines)).strip() + "\n"


