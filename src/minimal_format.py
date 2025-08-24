"""Functions for parsing and assembling SRT subtitle files."""

import re
from typing import Any, Dict, List, Optional, Tuple

TIME_RE_HHMMSS_MS = re.compile(r"^(\d{2}):(\d{2}):(\d{2})([.,](\d{1,3}))?$")
TIME_LINE_RE = re.compile(r'^\s*\[(.+?)\s*-\s*(.+?)\]\s*:\s*(.+?)\s*$')

def parse_time_value_to_ms(value: Any) -> Optional[int]:
    """Parse a flexible time value into milliseconds.

    Accepts:
    - HH:MM:SS[,.]ms
    - MM:SS[,.]ms
    - SS[,.]ms
    - integer milliseconds or seconds (floats)
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0, int(round(value * 1000)) if isinstance(value, float) else int(value))
    if isinstance(value, str):
        v = value.strip()
        # Normalize Arabic-Indic digits and punctuation to ASCII
        # U+0660-0669 (٠١٢٣٤٥٦٧٨٩), U+06F0-06F9 (۰۱۲۳۴۵۶۷۸۹)
        def _norm_digits(s: str) -> str:
            out_chars = []
            for ch in s:
                code = ord(ch)
                if 0x0660 <= code <= 0x0669:
                    out_chars.append(chr(ord('0') + (code - 0x0660)))
                elif 0x06F0 <= code <= 0x06F9:
                    out_chars.append(chr(ord('0') + (code - 0x06F0)))
                elif ch in ('،',):  # Arabic comma
                    out_chars.append(',')
                else:
                    out_chars.append(ch)
            return ''.join(out_chars)
        v = _norm_digits(v)
        # Try strict HH:MM:SS first
        m = TIME_RE_HHMMSS_MS.match(v)
        if m:
            hh, mm, ss, _, ms_part = m.groups()
            ms = int((ms_part or '0').ljust(3, '0'))
            return ((int(hh) * 60 + int(mm)) * 60 + int(ss)) * 1000 + ms
        # Flexible parsing for MM:SS, SS and decimal seconds
        vv = v.replace(',', '.')
        if ':' in vv:
            parts = vv.split(':')
            try:
                if len(parts) == 3:
                    h = int(parts[0])
                    m2 = int(parts[1])
                    s = float(parts[2])
                elif len(parts) == 2:
                    h = 0
                    m2 = int(parts[0])
                    s = float(parts[1])
                else:
                    h = 0
                    m2 = 0
                    s = float(parts[-1])
                total_ms = int(round(((h * 60 + m2) * 60 + s) * 1000))
                return max(0, total_ms)
            except ValueError:
                return None
        try:
            # Plain seconds or milliseconds as float or int
            if '.' in vv:
                return max(0, int(round(float(vv) * 1000)))
            return max(0, int(vv))
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
    """Parse relaxed timed-line formats into a unified list of subtitle dicts.

    Accepted forms per line:
    - [start - end]: text            (canonical minimal format)
    - [start - end] text             (missing colon)
    - start - end: text              (no brackets)
    - start to end: text             (no brackets, 'to')
    - start --> end text             (single-line arrow)
    Where start/end may be HH:MM:SS,mmm | HH:MM:SS.mmm | MM:SS | SS(.mmm)
    """
    items: List[Dict[str, Any]] = []
    if not text:
        return items

    # Precompiled relaxed patterns
    BRACKET_NO_COLON = re.compile(r"^\s*\[(.+?)\s*-\s*(.+?)\]\s*(.+?)\s*$")
    NO_BRACKETS_COLON = re.compile(r"^\s*(.+?)\s*-\s*(.+?)\s*:\s*(.+?)\s*$")
    NO_BRACKETS_TO = re.compile(r"^\s*(.+?)\s+to\s+(.+?)\s*:\s*(.+?)\s*$", re.IGNORECASE)
    ARROW_SINGLE_LINE = re.compile(r"^\s*(.+?)\s*-->\s*(.+?)\s*(.+?)?\s*$")

    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue

        start_raw: Optional[str] = None
        end_raw: Optional[str] = None
        content: Optional[str] = None

        m = TIME_LINE_RE.match(line)
        if m:
            start_raw, end_raw, content = m.groups()
        else:
            m2 = BRACKET_NO_COLON.match(line)
            if m2:
                start_raw, end_raw, content = m2.groups()
            else:
                m3 = NO_BRACKETS_COLON.match(line)
                if m3:
                    start_raw, end_raw, content = m3.groups()
                else:
                    m4 = NO_BRACKETS_TO.match(line)
                    if m4:
                        start_raw, end_raw, content = m4.groups()
                    else:
                        m5 = ARROW_SINGLE_LINE.match(line)
                        if m5:
                            start_raw, end_raw, maybe_text = m5.groups()
                            content = (maybe_text or "").strip()

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

def parse_srt_blocks(text: str) -> List[Dict[str, Any]]:
    """Parse standard SRT into items.

    Accepts blocks of the form:
        index\n
        HH:MM:SS,mmm --> HH:MM:SS,mmm\n
        text lines...\n
    """
    items: List[Dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        if not lines:
            continue
        # Optional numeric index in first line
        if len(lines) >= 2 and re.match(r"^\d+$", lines[0]):
            ts_line = lines[1]
            content_lines = lines[2:]
        else:
            ts_line = lines[0] if lines else ""
            content_lines = lines[1:]
        m = re.match(r"^(\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{1,3})$", ts_line)
        if not m:
            continue
        start_ms = parse_time_value_to_ms(m.group(1))
        end_ms = parse_time_value_to_ms(m.group(2))
        txt = " ".join(content_lines).strip()
        if start_ms is None or end_ms is None or end_ms <= start_ms or not txt:
            continue
        items.append({"start_ms": start_ms, "end_ms": end_ms, "text": txt})
    items.sort(key=lambda d: (d["start_ms"], d["end_ms"]))
    return items

def assemble_srt_from_minimal_segments(
    segment_outputs: List[str],
    offsets_ms: List[int],
) -> str:
    """Assemble a final SRT file from multiple timed-line segment outputs by appending and shifting."""
    entries = []
    for idx, text in enumerate(segment_outputs):
        items = parse_minimal_lines(text)
        if not items:
            # Fallback: try parsing as standard SRT blocks (models sometimes drift format)
            items = parse_srt_blocks(text)
        if not items:
            continue
        offset = offsets_ms[idx]  # Directly use the provided offset
        # Infer this segment's length using the next offset when available
        seg_len = None
        if idx + 1 < len(offsets_ms):
            nxt = offsets_ms[idx + 1]
            if nxt > offset:
                seg_len = nxt - offset
        # Normalize local times that drift beyond the segment window (e.g., 10:01 within a 10-min segment)
        def _normalize_local(ms: int) -> int:
            # Clamp to [0, seg_len-1] rather than wrapping by modulo to avoid
            # creating artificial time jumps across segment boundaries.
            if seg_len is None:
                return 0 if ms < 0 else ms
            if ms < 0:
                return 0
            if ms >= seg_len:
                return max(0, seg_len - 1)
            return ms
        for it in items:
            ls = _normalize_local(it["start_ms"])
            le = _normalize_local(it["end_ms"])
            # Ensure end after start minimally
            if le <= ls:
                le = ls + 1
            start = ls + offset
            end = le + offset
            entries.append((start, end, it["text"]))
    entries.sort(key=lambda t: (t[0], t[1]))
    srt_lines = [f"{i+1}\n{ms_to_hhmmssms(s)} --> {ms_to_hhmmssms(e)}\n{c}\n"
                 for i, (s, e, c) in enumerate(entries)]
    return "\n".join(srt_lines)
