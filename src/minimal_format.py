"""JSON-based minimal format parsing and SRT assembly."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

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
        return max(
            0, int(round(value)) if isinstance(value, int) else int(round(value * 1000))
        )
    if isinstance(value, str):
        v = value.strip()
        m = TIME_TOKEN_RE.match(v)
        if not m:
            return None
        mm = int(m.group(1) or 0)
        ss = int(m.group(2) or 0)
        ms_part = m.group(3) or "0"
        ms = int(ms_part.ljust(3, "0"))
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


def parse_json_segments(text: str) -> List[Dict[str, Any]]:
    """Parse JSON list of segments: [{"start": "...", "end": "...", "text": "..."}].

    Handles Markdown code blocks (```json ... ```) if present.
    """
    text = text.strip()
    if not text:
        return []

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find the first '[' and last ']'
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                data = json.loads(text[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    items: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        start_raw = entry.get("start")
        end_raw = entry.get("end")
        content = entry.get("text")

        if start_raw is None or end_raw is None or content is None:
            continue

        start_ms = parse_time_value_to_ms(start_raw)
        end_ms = parse_time_value_to_ms(end_raw)

        if start_ms is None or end_ms is None:
            continue

        text_val = str(content).strip()
        if not text_val or end_ms <= start_ms:
            continue

        items.append({"start_ms": start_ms, "end_ms": end_ms, "text": text_val})

    items.sort(key=lambda d: (d["start_ms"], d["end_ms"]))
    return items


def clean_json_text(text: str) -> str:
    """Validate and fix JSON transcript/translation.

    Re-emits valid JSON list.
    """
    items = parse_json_segments(text)
    cleaned: List[Dict[str, str]] = []
    for it in items:
        s = it["start_ms"]
        e = it["end_ms"]
        t = it["text"]
        if e <= s:
            s, e = e, s  # swap
        cleaned.append(
            {"start": format_ms_xmys(s), "end": format_ms_xmys(e), "text": t}
        )
    return json.dumps(cleaned, ensure_ascii=False, indent=2)


def assemble_srt_from_json_segments(
    segment_outputs: List[str],
    offsets_ms: List[int],
) -> str:
    """Assemble final SRT from JSON segment outputs."""
    entries: List[Tuple[int, int, str]] = []
    for idx, text in enumerate(segment_outputs):
        items = parse_json_segments(text)
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
    srt_lines = [
        f"{i + 1}\n{ms_to_hhmmssms(s)} --> {ms_to_hhmmssms(e)}\n{c}\n"
        for i, (s, e, c) in enumerate(entries)
    ]
    return "\n".join(srt_lines)
