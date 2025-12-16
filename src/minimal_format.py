"""JSON-based minimal format parsing and SRT assembly."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Strict token: optional minutes, optional seconds, optional milliseconds
# Strict token: optional minutes, optional seconds, optional milliseconds
TIME_TOKEN_RE = re.compile(
    r"^\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?\s*(?:(\d+)\s*ms)?\s*$",
    re.IGNORECASE,
)
# New Compact Format Regex: [Start - End] Text
# Matches: [0m0s214ms - 0m2s174ms] Some text here
COMPACT_SEGMENT_RE = re.compile(
    r"^\s*\[\s*([a-zA-Z0-9\s:.,]+)\s*-\s*([a-zA-Z0-9\s:.,]+)\s*\]\s*(.+)$", re.MULTILINE
)

# Standard SRT-style timestamp: HH:MM:SS,mmm or H:MM:SS.mmm (hours optional)
# Updated to handle brackets, parentheses, and angle brackets
TIME_COLON_RE = re.compile(
    r"^\s*[\[\(\)<>]?(?:(\d{1,2}):)?(\d{1,2}):(\d{1,2})(?:[.,](\d{1,3}))?[\]\)\)>]?\s*$"
)
# MM:SS,mmm short-hand (no hours) - updated to handle brackets and parentheses
TIME_MINSEC_RE = re.compile(
    r"^\s*[\[\(\)<>]?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?[\]\)\)>]?\s*$"
)


def parse_time_value_to_ms(value: Any) -> Optional[int]:
    """Parse XmYsZms-like token into milliseconds.

    Accepted examples:
    - 9m32s839ms
    - 1m2s
    - 32s
    - 839ms
    - 01:02:03,456
    - 02:15.4 (MM:SS.mmm)
    - [00:01:30] (bracketed)
    - (00:01:30) (parentheses)
    - 00:01:30,500 (comma decimal)
    - 00:01:30.500 (dot decimal)
    - 1:30 (minutes:seconds)
    - 30 (seconds only)
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

        # Handle empty or whitespace-only strings
        if not v:
            return None

        # Try SRT-style colon format first (handles brackets, parentheses, and angle brackets)
        m_colon = TIME_COLON_RE.match(v)
        if m_colon:
            h = int(m_colon.group(1) or 0)
            m = int(m_colon.group(2) or 0)
            s = int(m_colon.group(3) or 0)
            frac = m_colon.group(4) or ""
            ms_frac = (
                int((frac + "000")[:3]) if frac else 0
            )  # treat as fractional seconds
            return ((h * 3600 + m * 60 + s) * 1000) + ms_frac

        # Try MM:SS format (handles brackets, parentheses, and angle brackets)
        m_minsec = TIME_MINSEC_RE.match(v)
        if m_minsec:
            m = int(m_minsec.group(1) or 0)
            s = int(m_minsec.group(2) or 0)
            frac = m_minsec.group(3) or ""
            ms_frac = int((frac + "000")[:3]) if frac else 0
            return ((m * 60 + s) * 1000) + ms_frac

        # Try to parse as integer (treat as seconds) - but not empty strings
        try:
            # Handle case where it's just a number (treat as seconds)
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                seconds = int(v)
                if seconds >= 0:
                    return seconds * 1000
        except (ValueError, IndexError):
            pass

        # Try XmYsZms format
        m = TIME_TOKEN_RE.match(v)
        if m:
            mm = int(m.group(1) or 0)
            ss = int(m.group(2) or 0)
            ms_part = m.group(3)
            # For explicit "ms" suffix treat digits as milliseconds, not a fractional second
            if ms_part is None or ms_part == "":
                ms = 0
            else:
                try:
                    ms = max(0, int(ms_part[:4]))  # guard against huge values
                except ValueError:
                    return None
            return ((mm * 60) + ss) * 1000 + ms

        # Try to extract timestamp from malformed brackets or other wrappers
        # Strip common wrapper characters and retry
        stripped = v.strip("[](){}\"'<>")
        if stripped != v:
            return parse_time_value_to_ms(stripped)

        return None
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
    total_min = total_sec // 60
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
        if not text_val:
            continue

        items.append({"start_ms": start_ms, "end_ms": end_ms, "text": text_val})

    items.sort(key=lambda d: (d["start_ms"], d["end_ms"]))
    return items


def parse_compact_segments(text: str) -> List[Dict[str, Any]]:
    """Parse segments in format: [Start - End] Text

    Example:
    [0m0s214ms - 0m2s174ms] Hello world
    """
    items: List[Dict[str, Any]] = []
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

    # Try parsing line by line
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = COMPACT_SEGMENT_RE.match(line)
        if m:
            start_str, end_str, content = m.groups()
            start_ms = parse_time_value_to_ms(start_str)
            end_ms = parse_time_value_to_ms(end_str)

            if start_ms is not None and end_ms is not None and content:
                items.append(
                    {"start_ms": start_ms, "end_ms": end_ms, "text": content.strip()}
                )

    return items


def parse_any_segments(text: str) -> List[Dict[str, Any]]:
    """Try to parse as compact format first, then fallback to JSON."""
    # Heuristic: does it look like JSON?
    stripped = text.strip()
    if stripped.startswith("[") and (stripped.endswith("]") or "{" in stripped):
        # Try JSON first
        try:
            items = parse_json_segments(stripped)
            if items:
                return items
        except Exception:
            pass

    # Try compact format
    items = parse_compact_segments(stripped)
    if items:
        return items

    # Fallback to JSON if compact failed (maybe it was malformed JSON)
    return parse_json_segments(stripped)


def clean_segment_text(text: str) -> str:
    """Validate and normalize segment text to the new compact format.

    Output format:
    [XmYsZms - XmYsZms] Text
    """
    items = parse_any_segments(text)
    lines: List[str] = []

    for it in items:
        s = it["start_ms"]
        e = it["end_ms"]
        t = it["text"]

        if e <= s:
            # Swap and enforce a minimal 1ms duration to keep the segment usable
            s, e = e, s
            if e <= s:
                e = s + 1

        # Format as compact string
        start_fmt = format_ms_xmys(s)
        end_fmt = format_ms_xmys(e)
        lines.append(f"[{start_fmt} - {end_fmt}] {t}")

    return "\n".join(lines)


# Alias for backward compatibility if needed, but we should update callers
clean_json_text = clean_segment_text


def assemble_srt_from_segments(
    segment_outputs: List[str],
    offsets_ms: List[int],
    durations_ms: Optional[List[int]] = None,
    log_adjustments: bool = True,
    monotonic_tolerance_ms: int = 50,
    max_segment_length_ms: int = 30000,  # Cap individual subtitle lines to 30s to prevent hangs
) -> str:
    """Assemble final SRT from segment outputs (compact or JSON).

    Clamps each segment’s timestamps to its own window and enforces global monotonic order
    to avoid overlaps/gaps from model drift.
    """
    clamp_count = 0
    monotonic_adjust = 0
    entries: List[Tuple[int, int, str]] = []

    for idx, text in enumerate(segment_outputs):
        items = parse_any_segments(text)
        if not items:
            continue

        segment_offset = offsets_ms[idx]
        # segment_duration not strictly used for validation here, but could be

        for it in items:
            # CRITICAL FIX: These are RELATIVE timestamps within the segment
            # They need to be converted to global time by adding the segment offset
            relative_start = it["start_ms"]
            relative_end = it["end_ms"]

            # Convert to global timestamps
            global_start = segment_offset + relative_start
            global_end = segment_offset + relative_end

            # Store original for comparison
            original_start, original_end = global_start, global_end

            # Ensure minimum duration
            if global_end <= global_start:
                global_end = global_start + 1000  # 1 second minimum

            entries.append((global_start, global_end, it["text"]))

    entries.sort(key=lambda t: (t[0], t[1]))
    monotonic: List[Tuple[int, int, str]] = []
    prev_end = 0

    for s, e, c in entries:
        # 1. Cap excessive duration (e.g. 5 minutes for one word)
        # This is a common cause of "hanging" subtitles
        if (e - s) > max_segment_length_ms:
            e = s + max_segment_length_ms
            clamp_count += 1

        # 2. Strict Monotonicity with Priority to Current Start
        # If the current line starts BEFORE the previous ended, we have an overlap.
        # Strategy:
        # - If overlap is small (within tolerance), push Start forward (standard sync)
        # - If overlap is LARGE (bad model timing), CLAMP the PREVIOUS end instead of delaying current.
        #   Delaying current indefinitely is what causes "hanging" (one bad segment pushes everything else)

        if s < prev_end:
            overlap = prev_end - s
            if (
                overlap > 2000
            ):  # If overlap is > 2 seconds, assume previous segment ran too long
                # Fix previous segment in the 'monotonic' list if possible
                if monotonic:
                    last_s, last_e, last_c = monotonic[-1]
                    # Cap the last one to end where this one starts
                    # But don't make it shorter than 100ms if possible
                    new_last_e = max(last_s + 100, s)
                    monotonic[-1] = (last_s, new_last_e, last_c)
                    prev_end = new_last_e
                    monotonic_adjust += 1

                # Now take s as is (or max(s, prev_end))
                s_adj = max(s, prev_end)
            else:
                # Small overlap, just push current start forward
                s_adj = prev_end
        else:
            s_adj = s

        e_adj = max(e, s_adj + 100)  # Ensure at least 100ms duration

        if (s_adj, e_adj) != (s, e):
            monotonic_adjust += 1

        monotonic.append((s_adj, e_adj, c))
        prev_end = e_adj

    srt_lines = [
        f"{i + 1}\n{ms_to_hhmmssms(s)} --> {ms_to_hhmmssms(e)}\n{c}\n"
        for i, (s, e, c) in enumerate(monotonic)
    ]

    if log_adjustments and (clamp_count or monotonic_adjust):
        print(
            f"[ASSEMBLE] adjusted segments: clamp={clamp_count}, monotonic={monotonic_adjust}, total_entries={len(monotonic)}"
        )
    return "\n".join(srt_lines)


# Module-level alias for backward compatibility
assemble_srt_from_json_segments = assemble_srt_from_segments
