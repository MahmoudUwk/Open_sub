import os
from typing import List, Tuple, Dict

from src.minimal_format import ms_to_hhmmssms, parse_time_value_to_ms

def parse_srt(path: str) -> List[Tuple[int,int,str]]:
    entries: List[Tuple[int,int,str]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        block: List[str] = []
        for line in f:
            if line.strip() == '' and block:
                _consume_block(block, entries)
                block = []
            else:
                block.append(line.rstrip('\n'))
        if block:
            _consume_block(block, entries)
    return entries

def _consume_block(block: List[str], out: List[Tuple[int,int,str]]):
    # block may be: [index?, time, text...]
    lines = [l.strip() for l in block if l.strip()]
    if not lines:
        return
    if lines and lines[0].isdigit():
        lines = lines[1:]
    if not lines:
        return
    ts = lines[0]
    text = " ".join(lines[1:]).strip()
    import re
    m = re.match(r"^(\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{1,3})$", ts)
    if not m:
        return
    s = parse_time_value_to_ms(m.group(1))
    e = parse_time_value_to_ms(m.group(2))
    if s is None or e is None or e <= s or not text:
        return
    out.append((s,e,text))


def validate_entries(entries: List[Tuple[int,int,str]],
                     max_duration_ms: int = 20*1000) -> Dict[str, List[int]]:
    issues: Dict[str, List[int]] = {
        'non_monotonic': [],
        'overlap': [],
        'negative_or_zero': [],
        'too_long': [],
        'empty_text': [],
    }
    prev_end = -1
    for i,(s,e,t) in enumerate(entries):
        if s >= e:
            issues['negative_or_zero'].append(i)
        if prev_end > s:
            issues['overlap'].append(i)
        if s < prev_end:
            issues['non_monotonic'].append(i)
        if (e - s) > max_duration_ms:
            issues['too_long'].append(i)
        if not t.strip():
            issues['empty_text'].append(i)
        prev_end = max(prev_end, e)
    return issues


def repair_monotonic(entries: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    # Ensure strictly non-overlapping by clamping starts
    fixed: List[Tuple[int,int,str]] = []
    prev_end = -1
    for (s,e,t) in entries:
        s2 = max(s, prev_end)
        e2 = max(e, s2+1)
        fixed.append((s2,e2,t))
        prev_end = e2
    return fixed

def split_long_entries(entries: List[Tuple[int,int,str]], max_ms: int = 8000) -> List[Tuple[int,int,str]]:
    """Split entries longer than max_ms into smaller chunks at punctuation or spaces.

    Note: This preserves the original start/end window by dividing proportionally by text length.
    """
    out: List[Tuple[int,int,str]] = []
    import re
    for (s,e,t) in entries:
        dur = e - s
        if dur <= max_ms or len(t.strip()) <= 1:
            out.append((s,e,t))
            continue
        # Split by sentence-ish boundaries, then fallback to words
        parts = [p.strip() for p in re.split(r"([.!?]+\s+)", t) if p and not p.isspace()]
        # Rejoin punctuation separators with preceding chunk
        merged: List[str] = []
        buf = ""
        for p in parts:
            if re.match(r"^[.!?]+\s+$", p):
                buf += p
                merged.append(buf.strip())
                buf = ""
            else:
                if buf:
                    buf += p
                else:
                    buf = p
        if buf:
            merged.append(buf.strip())
        if not merged:
            merged = [t]

        # If still one big chunk, split by words roughly to fit max_ms
        if len(merged) == 1 and dur > max_ms:
            words = t.split()
            target_chunks = max(2, int(round(dur / max_ms)))
            chunk_size = max(1, len(words) // target_chunks)
            merged = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        total_chars = sum(len(x) for x in merged) or 1
        acc = s
        for i,chunk in enumerate(merged):
            frac = len(chunk) / total_chars
            cdur = int(round(dur * frac))
            if i == len(merged) - 1:
                se = e
            else:
                se = min(e, acc + max(1, cdur))
            out.append((acc, se, chunk.strip()))
            acc = se
    return out


def write_srt(entries: List[Tuple[int,int,str]], out_path: str):
    lines: List[str] = []
    for i,(s,e,t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{ms_to_hhmmssms(s)} --> {ms_to_hhmmssms(e)}")
        lines.append(t)
        lines.append("")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def validate_and_optionally_fix(srt_path: str, out_fixed_path: str | None = None) -> Dict[str, List[int]]:
    entries = parse_srt(srt_path)
    issues = validate_entries(entries)
    if out_fixed_path:
        fixed = repair_monotonic(entries)
        fixed = split_long_entries(fixed, max_ms=8000)
        fixed = repair_monotonic(fixed)
        write_srt(fixed, out_fixed_path)
    return issues
