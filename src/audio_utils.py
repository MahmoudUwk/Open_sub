import os
import re
import subprocess
from typing import List, Tuple, Optional


TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})$")


def hhmmssms_to_ms(h: int, m: int, s: int, ms: int) -> int:
    return ((h * 60 + m) * 60 + s) * 1000 + ms


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


def format_seconds_for_ffmpeg(ms: int) -> str:
    """Format milliseconds to ffmpeg-friendly seconds with millisecond precision.

    ffmpeg accepts "HH:MM:SS.mmm" or seconds with decimals; we use seconds.decimals for simplicity.
    """
    if ms < 0:
        ms = 0
    seconds = ms / 1000.0
    # Use 3 decimal places (milliseconds)
    return f"{seconds:.3f}"


def split_audio_ffmpeg(
    input_audio: str,
    segment_minutes: int = 15,
    tmp_dir: str = "tmp_segments",
    verbose: bool = False,
) -> List[str]:
    """Split audio into <= segment_minutes chunks using ffmpeg; returns list of paths.

    Uses -f segment with a fixed duration to avoid API length issues.
    """
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    os.makedirs(tmp_dir, exist_ok=True)

    segment_seconds = segment_minutes * 60
    out_pattern = os.path.join(tmp_dir, "segment_%03d.m4a")

    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "aac", "-b:a", "48k",
        "-f", "segment", "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        out_pattern,
    ]

    if verbose:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith("segment_")]
    files.sort()
    return files


def get_audio_duration_ms(audio_path: str) -> Optional[int]:
    """Return audio duration in ms via ffprobe; None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", audio_path,
            ],
            capture_output=True, text=True, check=True,
        )
        val = result.stdout.strip()
        if not val:
            return None
        seconds = float(val)
        return int(round(seconds * 1000))
    except Exception:
        return None


def compute_segment_offsets_ms(segment_paths: List[str], fallback_segment_ms: int) -> List[int]:
    """Compute cumulative start offsets for each segment using real durations.

    The offset for segment i is sum of durations of segments [0..i-1]. If a duration
    cannot be determined, fallback_segment_ms is used.
    """
    offsets: List[int] = []
    running = 0
    for path in segment_paths:
        offsets.append(running)
        dur = get_audio_duration_ms(path)
        running += dur if dur is not None else fallback_segment_ms
    return offsets


def parse_srt_blocks(srt_text: str) -> List[Tuple[int, str, List[str]]]:
    """Parse SRT into list of (index, timestamp_line, text_lines)."""
    blocks: List[Tuple[int, str, List[str]]] = []
    lines = [line.rstrip("\n\r") for line in srt_text.splitlines()]
    i = 0
    while i < len(lines):
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i >= len(lines):
            break
        try:
            idx = int(lines[i].strip())
        except ValueError:
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        ts_line = lines[i]
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1
        blocks.append((idx, ts_line, text_lines))
    return blocks


def shift_timestamp_line(ts_line: str, offset_ms: int) -> str:
    m = TIMESTAMP_RE.match(ts_line.strip())
    if not m:
        return ts_line
    sh, sm, ss, sms, eh, em, es, ems = map(int, m.groups())
    start_ms = hhmmssms_to_ms(sh, sm, ss, sms) + offset_ms
    end_ms = hhmmssms_to_ms(eh, em, es, ems) + offset_ms
    return f"{ms_to_hhmmssms(start_ms)} --> {ms_to_hhmmssms(end_ms)}"


def combine_srts_with_offset(srts: List[str], segment_ms: List[int]) -> str:
    """Combine multiple SRT texts with per-segment start offsets (in ms)."""
    combined_lines: List[str] = []
    next_index = 1
    for srt_text, offset in zip(srts, segment_ms):
        blocks = parse_srt_blocks(srt_text)
        for _idx, ts_line, text_lines in blocks:
            combined_lines.append(str(next_index))
            combined_lines.append(shift_timestamp_line(ts_line, offset))
            combined_lines.extend(text_lines)
            combined_lines.append("")
            next_index += 1
    return "\n".join(combined_lines).strip() + "\n"


def split_audio_into_equal_segments(
    input_audio: str,
    num_segments: int,
    tmp_dir: str = "tmp_segments",
    verbose: bool = False,
) -> Tuple[List[str], List[int], int]:
    """Split audio into exactly num_segments segments with millisecond precision.

    Returns (segment_paths, start_offsets_ms, total_duration_ms).
    """
    if num_segments <= 0:
        raise ValueError("num_segments must be > 0")
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    os.makedirs(tmp_dir, exist_ok=True)

    total_ms = get_audio_duration_ms(input_audio)
    if total_ms is None:
        raise RuntimeError("Unable to determine audio duration for equal split")

    # Edge case: extremely short audio -> create 1 segment
    if total_ms < num_segments:
        num_segments = 1

    # Compute boundaries using integer math to avoid drift
    boundaries: List[int] = [int((i * total_ms) / num_segments) for i in range(num_segments)] + [total_ms]
    offsets_ms: List[int] = boundaries[:-1]

    segment_paths: List[str] = []
    for i in range(num_segments):
        start_ms = boundaries[i]
        end_ms = boundaries[i + 1]
        # Skip zero-length segments (shouldn't happen, but guard anyway)
        if end_ms <= start_ms:
            continue
        out_path = os.path.join(tmp_dir, f"seg_{i:02d}.m4a")
        # Re-encode for consistent sample rate and precise cutting
        cmd = [
            "ffmpeg", "-y", "-i", input_audio,
            "-vn",
            "-ss", format_seconds_for_ffmpeg(start_ms),
            "-to", format_seconds_for_ffmpeg(end_ms),
            "-ac", "1", "-ar", "16000",
            "-c:a", "aac", "-b:a", "48k",
            out_path,
        ]
        if verbose:
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        segment_paths.append(out_path)

    return segment_paths, offsets_ms, total_ms


def split_audio_by_duration(
    input_audio: str,
    segment_minutes: int,
    overlap_seconds: int = 2,
    tmp_dir: str = "tmp_segments",
    verbose: bool = False,
) -> Tuple[List[str], List[int], List[int], int]:
    """Split audio into fixed-duration segments (>= segment_minutes) with optional overlap.

    Returns (segment_paths, start_offsets_ms, segment_durations_ms, total_duration_ms).
    Overlap is applied between consecutive segments; the last segment is not extended.
    """
    if segment_minutes <= 0:
        raise ValueError("segment_minutes must be > 0")
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    os.makedirs(tmp_dir, exist_ok=True)

    total_ms_opt = get_audio_duration_ms(input_audio)
    if total_ms_opt is None:
        raise RuntimeError("Unable to determine audio duration for duration-based split")
    total_ms = int(total_ms_opt)

    # Enforce minimum 5 minutes per the user's requirement
    min_minutes = max(5, int(segment_minutes))
    segment_ms = min_minutes * 60 * 1000
    overlap_ms = max(0, int(overlap_seconds) * 1000)

    # Compute segment boundaries with overlap between consecutive chunks
    boundaries: List[Tuple[int, int]] = []
    start_ms = 0
    while start_ms < total_ms:
        end_ms = min(start_ms + segment_ms, total_ms)
        if end_ms <= start_ms:
            break
        boundaries.append((start_ms, end_ms))
        if end_ms >= total_ms:
            break
        # Next start overlaps by overlap_ms with previous end
        start_ms = max(0, end_ms - overlap_ms)

    segment_paths: List[str] = []
    offsets_ms: List[int] = []
    approx_durations_ms: List[int] = []

    for i, (start_ms, end_ms) in enumerate(boundaries):
        out_path = os.path.join(tmp_dir, f"seg_{i:02d}.m4a")
        cmd = [
            "ffmpeg", "-y", "-i", input_audio,
            "-vn",
            "-ss", format_seconds_for_ffmpeg(start_ms),
            "-to", format_seconds_for_ffmpeg(end_ms),
            "-ac", "1", "-ar", "16000",
            "-c:a", "aac", "-b:a", "48k",
            out_path,
        ]
        if verbose:
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        segment_paths.append(out_path)
        offsets_ms.append(start_ms)
        approx_durations_ms.append(max(0, end_ms - start_ms))

    # Refine durations using actual encoded file lengths when available
    durations_ms: List[int] = []
    for path, approx in zip(segment_paths, approx_durations_ms):
        d = get_audio_duration_ms(path)
        durations_ms.append(d if d is not None else approx)

    return segment_paths, offsets_ms, durations_ms, total_ms


