import os
import subprocess
from typing import List, Tuple, Optional

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
    """Deprecated. Not used by the pipeline; use split_audio_by_duration() instead."""
    raise NotImplementedError(
        "split_audio_ffmpeg is deprecated and not implemented; use split_audio_by_duration instead."
    )


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

def split_audio_by_duration(
    input_audio: str,
    segment_minutes: int,
    tmp_dir: str = "tmp_segments",
    verbose: bool = False,
) -> Tuple[List[str], List[int], List[int], int]:
    """Split audio into fixed-duration segments (>= segment_minutes) without overlap.

    Returns (segment_paths, start_offsets_ms, segment_durations_ms, total_duration_ms).
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

    # Always use segment muxer with stream copy to avoid decoding errors entirely
    segment_seconds = min_minutes * 60
    out_pattern = os.path.join(tmp_dir, "seg_%02d.m4a")
    seg_cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", input_audio,
        "-vn", "-sn", "-dn",
        "-map", "0:a:0",
        "-c:a", "copy", "-bsf:a", "aac_adtstoasc",
        "-f", "segment", "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        out_pattern,
    ]
    if verbose:
        subprocess.run(seg_cmd, check=True)
    else:
        subprocess.run(seg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    segment_paths = [
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if f.startswith("seg_") and f.endswith(".m4a")
    ]
    segment_paths.sort()
    if not segment_paths:
        raise RuntimeError("Segmentation failed: no segment files produced")

    # Compute durations and cumulative offsets from actual files
    durations_ms: List[int] = []
    offsets_ms: List[int] = []
    running = 0
    for p in segment_paths:
        d = get_audio_duration_ms(p)
        dur = d if d is not None else segment_ms
        durations_ms.append(dur)
        offsets_ms.append(running)
        running += dur

    return segment_paths, offsets_ms, durations_ms, total_ms


