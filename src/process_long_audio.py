import os
import time
import asyncio
from typing import List, Tuple, Optional

from .audio_utils import (
    split_audio_into_equal_segments,
    split_audio_by_duration,
)
from .upload_transcribe_translate_audio import transcribe_translate_minimal
from .minimal_format import assemble_srt_from_minimal_segments


def _guess_mime_type_from_extension(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3",):
        return "audio/mp3"
    if ext in (".m4a", ".mp4", ".aac"):
        return "audio/mp4"
    if ext in (".wav",):
        return "audio/wav"
    if ext in (".flac",):
        return "audio/flac"
    if ext in (".ogg", ".oga"):
        return "audio/ogg"
    return "application/octet-stream"


async def _call_model_for_segment(
    segment_path: str,
    source_language: str,
    target_language: str,
    model: str,
    mime_type: str,
    verbose: bool,
) -> str:
    # Read bytes in thread
    def _read_bytes(p: str) -> bytes:
        with open(p, "rb") as f:
            return f.read()

    audio_bytes = await asyncio.to_thread(_read_bytes, segment_path)
    # Run model in thread (SDK is sync)
    return await asyncio.to_thread(
        transcribe_translate_minimal,
        audio_bytes,
        mime_type,
        source_language,
        target_language,
        model,
        3,
        2.0,
        verbose,
    )


def process_audio_equal_segments(
    input_audio: str,
    source_language: str,
    target_language: str,
    model: str = "gemini-2.5-flash",
    num_segments: int = 10,
    tmp_dir: str = "tmp_segments",
    output_dir: str = "output_srt",
    cleanup: bool = True,
    verbose: bool = True,
) -> str:
    """Split audio into equal num_segments, call model in parallel, assemble SRT."""
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    if verbose:
        print(f"[1/4] Splitting into {num_segments} equal segments...", flush=True)
    segment_paths, offsets_ms, total_ms = split_audio_into_equal_segments(
        input_audio=input_audio,
        num_segments=num_segments,
        tmp_dir=tmp_dir,
        verbose=False,
    )
    if not segment_paths:
        raise RuntimeError("No segments produced.")
    if verbose:
        print(f"  - Done: {len(segment_paths)} segments", flush=True)

    if verbose:
        print("[2/4] Launching parallel model calls...", flush=True)
    mime_type = _guess_mime_type_from_extension(segment_paths[0])

    async def _run_all() -> List[str]:
        # Respect a soft limit of 10 concurrent to reduce 429s in equal-segment path
        sem = asyncio.Semaphore(10)

        async def _guarded(seg_path: str) -> str:
            async with sem:
                return await _call_model_for_segment(
                    seg_path,
                    source_language,
                    target_language,
                    model,
                    mime_type,
                    verbose,
                )

        tasks = [_guarded(seg) for seg in segment_paths]
        results = await asyncio.gather(*tasks)
        return list(results)

    minimal_outputs: List[str] = asyncio.run(_run_all())

    if verbose:
        print("[3/4] Assembling global SRT...", flush=True)
    # Use true offsets; durations not available here
    srt_text = assemble_srt_from_minimal_segments(
        minimal_outputs,
        offsets_ms,
        segment_durations_ms=None,
        total_duration_ms=total_ms,
    )

    base = os.path.splitext(os.path.basename(input_audio))[0]
    out_path = os.path.join(output_dir, base + ".srt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    if verbose:
        print(f"  - Wrote {out_path}", flush=True)

    if cleanup:
        try:
            # Remove tmp_dir tree
            import shutil
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    if verbose:
        print(f"Total: {time.time() - t0:.1f}s", flush=True)
    return out_path


def process_audio_fixed_duration(
    input_audio: str,
    source_language: str,
    target_language: str,
    model: str = "gemini-2.5-flash",
    min_segment_minutes: int = 5,
    segment_overlap_seconds: int = 2,
    rate_limit_per_minute: int = 10,
    tmp_dir: str = "tmp_segments",
    output_dir: str = "output_srt",
    cleanup: bool = True,
    verbose: bool = True,
) -> str:
    """Split by duration (>=5 minutes), call model under rate limit, assemble SRT.

    Ensures at most `rate_limit_per_minute` requests are started per minute, in batches.
    Saves successful outputs and retries rate-limited failures in subsequent batches.
    """
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    if verbose:
        print(
            f"[1/5] Splitting by duration (>= {min_segment_minutes} min, overlap {segment_overlap_seconds}s)...",
            flush=True,
        )
    seg_paths, offsets_ms, seg_durations_ms, total_ms = split_audio_by_duration(
        input_audio=input_audio,
        segment_minutes=min_segment_minutes,
        overlap_seconds=segment_overlap_seconds,
        tmp_dir=tmp_dir,
        verbose=False,
    )
    if not seg_paths:
        raise RuntimeError("No segments produced.")
    if verbose:
        total_min = total_ms / 60000.0
        print(f"  - Done: {len(seg_paths)} segments (total ~{total_min:.1f} min)", flush=True)

    mime_type = _guess_mime_type_from_extension(seg_paths[0])

    async def _run_rate_limited() -> List[str]:
        results: List[Optional[str]] = [None] * len(seg_paths)
        retries_left: List[int] = [2] * len(seg_paths)  # Orchestrator-level retries after SDK retries
        pending_indices = list(range(len(seg_paths)))

        async def _launch_batch(indices: List[int]) -> List[object]:
            tasks = [
                _call_model_for_segment(
                    seg_paths[i],
                    source_language,
                    target_language,
                    model,
                    mime_type,
                    verbose,
                )
                for i in indices
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        while pending_indices:
            batch = pending_indices[: max(1, min(rate_limit_per_minute, len(pending_indices)))]
            if verbose:
                print(f"[2/5] Calling model for batch of {len(batch)} (rate={rate_limit_per_minute}/min)...", flush=True)
            window_start = time.time()
            outcomes = await _launch_batch(batch)

            next_pending: List[int] = []
            for idx, outcome in zip(batch, outcomes):
                if isinstance(outcome, Exception):
                    msg = str(outcome)
                    # Retry only on rate limit hints; otherwise mark empty
                    from .upload_transcribe_translate_audio import RATE_LIMIT_ERRORS
                    if any(tok.lower() in msg.lower() for tok in RATE_LIMIT_ERRORS) and retries_left[idx] > 0:
                        retries_left[idx] -= 1
                        next_pending.append(idx)
                        if verbose:
                            print(f"    - seg {idx}: rate-limited, will retry ({retries_left[idx]} left)", flush=True)
                    else:
                        results[idx] = ""
                        if verbose:
                            print(f"    - seg {idx}: failed (non-retryable): {msg}", flush=True)
                else:
                    results[idx] = str(outcome)

            # Rollover remaining and retriable indices
            remaining = [i for i in pending_indices if i not in batch]
            pending_indices = remaining + next_pending

            # Enforce 60s window between batch starts when there is more work
            if pending_indices:
                elapsed = time.time() - window_start
                sleep_for = max(0.0, 60.0 - elapsed)
                if verbose:
                    print(f"[2/5] Waiting {sleep_for:.1f}s to respect rate limit...", flush=True)
                await asyncio.sleep(sleep_for)

        return [r or "" for r in results]

    minimal_outputs: List[str] = asyncio.run(_run_rate_limited())

    if verbose:
        print("[4/5] Assembling global SRT...", flush=True)
    srt_text = assemble_srt_from_minimal_segments(
        minimal_outputs,
        offsets_ms,
        segment_durations_ms=seg_durations_ms,
        total_duration_ms=total_ms,
    )

    base = os.path.splitext(os.path.basename(input_audio))[0]
    out_path = os.path.join(output_dir, base + ".srt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    if verbose:
        print(f"  - Wrote {out_path}", flush=True)

    if cleanup:
        try:
            import shutil
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    if verbose:
        print(f"Total: {time.time() - t0:.1f}s", flush=True)
    return out_path


