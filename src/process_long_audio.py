import os
import time
import asyncio
from typing import List, Tuple

from .audio_utils import split_audio_into_equal_segments
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
        tasks = [
            _call_model_for_segment(
                seg,
                source_language,
                target_language,
                model,
                mime_type,
                verbose,
            )
            for seg in segment_paths
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    minimal_outputs: List[str] = asyncio.run(_run_all())

    # Ensure we have exactly num_segments outputs and offsets
    boundaries = [int((i * total_ms) / num_segments) for i in range(num_segments)]
    if len(minimal_outputs) < num_segments:
        minimal_outputs.extend([""] * (num_segments - len(minimal_outputs)))
    elif len(minimal_outputs) > num_segments:
        minimal_outputs = minimal_outputs[:num_segments]

    if verbose:
        print("[3/4] Assembling global SRT...", flush=True)
    srt_text = assemble_srt_from_minimal_segments(minimal_outputs, boundaries)

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


