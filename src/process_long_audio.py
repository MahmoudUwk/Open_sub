import os
import time
import asyncio
from typing import List, Tuple, Optional
import datetime
import re

from .audio_utils import (
    split_audio_by_duration,
)
from .upload_transcribe_translate_audio import transcribe_minimal
from .minimal_format import assemble_srt_from_minimal_segments

import google.genai as genai


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


def _call_model_for_segment(
    segment_path: str,
    source_language: str,
    model: str,
    mime_type: str,
    verbose: bool,
) -> str:
    # Read bytes in thread
    def _read_bytes(p: str) -> bytes:
        with open(p, "rb") as f:
            return f.read()

    audio_bytes = _read_bytes(segment_path)
    # Run model in thread (SDK is sync)
    return transcribe_minimal(
        audio_bytes,
        mime_type,
        source_language,
        model,
        3,
        2.0,
        verbose,
    )


def translate_segments(
    minimal_outputs: List[str],
    source_language: str,
    target_language: str,
    models: List[str],
    verbose: bool = False
) -> List[str]:
    client = genai.Client()
    translated = []
    for i, out in enumerate(minimal_outputs):
        if not out.strip():
            translated.append("")
            continue

        model = models[i % len(models)]
        prompt = (
            f"You are a professional subtitle translator.\n"
            f"Translate the following text from {source_language} to {target_language}.\n"
            f"The input is a list of timed subtitles. You MUST preserve the timestamps exactly.\n"
            f"Input:\n{out}\n\n"
            f"Output:\n"
        )
        
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            trans_text = response.text.strip()
            translated.append(trans_text)
            if verbose:
                print(f"Translated segment {i} with {model}")
        except Exception as e:
            print(f"Error translating segment {i} with {model}: {e}")
            translated.append(out)  # Fallback to original transcription

    return translated


def process_audio_fixed_duration(
    input_audio: str,
    source_language: str,
    target_language: str,
    model: str = "gemini-2.5-flash",  # This might be unused now, but keep as default
    min_segment_minutes: int = 5,
    segment_overlap_seconds: int = 2,
    tmp_dir: str = "tmp_segments",
    output_dir: str = "output_srt",
    cleanup: bool = True,
    verbose: bool = True,
    transcription_models: List[str] = ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    translation_models: List[str] = ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
) -> str:
    """Split by duration (>=5 minutes), call model under rate limit, assemble SRT.

    Ensures at most `rate_limit_per_minute` requests are started per minute, in batches.
    Saves successful outputs and retries rate-limited failures in subsequent batches.
    """
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "translated"), exist_ok=True)

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

    def _run_sequential_transcription() -> List[str]:
        results = ["" for _ in seg_paths]  # Default to empty
        for idx in range(len(seg_paths)):
            model = transcription_models[idx % len(transcription_models)]
            print(f"Using transcription model: {model} for segment {idx}")
            print(f"[{datetime.datetime.now()}] Starting call for segment {idx}")
            try:
                start_time = time.time()
                outcome = _call_model_for_segment(
                    seg_paths[idx],
                    source_language,
                    model,
                    mime_type,
                    verbose,
                )
                duration = time.time() - start_time
                results[idx] = str(outcome)
                print(f"[{datetime.datetime.now()}] Finished call for segment {idx}")
                print(f"Segment {idx} API call took {duration:.2f}s (chars={len(results[idx])})")
                raw_path = os.path.join(output_dir, "raw", f"segment_{idx}_raw.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(outcome)
                print(f"Saved raw output to {raw_path}")
            except Exception as e:
                print(f"Error for segment {idx}: {str(e)}")
                print(f"[{datetime.datetime.now()}] Finished call for segment {idx} (error)")
        return results

    minimal_outputs: List[str] = _run_sequential_transcription()

    if verbose:
        print("[4/5] Translating segments...", flush=True)
    translated_outputs = translate_segments(minimal_outputs, source_language, target_language, translation_models, verbose=True)
    for i, translated_text in enumerate(translated_outputs):
        translated_path = os.path.join(output_dir, "translated", f"segment_{i}_translated.txt")
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        if verbose:
            print(f"Saved translated output to {translated_path}")

    if verbose:
        print("[5/5] Assembling global SRT...", flush=True)
    srt_text = assemble_srt_from_minimal_segments(translated_outputs, offsets_ms, seg_durations_ms, total_ms)

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


