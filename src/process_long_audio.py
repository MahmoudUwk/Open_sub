"""Main audio processing pipeline."""

import os
import json
import time
import datetime
import shutil
from typing import List, Optional

import google.genai as genai

from .audio_utils import split_audio_by_duration
from .upload_transcribe_translate_audio import transcribe_minimal
from .minimal_format import assemble_srt_from_minimal_segments
from .srt_validate import validate_and_optionally_fix
from .audio_utils import get_audio_duration_ms


def _guess_mime_type_from_extension(path: str) -> str:
    """Guess the MIME type of an audio file from its extension."""
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
    """Read an audio segment and call the transcription model."""
    with open(segment_path, "rb") as f:
        audio_bytes = f.read()
    return transcribe_minimal(audio_bytes, mime_type, source_language, model, verbose=verbose)

def _transcribe_segments(
    seg_paths: List[str],
    source_language: str,
    transcription_models: List[str],
    output_dir: str,
    verbose: bool,
) -> List[str]:
    """Transcribe each audio segment sequentially."""
    minimal_outputs = []
    for i, seg_path in enumerate(seg_paths):
        model = transcription_models[i % len(transcription_models)]
        if verbose:
            print(f"Using transcription model: {model} for segment {i}")
            print(f"[{datetime.datetime.now()}] Starting call for segment {i}")
        try:
            start_time = time.time()
            outcome = _call_model_for_segment(
                seg_path, source_language, model, _guess_mime_type_from_extension(seg_path), verbose
            )
            duration = time.time() - start_time
            if not outcome.strip():
                print(f"Warning: Transcription empty for segment {i} (model: {model})")
            minimal_outputs.append(str(outcome))
            if verbose:
                print(f"[{datetime.datetime.now()}] Finished call for segment {i}")
                print(f"Segment {i} API call took {duration:.2f}s (chars={len(outcome)})")
            raw_path = os.path.join(output_dir, "raw", f"segment_{i}_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(outcome)
            if verbose:
                print(f"Saved raw output to {raw_path}")
        except Exception as e:
            print(f"Error transcribing segment {i} (model: {model}): {str(e)}")
            minimal_outputs.append("")
            if verbose:
                print(f"[{datetime.datetime.now()}] Finished call for segment {i} (error)")
    return minimal_outputs

def _translate_segments(
    minimal_outputs: List[str],
    source_language: str,
    target_language: str,
    translation_models: List[str],
    output_dir: str,
    verbose: bool,
) -> List[str]:
    """Translate each transcribed segment with configurable fallbacks."""
    translated_outputs = []
    client = genai.Client()

    def _call_once(m: str, text: str) -> tuple[str, str | None]:
        try:
            response = client.models.generate_content(model=m, contents=text)
            t = (getattr(response, "text", None) or "").strip()
            return t, None
        except Exception as e:
            return "", str(e)

    def _call_with_retries(m: str, text: str, max_retries: int = 4) -> str:
        for attempt in range(1, max_retries + 1):
            t, err = _call_once(m, text)
            if t:
                return t
            if verbose:
                if err:
                    print(f"        [API {m}] Translation error on attempt {attempt}/{max_retries}: '{err}'")
                else:
                    print(f"        [API {m}] Translation empty on attempt {attempt}/{max_retries}")
            if attempt < max_retries:
                if verbose:
                    print("        Waiting 60s before next translation retry...")
                time.sleep(60)
        return ""

    for i, out in enumerate(minimal_outputs):
        if not out.strip():
            translated_outputs.append("")
            print(f"Skipping translation for empty segment {i}")
            continue

        model = translation_models[i % len(translation_models)]
        prompt = (
            f"You are a professional subtitle translator.\n"
            f"Translate the following text from {source_language} to {target_language}.\n"
            f"The input is a list of timed subtitles. You MUST preserve the timestamps exactly.\n"
            f"Input:\n{out}\n\n"
            f"Output:\n"
        )

        # Call primary model with fixed-interval retries (no fallbacks)
        trans_text = _call_with_retries(model, prompt, max_retries=4)

        if not trans_text.strip():
            print(f"Warning: Translation empty for segment {i} (model: {model})")
        translated_outputs.append(trans_text)
        if verbose:
            print(f"Translated segment {i} with {model}")
        translated_path = os.path.join(output_dir, "translated", f"segment_{i}_translated.txt")
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(trans_text)
        if verbose:
            print(f"Saved translated output to {translated_path}")

    return translated_outputs

def process_audio_fixed_duration(
    input_audio: str,
    source_language: str,
    target_language: str,
    min_segment_minutes: int = 15,
    tmp_dir: str = "tmp_segments",
    output_dir: str = "output_srt",
    cleanup: bool = True,
    verbose: bool = True,
    transcription_models: Optional[List[str]] = None,
    translation_models: Optional[List[str]] = None,
) -> str:
    """Split audio by duration, transcribe, translate, and assemble SRT."""
    if not transcription_models:
        transcription_models = ["gemini-2.5-pro"]
    if not translation_models:
        translation_models = ["gemini-2.5-pro"]

    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    # Group all results for this input under output_dir/<base>/
    base = os.path.splitext(os.path.basename(input_audio))[0]
    work_output_dir = os.path.join(output_dir, base)
    os.makedirs(work_output_dir, exist_ok=True)
    os.makedirs(os.path.join(work_output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(work_output_dir, "translated"), exist_ok=True)
    offsets_json_path = os.path.join(work_output_dir, "offsets.json")

    t0 = time.time()
    total_ms = get_audio_duration_ms(input_audio)

    # If offsets.json exists and referenced segment files exist, reuse them
    seg_paths: List[str] = []
    offsets_ms: List[int] = []
    seg_durations_ms: List[int] = []
    if os.path.exists(offsets_json_path):
        try:
            with open(offsets_json_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved_paths = saved.get("segment_paths") or []
            saved_offsets = saved.get("offsets_ms") or []
            saved_durations = saved.get("durations_ms") or []
            if (
                isinstance(saved_paths, list)
                and isinstance(saved_offsets, list)
                and len(saved_paths) == len(saved_offsets)
                and (not saved_durations or len(saved_durations) == len(saved_paths))
                and all(isinstance(p, str) and os.path.exists(p) for p in saved_paths)
            ):
                seg_paths = list(saved_paths)
                offsets_ms = list(saved_offsets)
                seg_durations_ms = list(saved_durations) if saved_durations else []
                if verbose:
                    print("[1] Reusing existing segments and offsets from offsets.json")
        except Exception as e:
            if verbose:
                print(f"Warning: failed to load offsets.json, will re-split. Error: {e}")
    if seg_paths:
        # Already populated from offsets.json; ensure tmp_dir exists for downstream if needed
        os.makedirs(tmp_dir, exist_ok=True)
    elif total_ms < min_segment_minutes * 60 * 1000:
        if verbose:
            print("[1] Audio is short; skipping split...")
        seg_paths = [input_audio]
        offsets_ms = [0]
        seg_durations_ms = [total_ms]
        # Persist offsets for later reuse
        try:
            with open(offsets_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "base": base,
                    "total_ms": total_ms,
                    "segment_paths": seg_paths,
                    "offsets_ms": offsets_ms,
                    "durations_ms": seg_durations_ms,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if verbose:
                print(f"Warning: failed to write offsets file: {e}")
    else:
        if verbose:
            print("[1] Splitting by duration...", flush=True)
        seg_paths, offsets_ms, seg_durations_ms, total_ms = split_audio_by_duration(
            input_audio, min_segment_minutes, tmp_dir, verbose
        )
        # Persist offsets for later reuse
        try:
            with open(offsets_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "base": base,
                    "total_ms": total_ms,
                    "segment_paths": seg_paths,
                    "offsets_ms": offsets_ms,
                    "durations_ms": seg_durations_ms,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if verbose:
                print(f"Warning: failed to write offsets file: {e}")
    # Guard: ensure segments were created and valid
    if not seg_paths or any((not os.path.exists(p) or os.path.getsize(p) < 1024) for p in seg_paths):
        raise RuntimeError("Segmenting failed: missing or tiny segment files")

    if verbose:
        print("[2] Transcribe", flush=True)
    minimal_outputs = _transcribe_segments(
        seg_paths, source_language, transcription_models, work_output_dir, verbose
    )

    # Guard: ensure we have any non-empty transcription before translating
    if not any(s.strip() for s in minimal_outputs):
        raise RuntimeError("Transcription failed: all segments empty")

    if verbose:
        print("[3] Translate", flush=True)
    translated_outputs = _translate_segments(
        minimal_outputs, source_language, target_language, translation_models, work_output_dir, verbose
    )

    if not any(t.strip() for t in translated_outputs):
        raise RuntimeError("Translation failed: all segments empty")

    if verbose:
        print("[4] Assemble", flush=True)
    srt_text = assemble_srt_from_minimal_segments(
        translated_outputs, offsets_ms
    )

    out_path = os.path.join(work_output_dir, base + ".srt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    if verbose:
        print(f"Wrote {out_path}", flush=True)

    # Auto-validate and overwrite with fixed version
    try:
        tmp_fixed = out_path + ".fixed.tmp"
        _issues = validate_and_optionally_fix(out_path, out_fixed_path=tmp_fixed)
        if os.path.exists(tmp_fixed):
            try:
                os.replace(tmp_fixed, out_path)
                if verbose:
                    bad_counts = {k: len(v) for k, v in _issues.items()}
                    print(f"Validated and fixed SRT in place. Issues summary: {bad_counts}")
            except Exception as e:
                if verbose:
                    print(f"Warning: failed to replace SRT with fixed version: {e}")
    except Exception as e:
        if verbose:
            print(f"Warning: SRT validation step failed: {e}")

    if cleanup and verbose:
        print("[5] Cleanup", flush=True)
    if cleanup:
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"Error cleaning up tmp_dir: {e}")

    if verbose:
        print(f"Done in {time.time() - t0:.1f}s", flush=True)
    return out_path