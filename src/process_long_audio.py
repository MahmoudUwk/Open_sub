"""Main audio processing pipeline."""

import os
import time
import datetime
import shutil
from typing import List, Optional

import google.genai as genai

from .audio_utils import split_audio_by_duration
from .upload_transcribe_translate_audio import transcribe_minimal
from .minimal_format import assemble_srt_from_minimal_segments


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
    """Translate each transcribed segment."""
    translated_outputs = []
    client = genai.Client()
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
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            trans_text = response.text.strip()
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
        except Exception as e:
            print(f"Error translating segment {i} with {model}: {e}")
            translated_outputs.append(out)
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
    if transcription_models is None:
        transcription_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    if translation_models is None:
        translation_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "translated"), exist_ok=True)

    t0 = time.time()
    if verbose:
        print(f"[1/5] Splitting by duration...", flush=True)
    seg_paths, offsets_ms, seg_durations_ms, total_ms = split_audio_by_duration(
        input_audio, min_segment_minutes, tmp_dir, verbose
    )
    if not seg_paths:
        raise RuntimeError("No segments produced.")

    if verbose:
        print("[2/5] Transcribing segments...", flush=True)
    minimal_outputs = _transcribe_segments(
        seg_paths, source_language, transcription_models, output_dir, verbose
    )

    if verbose:
        print("[3/5] Translating segments...", flush=True)
    translated_outputs = _translate_segments(
        minimal_outputs, source_language, target_language, translation_models, output_dir, verbose
    )

    if verbose:
        print("[4/5] Assembling global SRT...", flush=True)
    srt_text = assemble_srt_from_minimal_segments(
        translated_outputs, offsets_ms, seg_durations_ms, total_ms
    )

    base = os.path.splitext(os.path.basename(input_audio))[0]
    out_path = os.path.join(output_dir, base + ".srt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    if verbose:
        print(f"  - Wrote {out_path}", flush=True)

    if cleanup:
        if verbose:
            print("[5/5] Cleaning up temporary files...", flush=True)
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"Error cleaning up tmp_dir: {e}")

    if verbose:
        print(f"Total: {time.time() - t0:.1f}s", flush=True)
    return out_path