"""Main audio processing pipeline."""

import os
import json
import time
import multiprocessing as mp
import datetime
import shutil
from typing import List, Optional

import google.genai as genai

from .audio_utils import split_audio_by_duration
from .upload_transcribe_translate_audio import transcribe_minimal, RATE_LIMIT_ERRORS
from .minimal_format import assemble_srt_from_minimal_segments, clean_minimal_text, parse_minimal_lines, format_ms_xmys
from .audio_utils import get_audio_duration_ms

TIMEOUT_TRANSCRIBE_S = 120
TIMEOUT_TRANSLATE_S = 90

def _call_with_timeout(func, timeout_s: int, *args, **kwargs):
    """Run func with a hard timeout using a child process.
    Returns (result, err_str or None). On timeout, returns (None, 'timeout').
    """
    q = mp.Queue(maxsize=1)

    def _runner(q):
        try:
            res = func(*args, **kwargs)
            q.put((res, None))
        except Exception as e:
            q.put((None, str(e)))

    p = mp.Process(target=_runner, args=(q,), daemon=True)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join(1)
        return None, "timeout"
    try:
        return q.get_nowait()
    except Exception:
        return None, "unknown_error"

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
    transcribe_timeout_s: int,
    transcribe_retry_wait_s: int,
) -> List[str]:
    """Transcribe each audio segment sequentially."""
    minimal_outputs = []
    for i, seg_path in enumerate(seg_paths):
        model = transcription_models[i % len(transcription_models)]
        try:
            while True:
                start_time = time.time()
                outcome, err = _call_with_timeout(
                    _call_model_for_segment,
                    transcribe_timeout_s,
                    seg_path, source_language, model, _guess_mime_type_from_extension(seg_path), verbose
                )
                duration = time.time() - start_time
                if err == "timeout":
                    print(f"[TR] seg {i} timeout {transcribe_timeout_s}s", flush=True)
                    print(f"[TR] wait {transcribe_retry_wait_s}s before retry...", flush=True)
                    time.sleep(transcribe_retry_wait_s)
                    continue  # Retry same segment indefinitely on timeout
                if err:
                    # Print concise error with duration, then continue to next segment
                    print(f"[TR] seg {i} error {duration:.1f}s: {err}")
                    minimal_outputs.append("")
                    break
                if not outcome.strip():
                    # Empty but not an exception: wait and retry indefinitely
                    print(f"[TR] seg {i} empty {duration:.1f}s", flush=True)
                    print(f"[TR] wait {transcribe_retry_wait_s}s before retry...", flush=True)
                    time.sleep(transcribe_retry_wait_s)
                    continue
                cleaned = clean_minimal_text(str(outcome))
                minimal_outputs.append(cleaned)
                # Always show a single concise success line for non-empty output
                print(f"[TR] seg {i} ok {duration:.1f}s")
                raw_path = os.path.join(output_dir, "raw", f"segment_{i}_raw.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(outcome)
                break
        except Exception as e:
            # Unexpected error in wrapper
            print(f"[TR] seg {i} error: {str(e)}")
            minimal_outputs.append("")
    return minimal_outputs

def _translate_segments(
    minimal_outputs: List[str],
    source_language: str,
    target_language: str,
    translation_models: List[str],
    output_dir: str,
    verbose: bool,
    translate_timeout_s: int,
    translate_max_retries: int,
    translate_retry_wait_s: int,
) -> List[str]:
    """Translate each transcribed segment with configurable fallbacks."""
    translated_outputs = []

    # Helper: extract simple bullet lines ("- some text"). If none, fallback to non-empty lines
    def _extract_bullets(text: str) -> List[str]:
        lines: List[str] = []
        if not text:
            return lines
        for raw in text.splitlines():
            l = raw.strip()
            if not l:
                continue
            if l.startswith("- "):
                lines.append(l[2:].strip())
        if not lines:
            for raw in text.splitlines():
                l = raw.strip()
                if l:
                    lines.append(l)
        return lines

    def _call_once(m: str, text: str) -> tuple[str, str | None]:
        try:
            client = genai.Client()
            response = client.models.generate_content(model=m, contents=text)
            t = (getattr(response, "text", None) or "").strip()
            return t, None
        except Exception as e:
            return "", str(e)

    def _call_with_retries(m: str, text: str, max_retries: int, validator=None):
        attempts = 0
        timeouts = 0
        errors = 0
        last_err = None
        non_timeout_attempts = 0
        while True:
            attempt = attempts + 1
            attempts += 1
            a_t0 = time.time()
            res, to_err = _call_with_timeout(_call_once, translate_timeout_s, m, text)
            a_dur = time.time() - a_t0
            t = ""
            err = to_err
            if to_err == "timeout":
                timeouts += 1
                print(f"[TL] {m} attempt {attempt}/{max_retries}: timeout {a_dur:.1f}s", flush=True)
                # Always retry on timeout (do not count against max_retries)
                print(f"[TL] wait {translate_retry_wait_s}s before retry...", flush=True)
                time.sleep(translate_retry_wait_s)
                continue
            elif to_err is None:
                # _call_once returns (text, err)
                if isinstance(res, tuple) and len(res) == 2:
                    t, call_err = res
                    err = call_err
                else:
                    # Defensive: if library changes
                    t = (res or "") if isinstance(res, str) else ""
                # Apply optional post-condition validator
                if t and callable(validator):
                    ok, reason = validator(t)
                    if not ok:
                        # Treat as failure and retry (counts against non-timeout attempts)
                        t = ""
                        err = reason or "validation_failed"
                if not t:
                    # Consider empty as an error-like outcome for statistics
                    errors += 1
                    non_timeout_attempts += 1
                    last_err = (err or "empty")
                    if verbose:
                        msg = err or "empty"
                        print(f"[TL] {m} attempt {attempt}/{max_retries}: {msg} {a_dur:.1f}s")
            else:
                # Exception path
                errors += 1
                non_timeout_attempts += 1
                last_err = err
                if verbose:
                    print(f"[TL] {m} attempt {attempt}/{max_retries}: {err} {a_dur:.1f}s")

            if t:
                return t, {"attempts": attempts, "timeouts": timeouts, "errors": errors, "last_error": last_err}

            # Stop if exceeded non-timeout retries; otherwise wait and retry
            if non_timeout_attempts >= max_retries:
                return "", {"attempts": attempts, "timeouts": timeouts, "errors": errors, "last_error": last_err}
            if verbose:
                print(f"[TL] wait {translate_retry_wait_s}s before retry...")
            time.sleep(translate_retry_wait_s)

    for i, out in enumerate(minimal_outputs):
        if not out.strip():
            translated_outputs.append("")
            print(f"[TL] seg {i} skip-empty")
            continue

        model = translation_models[i % len(translation_models)]
        # Prepare bullet-only input (no timestamps). Preserve line order and count
        items = parse_minimal_lines(clean_minimal_text(out))
        orig_texts = [it["text"] for it in items]
        time_pairs = [(int(it["start_ms"]), int(it["end_ms"])) for it in items]
        bullet_input = "\n".join(f"- {t}" for t in orig_texts)
        prompt = (
            f"Translate from {source_language} to {target_language}.\n"
            f"You will receive N lines starting with '- ' (hyphens).\n"
            f"Output ONLY N lines, in the SAME order, each starting with '- ' followed by the translated text.\n"
            f"Do NOT add timestamps, numbering, brackets, or extra commentary.\n"
            f"Input lines:\n{bullet_input}\n\n"
            f"Output lines:\n"
        )

        # Call primary model with fixed-interval retries (no fallbacks)
        t0 = time.time()
        trans_text, stats = _call_with_retries(model, prompt, max_retries=translate_max_retries, validator=None)
        dur = time.time() - t0
        if not trans_text.strip():
            last_err = stats.get("last_error") or ""
            err_low = last_err.lower() if isinstance(last_err, str) else ""
            rate_limited = any(k.lower() in err_low for k in RATE_LIMIT_ERRORS) if last_err else False
            tag = "rate_limit" if rate_limited else "empty"
            # Show concise summary with counts and last error snippet
            snippet = (last_err[:120] + "...") if isinstance(last_err, str) and len(last_err) > 120 else (last_err or "")
            print(
                f"[TL] seg {i} {tag} {dur:.1f}s attempts={stats.get('attempts', 0)} "
                f"timeouts={stats.get('timeouts', 0)} errors={stats.get('errors', 0)} last_err={snippet!r}"
            )
        else:
            print(f"[TL] seg {i} ok {dur:.1f}s attempts={stats.get('attempts', 0)}")
        # Reinsert original timestamps to produce minimal lines again
        translated_lines = _extract_bullets(trans_text)
        if len(translated_lines) != len(orig_texts) and verbose:
            print(
                f"[TL] seg {i} line_count_mismatch exp={len(orig_texts)} got={len(translated_lines)}"
            )
        # Align counts: truncate or pad with original text
        if len(translated_lines) < len(orig_texts):
            translated_lines += orig_texts[len(translated_lines):]
        else:
            translated_lines = translated_lines[:len(orig_texts)]

        minimal_lines = [
            f"[{format_ms_xmys(s)}-{format_ms_xmys(e)}]: {t.strip()}"
            for (s, e), t in zip(time_pairs, translated_lines)
            if str(t).strip()
        ]
        cleaned_t = clean_minimal_text("\n".join(minimal_lines))
        translated_outputs.append(cleaned_t)
        translated_path = os.path.join(output_dir, "translated", f"segment_{i}_translated.txt")
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(cleaned_t)
        # keep file-system noise out of console

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
    start_step: str = "split",
    # Configurable timeouts/retry settings (defaults preserve current behavior)
    transcribe_timeout_s: Optional[int] = None,
    translate_timeout_s: Optional[int] = None,
    translate_max_retries: Optional[int] = None,
    transcribe_retry_wait_s: Optional[int] = None,
    translate_retry_wait_s: Optional[int] = None,
) -> str:
    """Split audio by duration, transcribe, translate, and assemble SRT."""
    if not transcription_models:
        transcription_models = ["gemini-2.5-pro"]
    if not translation_models:
        translation_models = ["gemini-2.5-pro"]

    # Resolve timeout/retry defaults
    transcribe_timeout_s = int(transcribe_timeout_s or TIMEOUT_TRANSCRIBE_S)
    translate_timeout_s = int(translate_timeout_s or TIMEOUT_TRANSLATE_S)
    translate_max_retries = int(translate_max_retries or 3)
    transcribe_retry_wait_s = int(transcribe_retry_wait_s or 20)
    translate_retry_wait_s = int(translate_retry_wait_s or 20)

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
    split_t0 = t0
    print("[SPLIT] start", flush=True)
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
                print(f"[SPLIT] reuse {len(seg_paths)} segs")
        except Exception as e:
            if verbose:
                print(f"[SPLIT] cannot reuse: {e}")
    if seg_paths:
        # Already populated from offsets.json; ensure tmp_dir exists for downstream if needed
        os.makedirs(tmp_dir, exist_ok=True)
    elif total_ms < min_segment_minutes * 60 * 1000:
        print("[SPLIT] skip-short 1 seg")
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
            print("[SPLIT] splitting...", flush=True)
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
    print(f"[SPLIT] ready {len(seg_paths)} segs {(time.time()-split_t0):.1f}s")

    minimal_outputs: List[str]
    if start_step.lower() in ("translate", "assemble"):
        # Resume from saved raw transcripts
        print("[TRANSCRIBE] resume", flush=True)
        minimal_outputs = []
        for i in range(len(seg_paths)):
            raw_path = os.path.join(work_output_dir, "raw", f"segment_{i}_raw.txt")
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    minimal_outputs.append(clean_minimal_text(f.read()))
            except Exception:
                minimal_outputs.append("")
    else:
        print("[TRANSCRIBE] start", flush=True)
        tr_t0 = time.time()
        minimal_outputs = _transcribe_segments(
            seg_paths, source_language, transcription_models, work_output_dir, verbose,
            transcribe_timeout_s, transcribe_retry_wait_s,
        )
        print(f"[TRANSCRIBE] done {(time.time()-tr_t0):.1f}s")

    # Guard: ensure we have any non-empty transcription before translating
    if not any(s.strip() for s in minimal_outputs):
        raise RuntimeError("Transcription failed: all segments empty")

    if start_step.lower() in ("assemble",):
        print("[TRANSLATE] resume", flush=True)
        translated_outputs: List[str] = []
        for i in range(len(minimal_outputs)):
            t_path = os.path.join(work_output_dir, "translated", f"segment_{i}_translated.txt")
            try:
                with open(t_path, "r", encoding="utf-8") as f:
                    translated_outputs.append(clean_minimal_text(f.read()))
            except Exception:
                translated_outputs.append("")
    else:
        print("[TRANSLATE] start", flush=True)
        tl_t0 = time.time()
        translated_outputs = _translate_segments(
            minimal_outputs, source_language, target_language, translation_models, work_output_dir, verbose,
            translate_timeout_s, translate_max_retries, translate_retry_wait_s,
        )
        print(f"[TRANSLATE] done {(time.time()-tl_t0):.1f}s")

    if not any(t.strip() for t in translated_outputs):
        raise RuntimeError("Translation failed: all segments empty")

    out_path = os.path.join(work_output_dir, base + ".srt")
    if start_step.lower() == "assemble" and os.path.exists(out_path):
        print("[ASSEMBLE] skip (existing SRT)", flush=True)
    else:
        print("[ASSEMBLE] start", flush=True)
        srt_text = assemble_srt_from_minimal_segments(
            translated_outputs, offsets_ms
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        print(f"[ASSEMBLE] wrote {out_path}")

    # No SRT post-validation: we validate/fix at source (transcription/translation minimal outputs)

    if cleanup:
        try:
            shutil.rmtree(tmp_dir)
            print("[CLEANUP] done")
        except Exception as e:
            print(f"[CLEANUP] error: {e}")

    print(f"[DONE] {time.time() - t0:.1f}s")
    return out_path