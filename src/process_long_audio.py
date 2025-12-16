"""Main audio processing pipeline."""

import os
import json
import time
import concurrent.futures
import shutil
from typing import List, Optional

import google.genai as genai

from .audio_utils import split_audio_by_duration
from .upload_transcribe_translate_audio import transcribe_minimal, RATE_LIMIT_ERRORS
from .minimal_format import (
    assemble_srt_from_segments,
    clean_segment_text,
    parse_any_segments,
    format_ms_xmys,
    parse_compact_segments,
)
from .audio_utils import get_audio_duration_ms

TIMEOUT_TRANSCRIBE_S = 120
TIMEOUT_TRANSLATE_S = 90


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
    return transcribe_minimal(
        audio_bytes, mime_type, source_language, model, verbose=verbose
    )


def _transcribe_segments(
    seg_paths: List[str],
    source_language: str,
    transcription_models: List[str],
    output_dir: str,
    verbose: bool,
    transcribe_timeout_s: int,
    transcribe_retry_wait_s: int,
    transcribe_max_attempts: int,
    max_workers: int,
) -> List[str]:
    """Transcribe audio segments with bounded retries and concurrency."""
    minimal_outputs = [""] * len(seg_paths)

    def _transcribe_one(idx: int, seg_path: str, model: str) -> None:
        attempts = 0
        try:
            while True:
                if attempts >= transcribe_max_attempts:
                    raise RuntimeError("max attempts exceeded")
                attempts += 1
                start_time = time.time()
                # Use inner executor to enforce timeout per attempt
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as inner:
                    fut = inner.submit(
                        _call_model_for_segment,
                        seg_path,
                        source_language,
                        model,
                        _guess_mime_type_from_extension(seg_path),
                        verbose,
                    )
                    try:
                        outcome = fut.result(timeout=transcribe_timeout_s)
                        err = None
                    except concurrent.futures.TimeoutError:
                        fut.cancel()
                        outcome = None
                        err = "timeout"
                    except Exception as e:
                        fut.cancel()
                        outcome = None
                        err = str(e)

                duration = time.time() - start_time
                if err == "timeout":
                    print(f"[TR] seg {idx} timeout {transcribe_timeout_s}s", flush=True)
                    time.sleep(transcribe_retry_wait_s)
                    continue
                if err:
                    print(f"[TR] seg {idx} error {duration:.1f}s: {err}", flush=True)
                    time.sleep(transcribe_retry_wait_s)
                    continue
                if not outcome or not outcome.strip():
                    print(f"[TR] seg {idx} empty {duration:.1f}s", flush=True)
                    time.sleep(transcribe_retry_wait_s)
                    continue

                try:
                    cleaned = clean_segment_text(str(outcome))
                    parsed = parse_any_segments(cleaned)
                    if not parsed:
                        raise ValueError("parsed entries empty")
                except Exception as e:
                    print(
                        f"[TR] seg {idx} invalid json {duration:.1f}s: {e}", flush=True
                    )
                    time.sleep(transcribe_retry_wait_s)
                    continue

                minimal_outputs[idx] = cleaned
                raw_path = os.path.join(output_dir, "raw", f"segment_{idx}_raw.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(outcome)
                print(f"[TR] seg {idx} ok {duration:.1f}s")
                return
        except Exception as e:
            print(f"[TR] seg {idx} error: {str(e)}; using placeholder transcript")
            dur_ms = get_audio_duration_ms(seg_path) or (60 * 1000)
            placeholder = [
                {
                    "start": format_ms_xmys(0),
                    "end": format_ms_xmys(dur_ms),
                    "text": "[untranscribed]",
                }
            ]
            cleaned = clean_segment_text(
                json.dumps(placeholder)
            )  # Re-use cleaner to format placeholder as compact
            minimal_outputs[idx] = cleaned
            raw_path = os.path.join(output_dir, "raw", f"segment_{idx}_raw.txt")
            try:
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
            except Exception:
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, seg_path in enumerate(seg_paths):
            model = transcription_models[i % len(transcription_models)]
            futures.append(pool.submit(_transcribe_one, i, seg_path, model))
        concurrent.futures.wait(futures)

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
    translate_max_empty_attempts: int,
    segment_durations_ms: Optional[List[int]],
    max_workers: int,
) -> List[str]:
    """Translate each transcribed segment with configurable fallbacks."""
    translated_outputs = [""] * len(minimal_outputs)

    def _strip_code_fences(text: str) -> str:
        """Remove Markdown fences that break JSON parsing."""
        t = (text or "").strip()
        if not t.startswith("```"):
            return t
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _call_with_retries(
        m: str,
        text: str,
        max_retries: int,
        validator=None,
        postprocess=None,
        max_empty_attempts: int = 6,
        translate_timeout_s: int = 90,
    ):
        attempts = 0
        timeouts = 0
        errors = 0
        last_err = None
        non_timeout_attempts = 0
        empty_attempts = 0
        while True:
            attempt = attempts + 1
            attempts += 1
            a_t0 = time.time()

            def _call_once() -> tuple[str, str | None]:
                try:
                    client = genai.Client()
                    response = client.models.generate_content(
                        model=m,
                        contents=text,
                        config=genai.types.GenerateContentConfig(
                            safety_settings=[
                                genai.types.SafetySetting(
                                    category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                    threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
                                )
                            ]
                        ),
                    )
                    t = (getattr(response, "text", None) or "").strip()
                    return t, None
                except Exception as exc:
                    return "", str(exc)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as inner:
                fut = inner.submit(_call_once)
                try:
                    res = fut.result(timeout=translate_timeout_s)
                    to_err = None
                except concurrent.futures.TimeoutError:
                    res = None
                    to_err = "timeout"
                except Exception as e:
                    res = None
                    to_err = str(e)

            a_dur = time.time() - a_t0
            t = ""
            err = to_err
            if to_err == "timeout":
                timeouts += 1
                last_err = "timeout"
                print(
                    f"[TL] {m} attempt {attempt}/{max_retries}: timeout {a_dur:.1f}s",
                    flush=True,
                )
                if timeouts >= max_retries:
                    return "", {
                        "attempts": attempts,
                        "timeouts": timeouts,
                        "errors": errors,
                        "last_error": last_err,
                    }
                print(
                    f"[TL] wait {translate_retry_wait_s}s before retry...", flush=True
                )
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
                if postprocess:
                    t = postprocess(t)
                # Apply optional post-condition validator
                if t and callable(validator):
                    ok, reason = validator(t)
                    if not ok:
                        # Treat as failure and retry (counts against non-timeout attempts)
                        t = ""
                        err = reason or "validation_failed"
                if not t:
                    # Empty response: retry indefinitely (do not count against max_retries)
                    errors += 1
                    last_err = err or "empty"
                    empty_attempts += 1
                    if verbose:
                        msg = err or "empty"
                        print(
                            f"[TL] {m} attempt {attempt}: {msg} {a_dur:.1f}s (retrying indefinitely on empty)"
                        )
                    if empty_attempts >= max_empty_attempts:
                        return "", {
                            "attempts": attempts,
                            "timeouts": timeouts,
                            "errors": errors,
                            "last_error": last_err or "empty_limit",
                        }
                    print(
                        f"[TL] wait {translate_retry_wait_s}s before retry...",
                        flush=True,
                    )
                    time.sleep(translate_retry_wait_s)
                    continue
            else:
                # Exception path
                errors += 1
                non_timeout_attempts += 1
                last_err = err
                if verbose:
                    print(
                        f"[TL] {m} attempt {attempt}/{max_retries}: {err} {a_dur:.1f}s"
                    )

            if t:
                return t, {
                    "attempts": attempts,
                    "timeouts": timeouts,
                    "errors": errors,
                    "last_error": last_err,
                }

            # Stop only if exceeded non-timeout retries (actual errors, not empty responses)
            if non_timeout_attempts >= max_retries:
                return "", {
                    "attempts": attempts,
                    "timeouts": timeouts,
                    "errors": errors,
                    "last_error": last_err,
                }

    def _translate_one(i: int, out: str) -> None:
        if not out.strip():
            translated_outputs[i] = ""
            print(f"[TL] seg {i} skip-empty")
            return

        model = translation_models[i % len(translation_models)]

        # Parse compact segments to extract text for translation
        # Input format: [Start - End] Text
        items = parse_any_segments(clean_segment_text(out))
        if not items:
            translated_outputs[i] = ""
            return

        # Prepare Input as [Start - End] Text
        # We send the timestamps too so the model mimics the structure
        input_lines = []
        for it in items:
            s_fmt = format_ms_xmys(it["start_ms"])
            e_fmt = format_ms_xmys(it["end_ms"])
            input_lines.append(f"[{s_fmt} - {e_fmt}] {it['text']}")

        block_input = "\n".join(input_lines)

        prompt = (
            f"Role: Expert Translator.\n"
            f"Task: Translate the subtitles from {source_language} to {target_language}.\n"
            f"Context: Enriched, fine-grained, amazing movie transcription.\n"
            f"\n"
            f"Input Format:\n"
            f"[Start - End] Source Text\n"
            f"\n"
            f"Output Format:\n"
            f"[Start - End] Translated Text\n"
            f"\n"
            f"Requirements:\n"
            f"1. PRESERVE TIMESTAMPS: You must keep the exact same [Start - End] tags for each line.\n"
            f"2. FIDELITY: Preserve meaning, tone, and nuance. Translate everything.\n"
            f"3. STRICT FORMAT: Output line-by-line corresponding to input. No extra text, no markdown.\n"
            f"\n"
            f"Input:\n"
            f"{block_input}\n"
            f"\n"
            f"Output:\n"
        )

        # Call models with fixed-interval retries (with fallbacks)
        t0 = time.time()

        def enhanced_validator(text):
            """Enhanced validation with content quality checks."""
            # Parse back using compact parser
            parsed = parse_compact_segments(text)
            if not parsed:
                return False, "failed to parse any segments"

            # permissive count check (allow slight drift if model merged lines, but warn)
            if abs(len(parsed) - len(items)) > max(2, len(items) * 0.2):
                return (
                    False,
                    f"count mismatch severe: in={len(items)} out={len(parsed)}",
                )

            # Refusal checks
            refusal_patterns = [
                "i cannot",
                "prioritize the spoken words",
                "as an ai",
                "i am unable",
            ]
            lower = text.lower()
            if any(p in lower for p in refusal_patterns) and len(parsed) < 3:
                return False, "potential refusal detected"

            return True, None

        model_order = [model] + [m for m in translation_models if m != model]
        trans_text = ""
        stats = {}
        chosen_model = None
        total_attempts = 0
        for cand in model_order:
            trans_text, stats = _call_with_retries(
                cand,
                prompt,
                max_retries=translate_max_retries,
                validator=enhanced_validator,
                postprocess=_strip_code_fences,
                max_empty_attempts=translate_max_empty_attempts,
                translate_timeout_s=translate_timeout_s,
            )
            total_attempts += stats.get("attempts", 0) if stats else 0
            if trans_text.strip():
                chosen_model = cand
                break
        dur = time.time() - t0

        # Parse the translated output
        translated_items = parse_compact_segments(trans_text)

        # Fallback if empty
        if not translated_items:
            if verbose:
                print(f"[TL] seg {i} parse fail on {chosen_model}, fallback to source")
            # Re-construct source as fallback
            translated_items = items
            used_src = True
        else:
            used_src = False

        # Re-serialize to clean format
        # We don't need complex alignment logic anymore because we asked the model to preserve timestamps.
        # But if the model messed up timestamps, we might want to force-align to original timestamps?
        # Let's trust the model's preservation for now, but if counts match exactly, we could overlay original timestamps.
        # Actually, let's overlay original timestamps if counts match exactly to ensure perfect sync.

        final_lines = []
        if len(translated_items) == len(items):
            # Perfect match, enforce original timing
            for src_it, tr_it in zip(items, translated_items):
                s_fmt = format_ms_xmys(src_it["start_ms"])
                e_fmt = format_ms_xmys(src_it["end_ms"])
                final_lines.append(f"[{s_fmt} - {e_fmt}] {tr_it['text']}")
        else:
            # Count mismatch, use model's timing (it might have merged/split)
            for tr_it in translated_items:
                s_fmt = format_ms_xmys(tr_it["start_ms"])
                e_fmt = format_ms_xmys(tr_it["end_ms"])
                final_lines.append(f"[{s_fmt} - {e_fmt}] {tr_it['text']}")

        cleaned_t = "\n".join(final_lines)
        translated_outputs[i] = cleaned_t

        translated_path = os.path.join(
            output_dir, "translated", f"segment_{i}_translated.txt"
        )
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(cleaned_t)

        print(
            f"[TL] seg {i} ok {dur:.1f}s attempts={total_attempts} items={len(translated_items)}"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_translate_one, i, out) for i, out in enumerate(minimal_outputs)
        ]
        concurrent.futures.wait(futures)

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
    transcribe_max_attempts: Optional[int] = None,
    translate_max_empty_attempts: Optional[int] = None,
    concurrency: Optional[int] = None,
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
    transcribe_max_attempts = int(transcribe_max_attempts or 6)
    translate_max_empty_attempts = int(translate_max_empty_attempts or 6)
    concurrency = int(concurrency or 1)

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
    if total_ms is None:
        raise RuntimeError(
            "Unable to determine audio duration (ffprobe missing or failed). "
            f"Input: {input_audio}"
        )

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
                and saved_durations
                and isinstance(saved_durations, list)
                and len(saved_durations) == len(saved_paths)
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
                json.dump(
                    {
                        "base": base,
                        "total_ms": total_ms,
                        "segment_paths": seg_paths,
                        "offsets_ms": offsets_ms,
                        "durations_ms": seg_durations_ms,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
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
                json.dump(
                    {
                        "base": base,
                        "total_ms": total_ms,
                        "segment_paths": seg_paths,
                        "offsets_ms": offsets_ms,
                        "durations_ms": seg_durations_ms,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            if verbose:
                print(f"Warning: failed to write offsets file: {e}")
    # Guard: ensure segments were created and valid
    if not seg_paths or any(
        (not os.path.exists(p) or os.path.getsize(p) < 1024) for p in seg_paths
    ):
        raise RuntimeError("Segmenting failed: missing or tiny segment files")
    print(f"[SPLIT] ready {len(seg_paths)} segs {(time.time() - split_t0):.1f}s")
    durations_available = bool(seg_durations_ms) and len(seg_durations_ms) == len(
        seg_paths
    )
    if not durations_available and verbose:
        print("[SPLIT] warning: segment durations missing; assembly clamping disabled")

    minimal_outputs: List[str]
    if start_step.lower() in ("translate", "assemble"):
        # Resume from saved raw transcripts
        print("[TRANSCRIBE] resume", flush=True)
        minimal_outputs = []
        for i in range(len(seg_paths)):
            raw_path = os.path.join(work_output_dir, "raw", f"segment_{i}_raw.txt")
            dur_ms = (
                seg_durations_ms[i]
                if seg_durations_ms and i < len(seg_durations_ms)
                else (get_audio_duration_ms(seg_paths[i]) or 60 * 1000)
            )
            placeholder = json.dumps(
                [
                    {
                        "start": format_ms_xmys(0),
                        "end": format_ms_xmys(dur_ms),
                        "text": "[untranscribed]",
                    }
                ],
                ensure_ascii=False,
                indent=2,
            )
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                if raw_text.strip():
                    minimal_outputs.append(clean_segment_text(raw_text))
                else:
                    print(f"[TRANSCRIBE] seg {i} missing/empty raw; using placeholder")
                    minimal_outputs.append(placeholder)
            except Exception:
                print(f"[TRANSCRIBE] seg {i} missing/invalid raw; using placeholder")
                minimal_outputs.append(placeholder)
    else:
        print("[TRANSCRIBE] start", flush=True)
        tr_t0 = time.time()
        minimal_outputs = _transcribe_segments(
            seg_paths,
            source_language,
            transcription_models,
            work_output_dir,
            verbose,
            transcribe_timeout_s,
            transcribe_retry_wait_s,
            transcribe_max_attempts,
            concurrency,
        )
        print(f"[TRANSCRIBE] done {(time.time() - tr_t0):.1f}s")

    # Guard: ensure we have any non-empty transcription before translating
    if not any(s.strip() for s in minimal_outputs):
        raise RuntimeError("Transcription failed: all segments empty")

    if start_step.lower() in ("assemble",):
        print("[TRANSLATE] resume", flush=True)
        translated_outputs: List[str] = []
        for i in range(len(minimal_outputs)):
            t_path = os.path.join(
                work_output_dir, "translated", f"segment_{i}_translated.txt"
            )
            try:
                with open(t_path, "r", encoding="utf-8") as f:
                    translated_outputs.append(clean_segment_text(f.read()))
            except Exception:
                translated_outputs.append("")
    else:
        print("[TRANSLATE] start", flush=True)
        tl_t0 = time.time()
        translated_outputs = _translate_segments(
            minimal_outputs,
            source_language,
            target_language,
            translation_models,
            work_output_dir,
            verbose,
            translate_timeout_s,
            translate_max_retries,
            translate_retry_wait_s,
            translate_max_empty_attempts,
            seg_durations_ms if durations_available else None,
            concurrency,
        )
        print(f"[TRANSLATE] done {(time.time() - tl_t0):.1f}s")

    if not any(t.strip() for t in translated_outputs):
        raise RuntimeError("Translation failed: all segments empty")

    out_path = os.path.join(work_output_dir, base + ".srt")
    if start_step.lower() == "assemble" and os.path.exists(out_path):
        print("[ASSEMBLE] skip (existing SRT)", flush=True)
    else:
        print("[ASSEMBLE] start", flush=True)
        srt_text = assemble_srt_from_segments(
            translated_outputs,
            offsets_ms,
            seg_durations_ms if durations_available else None,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        print(f"[ASSEMBLE] wrote {out_path}")

    if cleanup:
        try:
            shutil.rmtree(tmp_dir)
            print("[CLEANUP] done")
        except Exception as e:
            print(f"[CLEANUP] error: {e}")

    print(f"[DONE] {time.time() - t0:.1f}s")
    return out_path
