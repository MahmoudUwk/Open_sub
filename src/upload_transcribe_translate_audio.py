"""Handles interaction with the Gemini API for transcription."""

import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

RATE_LIMIT_ERRORS = (
    "429", "rate limit", "exceeded", "quota", "Resource has been exhausted"
)

TRANSCRIPTION_PROMPT_TEMPLATE = (
    """You are given an audio segment from a movie.
Task: Transcribe all speech accurately from the audio in {source_language}.
Output format: Plain text lines, each: [HH:MM:SS,ms - HH:MM:SS,ms]: transcribed text
- Use a comma before milliseconds, e.g., [00:01:02,345 - 00:01:05,678]: Text
- Break into many short subtitles: each 5-10 seconds long, with at most 1-2 sentences in text.
"""
)

def transcribe_minimal(
    audio_bytes: bytes,
    mime_type: str,
    source_language: str,
    model: str = "gemini-2.5-pro",
    verbose: bool = False,
) -> str:
    """Call the model with retries on empty response or errors like rate limits."""
    if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
        return ""

    prompt = TRANSCRIPTION_PROMPT_TEMPLATE.format(source_language=source_language)

    def _call_once(use_model: str) -> tuple[str, str | None]:
        try:
            client = genai.Client()
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model=use_model,
                contents=[prompt, audio_part],
            )
            text = (getattr(response, "text", None) or "").strip()
            return text, None
        except Exception as e:
            return "", str(e)

    def _call_with_retries(use_model: str, max_retries: int = 4) -> str:
        for attempt in range(1, max_retries + 1):
            text, err = _call_once(use_model)
            if text:
                return text
            if verbose:
                if err:
                    print(f"        [API {use_model}] Error on attempt {attempt}/{max_retries}: '{err}'")
                else:
                    print(f"        [API {use_model}] Empty response on attempt {attempt}/{max_retries}")
            # Fixed backoff: wait 60s on any empty response or error before retrying (no exponential)
            if attempt < max_retries:
                if verbose:
                    print("        Waiting 60s before next retry...")
                time.sleep(60)
        return ""

    # Call primary model with fixed-interval retries (no fallbacks)
    return _call_with_retries(model, max_retries=4)