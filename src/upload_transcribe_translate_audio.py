"""Handles interaction with the Gemini API for transcription."""

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
    model: str = "gemini-2.5-flash",
    verbose: bool = False,
) -> str:
    """Call the model with retries on empty response; fallback on rate limit or persistent empty."""
    if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
        return ""

    prompt = TRANSCRIPTION_PROMPT_TEMPLATE.format(source_language=source_language)

    def _call_with_retries(use_model: str, max_retries: int = 3) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                client = genai.Client()
                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
                response = client.models.generate_content(
                    model=use_model,
                    contents=[prompt, audio_part],
                )
                text = (getattr(response, "text", None) or "").strip()
                if text:
                    return text
                if verbose:
                    print(f"        [API {use_model}] Empty response on attempt {attempt}/{max_retries}")
            except Exception as e:
                if verbose:
                    print(f"        [API {use_model}] Error on attempt {attempt}: '{str(e)}'")
                if attempt == max_retries:
                    raise
        return ""  # All retries failed or empty

    try:
        return _call_with_retries(model)
    except Exception as e:
        msg = str(e)
        if any(token.lower() in msg.lower() for token in RATE_LIMIT_ERRORS):
            if verbose:
                print("        [API] rate-limited; falling back to gemini-2.5-flash-lite", flush=True)
            try:
                return _call_with_retries("gemini-2.5-flash-lite")
            except Exception as fallback_e:
                if verbose:
                    print(f"        [API gemini-2.5-flash-lite] fallback error: '{fallback_e}'", flush=True)
                return ""
        else:
            if verbose:
                print(f"        [API {model}] error: '{msg}'", flush=True)
            return ""