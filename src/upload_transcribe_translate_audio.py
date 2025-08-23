from google import genai
from google.genai import types
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


RATE_LIMIT_ERRORS = (
    "429", "rate limit", "exceeded", "quota", "Resource has been exhausted"
)


TRANSCRIPTION_PROMPT_TEMPLATE = (
    """
You are given an audio segment.
Task: Transcribe all speech accurately from the audio (speech is in {source_language}).
Output format: Plain text lines, each: [HH:MM:SS,ms - HH:MM:SS,ms]: transcribed text
- Use a comma before milliseconds, e.g., [00:01:02,345 - 00:01:05,678]: Text
- Times are RELATIVE to the start of this segment.
- End must be strictly greater than start.
- Break into many short subtitles: each 5-10 seconds long, with at most 1-2 sentences in text.
- Text must contain ONLY the transcribed content in {source_language}.
Rules: No JSON, no SRT, no numbering, no code fences, and no extra commentary. Just the lines.
"""
)


def transcribe_minimal(
    audio_bytes: bytes,
    mime_type: str,
    source_language: str,
    model: str = "gemini-2.5-flash",
    max_retries: int = 1,
    initial_backoff_seconds: float = 0.0,
    verbose: bool = False,
) -> str:
    """Call the model once; on rate limit, immediately fallback to flash-lite. No sleeps.

    Returns plain text per-line minimal output or empty string on failure.
    """ 
    if not isinstance(audio_bytes, (bytes, bytearray)) or len(audio_bytes) == 0:
        return ""

    prompt = TRANSCRIPTION_PROMPT_TEMPLATE.format(source_language=source_language)

    def _call_once(use_model: str) -> str:
        start = time.time()
        client = genai.Client()
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model=use_model,
            contents=[prompt, audio_part],
        )
        content_text = getattr(response, "text", None)
        return (content_text or "").strip()

    # Primary call
    try:
        out = _call_once(model)
        return out
    except Exception as e:
        msg = str(e)
        if any(token.lower() in msg.lower() for token in RATE_LIMIT_ERRORS):
            # Immediate fallback without sleep
            try:
                if verbose:
                    print("        [API] rate-limited; falling back to gemini-2.5-flash-lite", flush=True)
                return _call_once("gemini-2.5-flash-lite")
            except Exception:
                return ""
        else:
            if verbose:
                print(f"        [API {model}] error: '{msg}'", flush=True)
            return ""


