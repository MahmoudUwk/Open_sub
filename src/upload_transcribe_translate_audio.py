from google import genai
from google.genai import types
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


RATE_LIMIT_ERRORS = (
    "429", "rate limit", "exceeded", "quota", "Resource has been exhausted"
)


MINIMAL_JSONL_PROMPT_TEMPLATE = (
    """
You are given an audio segment.
Task: Transcribe all speech accurately from the audio (speech is in {source_language}), then translate the transcription to {target_language}.
Output format: STRICT JSONL (one JSON object per line).
Each object MUST be: {{"start": "HH:MM:SS,ms", "end": "HH:MM:SS,ms", "text": "..."}}
- Use a comma before milliseconds, e.g., 00:01:02,345 (not dots).
- Times are RELATIVE to the start of this segment.
- end must be strictly greater than start.
- Break into many short subtitles: each 5-10 seconds long, with at most 1-2 sentences in 'text'.
- 'text' must contain ONLY the translated content in {target_language}. Do NOT include original language text.
Rules: No SRT, no numbering, no code fences, and no extra commentary.
"""
)


def transcribe_translate_minimal(
    audio_bytes: bytes,
    mime_type: str,
    source_language: str,
    target_language: str,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
    initial_backoff_seconds: float = 2.0,
    verbose: bool = True,
) -> str:
    """Call the model with inline audio bytes; return minimal JSONL string.

    Raises ValueError if empty after retries.
    """
    if not isinstance(audio_bytes, (bytes, bytearray)) or len(audio_bytes) == 0:
        return ""

    prompt = MINIMAL_JSONL_PROMPT_TEMPLATE.format(
        source_language=source_language,
        target_language=target_language,
    )

    backoff = initial_backoff_seconds
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            if verbose:
                print(f"        [API {model}] attempt {attempt}/{max_retries}...", flush=True)
            start = time.time()
            client = genai.Client()
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model=model,
                contents=[prompt, audio_part],
            )
            content_text = getattr(response, "text", None)
            if isinstance(content_text, str) and content_text.strip():
                if verbose:
                    dur = time.time() - start
                    print(f"        [API {model}] success in {dur:.2f}s (chars={len(content_text)})", flush=True)
                return content_text.strip()
            last_error = ValueError("Empty response text from model")
            if verbose:
                print(f"        [API {model}] empty response; retrying...", flush=True)
        except Exception as e:
            last_error = e
            msg = str(e)
            if any(token.lower() in msg.lower() for token in RATE_LIMIT_ERRORS):
                if verbose:
                    print(f"        [API {model}] rate limit: '{msg}'. Backing off {backoff:.1f}s", flush=True)
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                if verbose:
                    print(f"        [API {model}] error: '{e}'. Retrying in {min(backoff, 5):.1f}s", flush=True)
                time.sleep(min(backoff, 5))
                continue
        time.sleep(backoff)
        backoff *= 2

    raise ValueError(f"Failed to obtain minimal output after {max_retries} attempts: {last_error}")


