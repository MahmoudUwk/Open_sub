"""Handles interaction with the Gemini API for transcription."""

import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

RATE_LIMIT_ERRORS = (
    "429",
    "rate limit",
    "exceeded",
    "quota",
    "Resource has been exhausted",
)

TRANSCRIPTION_PROMPT_TEMPLATE = """Transcribe speech in {source_language}.
Output ONLY a JSON list of objects in this EXACT format:
[
  {{"start": "XmYsZms", "end": "XmYsZms", "text": "..."}},
  ...
]
Rules:
- start,end are in format XmYsZms. Example 9m32s839ms is 9 minutes, 32 seconds, and 839 milliseconds.
- Group words into readable phrases suitable for subtitles (max 1–2 sentences per line).
"""


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
                config=genai.types.GenerateContentConfig(
                    safety_settings=[
                        genai.types.SafetySetting(
                            category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
                        )
                    ]
                ),
            )
            text = (getattr(response, "text", None) or "").strip()
            return text, None
        except Exception as e:
            return "", str(e)

    def _call_with_retries(use_model: str, max_retries: int = 3) -> str:
        for attempt in range(1, max_retries + 1):
            text, err = _call_once(use_model)
            if text:
                return text
            if verbose:
                msg = err or "empty"
                print(f"[TRAPI] {use_model} attempt {attempt}/{max_retries}: {msg}")
            # Fixed backoff: wait 20s on any empty response or error before retrying (no exponential)
            if attempt < max_retries:
                if verbose:
                    print("[TRAPI] wait 20s before retry...")
                time.sleep(20)
        return ""

    # Call primary model with fixed-interval retries (no fallbacks)
    return _call_with_retries(model, max_retries=3)
