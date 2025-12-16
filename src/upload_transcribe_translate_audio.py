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

TRANSCRIPTION_PROMPT_TEMPLATE = """Role: Expert Movie Transcriber.
Task: Create an enriched, fine-grained, amazing transcription of the speech in {source_language}. Capture every nuance of the dialogue for high-quality subtitles.

Output Format:
[XmYsZms - XmYsZms] Spoken text

Example:
[0m0s214ms - 0m2s174ms] I don't think so.
[0m2s314ms - 0m3s864ms] Wait, did you hear that?

Requirements:
1. Format: Use the strict `[Start - End] Text` format for every line. Start and End must be in XmYsZms format (e.g., 9m32s839ms).
2. Accuracy & Nuance: Transcribe exactly what is said. Capture stuttering, interruptions, and emotional tone if relevant to the dialogue (e.g., [whispering], [screaming] if helpful for context, but primarily focus on the spoken words).
3. Segmentation: Break text into natural subtitle lines (1-2 sentences max). Keep segments short and rhythmic for easy reading.
4. Multilingual: The primary language is {source_language}, but preserve any other spoken languages as they are (or transliterate if necessary for the target audience).
5. Reliability: Output ONLY the transcription lines. No markdown, no JSON, no introductory text.
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
