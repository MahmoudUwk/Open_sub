import os

from .process_long_audio import process_audio_equal_segments


AUDIO_PATH = "extracted_audio/Film1.m4a"
OUTPUT_DIR = "output_srt"
MODEL = "gemini-2.5-flash"
SOURCE_LANGUAGE = "Arabic"
TARGET_LANGUAGE = "Spanish"


if __name__ == "__main__":
    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    out_path = process_audio_equal_segments(
        input_audio=AUDIO_PATH,
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        model=MODEL,
        num_segments=10,
        tmp_dir="tmp_segments",
        output_dir=OUTPUT_DIR,
        cleanup=True,
        verbose=True,
    )
    print(f"Wrote SRT to {out_path}")


