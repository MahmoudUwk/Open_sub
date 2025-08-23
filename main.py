import os
import json
import time

from src.process_long_audio import process_audio_equal_segments


DEFAULT_CONFIG_PATH = "config.json"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_from_config(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    cfg = load_config(config_path)

    audio_path = cfg["audio_path"]
    source_language = cfg.get("source_language", "Arabic")
    target_language = cfg.get("target_language", "Spanish")
    model = cfg.get("model", "gemini-2.5-flash")
    num_segments = int(cfg.get("num_segments", cfg.get("num_calls", 10)))
    output_dir = cfg.get("output_dir", "output_srt")
    tmp_dir = cfg.get("tmp_dir", "tmp_segments")
    cleanup = bool(cfg.get("cleanup", True))

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    # Always use equal 10-way (or num_calls) parallel pipeline
    t0 = time.time()
    out_path = process_audio_equal_segments(
        input_audio=audio_path,
        source_language=source_language,
        target_language=target_language,
        model=model,
        num_segments=num_segments,
        tmp_dir=tmp_dir,
        output_dir=output_dir,
        cleanup=cleanup,
        verbose=True,
    )
    t1 = time.time()
    print(f"    -> Wrote {out_path} in {t1 - t0:.2f}s", flush=True)
    return out_path


if __name__ == "__main__":
    # No CLI; run from default config file
    result_path = run_from_config(DEFAULT_CONFIG_PATH)
    print(result_path)
