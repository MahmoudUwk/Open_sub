"""Main entry point for the audio processing pipeline."""

import os
import json
import time
from dataclasses import dataclass
from typing import List, Optional

from src.process_long_audio import process_audio_fixed_duration

DEFAULT_CONFIG_PATH = "config.json"

@dataclass
class PipelineConfig:
    """Configuration for the audio processing pipeline."""
    audio_path: str
    source_language: str
    target_language: str
    output_dir: str
    tmp_dir: str
    cleanup: bool
    min_segment_minutes: int
    segment_overlap_seconds: int
    transcription_models: Optional[List[str]] = None
    translation_models: Optional[List[str]] = None

def load_config(path: str = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Load pipeline configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return PipelineConfig(
        audio_path=cfg["audio_path"],
        source_language=cfg.get("source_language", "Arabic"),
        target_language=cfg.get("target_language", "Spanish"),
        output_dir=cfg.get("output_dir", "output_srt"),
        tmp_dir=cfg.get("tmp_dir", "tmp_segments"),
        cleanup=bool(cfg.get("cleanup", True)),
        min_segment_minutes=int(cfg.get("min_segment_minutes", 15)),
        segment_overlap_seconds=int(cfg.get("segment_overlap_seconds", 2)),
        transcription_models=cfg.get("transcription_models"),
        translation_models=cfg.get("translation_models"),
    )

def run_from_config(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Run the audio processing pipeline from a configuration file."""
    config = load_config(config_path)

    if not os.path.exists(config.audio_path):
        raise FileNotFoundError(f"Input audio not found: {config.audio_path}")

    t0 = time.time()
    out_path = process_audio_fixed_duration(
        input_audio=config.audio_path,
        source_language=config.source_language,
        target_language=config.target_language,
        min_segment_minutes=config.min_segment_minutes,
        segment_overlap_seconds=config.segment_overlap_seconds,
        tmp_dir=config.tmp_dir,
        output_dir=config.output_dir,
        cleanup=config.cleanup,
        verbose=True,
        transcription_models=config.transcription_models,
        translation_models=config.translation_models,
    )
    t1 = time.time()
    print(f"    -> Wrote {out_path} in {t1 - t0:.2f}s", flush=True)
    return out_path

if __name__ == "__main__":
    run_from_config(DEFAULT_CONFIG_PATH)