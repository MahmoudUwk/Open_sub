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

def run_from_config(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Load configuration from JSON and run the processing pipeline."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    audio_path = config.get("audio_path")
    source_language = config.get("source_language")
    target_language = config.get("target_language")
    min_segment_minutes = config.get("min_segment_minutes", 15)
    tmp_dir = config.get("tmp_dir", "tmp_segments")
    output_dir = config.get("output_dir", "output_srt")
    cleanup = config.get("cleanup", True)
    transcription_models = config.get("transcription_models", ["gemini-2.5-pro"])
    translation_models = config.get("translation_models", ["gemini-2.5-pro"])
    
    if not audio_path:
        raise ValueError("audio_path must be specified in config")
    
    out_path = process_audio_fixed_duration(
        input_audio=audio_path,
        source_language=source_language,
        target_language=target_language,
        min_segment_minutes=min_segment_minutes,
        tmp_dir=tmp_dir,
        output_dir=output_dir,
        cleanup=cleanup,
        verbose=True,  # Assuming verbose=True
        transcription_models=transcription_models,
        translation_models=translation_models,
    )
    print(f"Processing completed. Output SRT: {out_path}")

if __name__ == "__main__":
    run_from_config()