"""Main entry point for the audio processing pipeline."""

import os
import json
import time
import shutil
from dataclasses import dataclass
from typing import List, Optional

from src.process_long_audio import process_audio_fixed_duration
from src.get_audio import extract_audio
from src.audio_utils import get_audio_duration_ms
from scripts.download_youtube import download_youtube

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

def run_from_config(config_path: str = "config.json") -> None:
    """Load configuration from JSON and run the processing pipeline."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    tmp_dir = config.get("tmp_dir", "tmp_segments")
    output_dir = config.get("output_dir", "output_srt")
    extracted_dir = "extracted_audio"
    
    # Clean up previous runs (tmp, output, and extracted_audio)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir, ignore_errors=True)

    # Recreate fresh directories
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(extracted_dir, exist_ok=True)
    
    path_to_vid = config.get("path_to_vid")
    if not path_to_vid:
        raise ValueError("path_to_vid must be specified in config (URL or local file path)")

    # Decide whether to download or use local file directly
    def _is_url(s: str) -> bool:
        s = s.strip().lower()
        return s.startswith("http://") or s.startswith("https://") or s.startswith("youtu.be/") or s.startswith("www.youtube.com/")

    if _is_url(path_to_vid):
        print("[0] Downloading video...", flush=True)
        local_video_path = download_youtube(path_to_vid, out_dir="downloaded_videos")
        if not os.path.exists(local_video_path):
            raise RuntimeError(f"Download failed or missing file: {local_video_path}")
        print(f"Downloaded: {local_video_path}", flush=True)
    else:
        local_video_path = path_to_vid
        if not os.path.exists(local_video_path) or os.path.isdir(local_video_path):
            raise FileNotFoundError(f"Local video path not found or is a directory: {local_video_path}")
        print(f"[0] Using local video: {local_video_path}", flush=True)

    # Extract audio from downloaded video
    audio_path = extract_audio(local_video_path, transcribe=True)

    # Validate extracted audio
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 4096:
        raise RuntimeError(f"Audio extraction failed or too small: {audio_path}")
    extracted_ms = get_audio_duration_ms(audio_path)
    if not extracted_ms or extracted_ms < 5000:
        raise RuntimeError(f"Invalid extracted audio duration: {extracted_ms} ms")

    source_language = config.get("source_language")
    target_language = config.get("target_language")
    min_segment_minutes = config.get("min_segment_minutes", 15)
    tmp_dir = config.get("tmp_dir", "tmp_segments")
    output_dir = config.get("output_dir", "output_srt")
    cleanup = config.get("cleanup", True)
    transcription_models = config.get("transcription_models", ["gemini-2.5-pro"])
    translation_models = config.get("translation_models", ["gemini-2.5-pro"])
    max_workers = int(config.get("max_workers", 5))
    
    out_path = process_audio_fixed_duration(
        input_audio=audio_path,
        source_language=source_language,
        target_language=target_language,
        min_segment_minutes=min_segment_minutes,
        tmp_dir=tmp_dir,
        output_dir=output_dir,
        cleanup=cleanup,
        verbose=True,
        transcription_models=transcription_models,
        translation_models=translation_models,
        max_workers=max_workers,
    )
    print(f"SRT: {out_path}")

if __name__ == "__main__":
    run_from_config()