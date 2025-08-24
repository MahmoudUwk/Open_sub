
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.audio_utils import split_audio_by_duration

CONFIG_PATH = "config.json"  # At workspace root

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

audio_path = config.get("audio_path")
min_segment_minutes = config.get("min_segment_minutes", 15)
tmp_dir = config.get("tmp_dir", "tmp_segments")

if not audio_path or not os.path.exists(audio_path):
    print(f"Audio file not found: {audio_path}")
else:
    print(f"Splitting {audio_path} into {tmp_dir} with {min_segment_minutes}min segments...")
    seg_paths, offsets_ms, durations_ms, total_ms = split_audio_by_duration(
        audio_path, min_segment_minutes, tmp_dir, verbose=True
    )
    print("Splitting completed.")
    print("Generated segments:", seg_paths)
