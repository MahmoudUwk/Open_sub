
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.minimal_format import assemble_srt_from_minimal_segments
from src.audio_utils import get_audio_duration_ms

TRANSLATED_DIR = "output_srt/translated"
TMP_DIR = "tmp_segments"
OUTPUT_PATH = "test_outputs/merged.srt"
SEGMENT_MINUTES = 15  # From config
SEGMENT_MS = SEGMENT_MINUTES * 60 * 1000

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# List and sort translated files
trans_files = sorted([f for f in os.listdir(TRANSLATED_DIR) if f.endswith("_translated.txt")])

segment_outputs = []
for file in trans_files:
    with open(os.path.join(TRANSLATED_DIR, file), "r", encoding="utf-8") as f:
        segment_outputs.append(f.read())

# List matching segments for durations
seg_files = []
if os.path.exists(TMP_DIR):
    seg_files = sorted([f for f in os.listdir(TMP_DIR) if f.startswith("seg_") and f.endswith(".m4a")])
if len(seg_files) != len(trans_files):
    print("Warning: Mismatch or no segments found; using fixed offsets")
    offsets_ms = [i * SEGMENT_MS for i in range(len(segment_outputs))]
    durations_ms = None
else:
    durations_ms = []
    for seg in seg_files:
        path = os.path.join(TMP_DIR, seg)
        dur = get_audio_duration_ms(path)
        durations_ms.append(dur if dur is not None else SEGMENT_MS)  # Fallback to fixed if error

    # Compute cumulative offsets
    offsets_ms = [0]
    for d in durations_ms[:-1]:
        offsets_ms.append(offsets_ms[-1] + d)

# Call merging
srt_text = assemble_srt_from_minimal_segments(segment_outputs, offsets_ms, durations_ms)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(srt_text)

print(f"Merged SRT saved to {OUTPUT_PATH}")
