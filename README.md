# Open Translate Pipeline

## Overview
This project extracts audio from videos, transcribes and translates it using Gemini models, and generates SRT subtitles. It splits long audios into non-overlapping segments, processes sequentially, and assembles into a single SRT with precise timing.

## Setup
- Activate conda env: `conda activate ST`
- Install: `pip install google-generativeai yt-dlp`
- Ensure ffmpeg is installed.
- Set GEMINI_API_KEY in environment (no .env files).

## Configuration (config.json)
- `path_to_vid`: YouTube URL (http/https) or local video path.
- `source_language`: e.g., "Arabic".
- `target_language`: e.g., "Latin American Spanish".
- `output_dir`: e.g., "output_srt".
- `tmp_dir`: e.g., "tmp_segments".
- `cleanup`: true/false.
- `min_segment_minutes`: e.g., 5 or 10.
- `transcription_models`: e.g., ["gemini-2.5-pro"] or ["gemini-2.5-flash"].
- `translation_models`: e.g., ["gemini-2.5-pro"] or ["gemini-2.5-flash"].
- `start_step` (number): where to start/resume.
  - 0 = download
  - 1 = split
  - 2 = transcribe
  - 3 = translate
  - 4 = assemble

## How It Works
1. **Config Load**: `main.py` loads `config.json`.
2. **Download/Local**: If URL, downloads; if local, uses directly.
3. **Audio Extraction**: Writes `extracted_audio/<base>.m4a` under `output_dir/<base>/`.
4. **Split**: Creates segments in `<tmp_dir>/` and `offsets.json` in the run root.
5. **Transcribe**: Saves `raw/segment_{i}_raw.txt`.
6. **Translate**: Saves `translated/segment_{i}_translated.txt`.
7. **Assemble**: Writes `<base>.srt`.
   - Note: There is no final SRT validation step. Validation/fixing happens at the source by cleaning transcription/translation minimal outputs.

### Start/Resume behavior
- Set `start_step` to start EXACTLY at that step:
  - `2 (transcribe)`: uses existing segments; transcribes now.
  - `3 (translate)`: loads `raw/*.txt`; translates now.
  - `4 (assemble)`: loads `translated/*.txt`; assembles now.
- When `start_step >= 2`, the app auto-detects the latest run folder in `output_dir/` (by most recent `offsets.json`). If required assets are missing, it raises a clear error.

Minimal output cleaning ensures source outputs are valid before assembly; no post-assembly SRT validation.

## Running
- Edit config.json.
- Run: `python main.py`
- Output: All results grouped under a single run folder: `output_dir/<video_base>/`.

### Run folder structure
```
output_srt/
  <video_base>/
    extracted_audio/
      <video_base>.m4a
    raw/
      segment_0_raw.txt
      ...
    translated/
      segment_0_translated.txt
      ...
    <video_base>.srt
```

For issues, check console for concise logs.
