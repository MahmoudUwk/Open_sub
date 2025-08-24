# Open Translate Pipeline

## Overview
This project extracts audio from videos, transcribes and translates it using Gemini models, and generates SRT subtitles. It splits long audios into non-overlapping segments, processes sequentially, and assembles into a single SRT with precise timing.

## Setup
- Activate conda env: `conda activate ST`
- Install: `pip install google-generativeai yt-dlp`
- Ensure ffmpeg is installed.
- Set GEMINI_API_KEY in environment (no .env files).

## Configuration (config.json)
- `path_to_vid`: Either a YouTube URL (http/https) or a local video file path (e.g., .mp4). If URL, the pipeline downloads it first; if local path, it uses it directly.
- `source_language`: Audio language (e.g., "Arabic").
- `target_language`: Translation language (e.g., "Latin American Spanish").
- `output_dir`: SRT output dir (e.g., "output_srt").
- `tmp_dir`: Temp segments dir (e.g., "tmp_segments").
- `cleanup`: Delete tmp files after (true/false).
- `min_segment_minutes`: Segment length in minutes (e.g., 5).
- `transcription_models`: ["gemini-2.5-pro"] (default primary model per segment).
- `transcription_fallback_models`: list of fallback models tried in order if the primary attempt is empty or errors (e.g., rate limit). Default: `["gemini-2.5-flash"]`.
- `translation_models`: ["gemini-2.5-pro"] (or your preferred primary model per segment).
- `translation_fallback_models`: list of fallback models for translation, tried in order. Default: `["gemini-2.5-flash"]`.

## How It Works
1. **Config Load**: `main.py` loads `config.json`.
2. **Input handling**: If `path_to_vid` is a URL, it downloads the video to `downloaded_videos/` via `src/download_youtube.py`. If it's a local path, it uses it directly.
3. **Audio Extraction** (`src/get_audio.py`): Stream-copies audio from the video (no re-encode) into the run folder under `output_dir/<video_base>/extracted_audio/`.
4. **Splitting** (`src/audio_utils.py`): Splits audio into segments inside `output_dir/<video_base>/<tmp_dir>/`.
5. **Transcription** (`src/upload_transcribe_translate_audio.py`): For each segment, tries the primary model once; on empty/error, iterates `transcription_fallback_models` with up to 3 retries each. Saves raw outputs in `output_dir/<video_base>/raw/`.
6. **Translation** (`src/process_long_audio.py`): For each segment, tries the primary translation model once; on empty/error, iterates `translation_fallback_models` with up to 3 retries each. Saves translated text in `output_dir/<video_base>/translated/`.
7. **Assembly** (`src/minimal_format.py`): Produces final SRT at `output_dir/<video_base>/<video_base>.srt`.
8. **Cleanup**: Optional deletion of only the run-specific temp folder inside `output_dir/<video_base>/`.

Validations ensure steps succeed before proceeding.

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
