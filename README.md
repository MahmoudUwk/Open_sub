# Open Translate Pipeline

## Overview
This project extracts audio from videos, transcribes and translates it using Gemini-2.5-pro, and generates SRT subtitles. It splits long audios into non-overlapping segments, processes sequentially, and assembles into a single SRT with precise timing.

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
- `transcription_models`: ["gemini-2.5-pro"] (default).
- `translation_models`: ["gemini-2.5-pro"] (default).

## How It Works
1. **Config Load**: `main.py` loads `config.json`.
2. **Input handling**: If `path_to_vid` is a URL, it downloads the video to `downloaded_videos/` via `scripts/download_youtube.py`. If it's a local path, it uses it directly.
3. **Audio Extraction** (`src/get_audio.py`): Stream-copies audio from the downloaded video to `.m4a` in `extracted_audio/` (no re-encode).
4. **Splitting** (`src/audio_utils.py`): Splits audio with ffmpeg segmenter using stream copy (no decode/re-encode) into `tmp_segments/`.
5. **Transcription** (`src/upload_transcribe_translate_audio.py`): Uses Gemini for timed transcription per segment.
6. **Translation**: Translates each transcription using Gemini.
7. **Assembly** (`src/minimal_format.py`): Shifts timings cumulatively, sorts, formats to SRT.
8. **Cleanup**: Optional deletion of temp files.

Validations ensure steps succeed before proceeding.

## Running
- Edit config.json.
- Run: `python main.py`
- Output: SRT in output_dir (e.g., Film1.srt).

For issues, check console for concise logs.
