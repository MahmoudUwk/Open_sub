# Open Translate Pipeline

## Overview
This project transcribes and translates audio files (e.g., from videos) using Google's Gemini models, generating SRT subtitle files. It handles long audios by splitting into segments, processing in parallel/sequentially with model alternation, and assembling with overlap handling.

## Setup
- Activate conda environment: `conda activate ST`
- Install dependencies: `pip install google-generativeai`
- Ensure ffmpeg is installed (for audio splitting/duration).
- Set GEMINI_API_KEY in .env or environment.

## Configuration (config.json)
- `audio_path`: Path to input audio file (e.g., "extracted_audio/Film1.m4a").
- `source_language`: Language of speech in audio (e.g., "Arabic").
- `target_language`: Target translation language (e.g., "Spanish").
- `output_dir`: Directory for SRT output (e.g., "output_srt").
- `tmp_dir`: Temporary directory for segments (e.g., "tmp_segments").
- `cleanup`: Boolean to delete tmp files after (true/false).
- `min_segment_minutes`: Minimum segment length in minutes (e.g., 15).
- `segment_overlap_seconds`: Overlap between segments in seconds (e.g., 2) to avoid gaps.
- `transcription_models`: List of models to alternate for transcription (e.g., ["gemini-2.5-flash", "gemini-2.5-flash-lite"]).
- `translation_models`: List of models to alternate for translation (similar).

## How It Works
1. **Loading/Config**: `main.py` loads config.json and calls `process_audio_fixed_duration` from `src/process_long_audio.py`.
2. **Audio Splitting** (`src/audio_utils.py`): Splits input audio into ~15-min segments with 2s overlap using ffmpeg for precise cutting and mono downmix.
3. **Transcription** (`src/upload_transcribe_translate_audio.py`): Calls Gemini (alternating models) on each segment for timed transcription in simple [start - end]: text format.
4. **Translation** (`src/process_long_audio.py`): Translates each transcribed line using alternating models via Gemini API.
5. **Assembly** (`src/minimal_format.py`): Parses outputs, applies global offsets, sorts, dedups/merges overlapping similar entries (normalizes text, merges if overlap or close), builds SRT.
6. **Outputs**: SRT in output_dir; raw transcriptions in output_dir/raw/.

Overlaps ensure continuity; assembly merges duplicates across overlaps.

## Running
- Edit config.json as needed.
- Run: `python main.py`
- Output: SRT file in output_dir, e.g., Film1.srt.

For issues, check console logs for errors/rate limits.
