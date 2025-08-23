#!/usr/bin/env python3

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Extract audio from a video file.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", help="Path to the output audio file (default: extracted_audio/<basename>.<format>)")
    parser.add_argument("--format", default="mp3", choices=["mp3", "aac", "vorbis", "flac", "wav", "aiff"], help="Output audio format (default: mp3; vorbis=OGG Vorbis)")
    parser.add_argument("--quality", default="high", choices=["high", "medium", "low"], help="Quality level (default: high)")
    parser.add_argument("--mono", action="store_true", help="Downmix to mono audio (reduces size)")
    parser.add_argument("--copy", action="store_true", help="Copy audio stream without re-encoding (preserves exact quality, format becomes m4a)")
    parser.add_argument("--bitrate", help="Target audio bitrate, e.g., 48k (overrides --quality; not for wav/aiff/flac)")
    parser.add_argument("--sample_rate", type=int, help="Output sample rate in Hz, e.g., 16000")
    parser.add_argument("--transcribe", action="store_true", help="Use speech-optimized settings (mono, 16kHz, AAC 48k)")
    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    os.makedirs("extracted_audio", exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]

    if args.transcribe:
        output_path = args.output or os.path.join("extracted_audio", base + ".m4a")
        bitrate_value = args.bitrate or "48k"
        sample_rate_value = str(args.sample_rate or 16000)
        cmd = [
            "ffmpeg", "-i", video_path, "-vn",
            "-ac", "1",
            "-ar", sample_rate_value,
            "-c:a", "aac",
            "-b:a", bitrate_value,
            output_path,
        ]
    elif args.copy:
        output_path = args.output or os.path.join("extracted_audio", base + ".m4a")
        if args.mono or args.sample_rate or args.bitrate:
            print("Note: Cannot copy with channel/sample-rate/bitrate changes; falling back to high-quality re-encoding.")
            bitrate_value = args.bitrate or "128k"
            cmd = ["ffmpeg", "-i", video_path, "-vn"]
            if args.mono:
                cmd.extend(["-ac", "1"])
            if args.sample_rate:
                cmd.extend(["-ar", str(args.sample_rate)])
            cmd.extend(["-c:a", "aac", "-b:a", bitrate_value, output_path])
        else:
            cmd = ["ffmpeg", "-i", video_path, "-vn", "-c:a", "copy", output_path]
    else:
        format_ext = {"mp3": "mp3", "aac": "m4a", "vorbis": "ogg", "flac": "flac", "wav": "wav", "aiff": "aiff"}
        output_path = args.output or os.path.join("extracted_audio", base + "." + format_ext[args.format])

        quality_map = {
            "mp3": {"high": "2", "medium": "5", "low": "7"},
            "aac": {"high": "192k", "medium": "128k", "low": "96k"},
            "vorbis": {"high": "192k", "medium": "128k", "low": "96k"},
            "flac": {"high": "0", "medium": "5", "low": "8"},
            "wav": {"high": None, "medium": None, "low": None},
            "aiff": {"high": None, "medium": None, "low": None},
        }
        quality_param = quality_map[args.format][args.quality]

        cmd = ["ffmpeg", "-i", video_path, "-vn"]
        if args.mono:
            cmd.extend(["-ac", "1"])
        if args.sample_rate:
            cmd.extend(["-ar", str(args.sample_rate)])

        if args.format == "mp3":
            if args.bitrate:
                cmd.extend(["-c:a", "libmp3lame", "-b:a", args.bitrate])
            else:
                cmd.extend(["-c:a", "libmp3lame", "-q:a", quality_param])
        elif args.format == "aac":
            cmd.extend(["-c:a", "aac", "-b:a", args.bitrate or quality_param])
        elif args.format == "vorbis":
            cmd.extend(["-c:a", "libvorbis", "-b:a", args.bitrate or quality_param])
        elif args.format == "flac":
            cmd.extend(["-c:a", "flac"])
            if quality_param is not None:
                cmd.extend(["-compression_level", quality_param])
        elif args.format == "wav":
            cmd.extend(["-c:a", "pcm_s16le"])
        elif args.format == "aiff":
            cmd.extend(["-c:a", "pcm_s16be"])
        cmd.append(output_path)

    try:
        subprocess.run(cmd, check=True)
        print(f"Audio extracted successfully to '{output_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")


if __name__ == "__main__":
    main()


