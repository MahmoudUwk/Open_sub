import os
import subprocess


def extract_audio(
    video_path: str,
    output: str = None,
    format: str = "mp3",
    quality: str = "high",
    mono: bool = False,
    copy: bool = False,
    bitrate: str = None,
    sample_rate: int = None,
    transcribe: bool = False,
) -> str:
    """Extract audio from a video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")
    if os.path.isdir(video_path):
        raise ValueError(f"Input '{video_path}' is a directory; please provide a video file path.")
    
    os.makedirs("extracted_audio", exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    
    if transcribe:
        output_path = output or os.path.join("extracted_audio", base + ".m4a")
        sample_rate_value = str(sample_rate or 16000)
        aac_cmd = [
            "ffmpeg", "-y", "-nostdin", "-i", video_path, "-vn",
            "-loglevel", "error", "-sn", "-dn",
            "-err_detect", "ignore_err",
            "-ac", "1",
            "-ar", sample_rate_value,
            "-c:a", "aac",
            "-b:a", "16k",
            output_path,
        ]
        try:
            subprocess.run(aac_cmd, check=True, capture_output=True, text=True)
            print(f"Extracted audio: {output_path}")
            return output_path
        except subprocess.CalledProcessError:
            # Fallback to WAV PCM extraction
            fallback_path = (output or os.path.join("extracted_audio", base + ".wav"))
            wav_cmd = [
                "ffmpeg", "-y", "-nostdin", "-i", video_path, "-vn",
                "-loglevel", "error", "-sn", "-dn",
                "-map", "0:a:0",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                fallback_path,
            ]
            try:
                subprocess.run(wav_cmd, check=True, capture_output=True, text=True)
                print(f"Extracted audio: {fallback_path}")
                return fallback_path
            except subprocess.CalledProcessError:
                # Final fallback: stream copy (no decode). Downsampling will happen during splitting.
                copy_path = output or os.path.join("extracted_audio", base + ".m4a")
                copy_cmd = [
                    "ffmpeg", "-y", "-nostdin", "-i", video_path, "-vn",
                    "-loglevel", "error", "-sn", "-dn",
                    "-map", "0:a:0",
                    "-c:a", "copy",
                    copy_path,
                ]
                subprocess.run(copy_cmd, check=True, capture_output=True, text=True)
                print(f"Extracted audio: {copy_path}")
                return copy_path
    elif copy:
        output_path = output or os.path.join("extracted_audio", base + ".m4a")
        if mono or sample_rate or bitrate:
            print("Note: copy with channel/rate/bitrate changes not supported; re-encoding.")
            bitrate_value = bitrate or "128k"
            cmd = ["ffmpeg", "-y", "-nostdin", "-i", video_path, "-vn"]
            if mono:
                cmd.extend(["-ac", "1"])
            if sample_rate:
                cmd.extend(["-ar", str(sample_rate)])
            cmd.extend(["-c:a", "aac", "-b:a", bitrate_value, output_path])
        else:
            cmd = ["ffmpeg", "-y", "-nostdin", "-i", video_path, "-vn", "-c:a", "copy", output_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Extracted audio: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error extracting audio: {e}")
    else:
        format_ext = {"mp3": "mp3", "aac": "m4a", "vorbis": "ogg", "flac": "flac", "wav": "wav", "aiff": "aiff"}
        output_path = output or os.path.join("extracted_audio", base + "." + format_ext[format])
        
        quality_map = {
            "mp3": {"high": "2", "medium": "5", "low": "7"},
            "aac": {"high": "192k", "medium": "128k", "low": "96k"},
            "vorbis": {"high": "192k", "medium": "128k", "low": "96k"},
            "flac": {"high": "0", "medium": "5", "low": "8"},
            "wav": {"high": None, "medium": None, "low": None},
            "aiff": {"high": None, "medium": None, "low": None},
        }
        quality_param = quality_map[format][quality]
        
        cmd = ["ffmpeg", "-y", "-nostdin", "-i", video_path, "-vn", "-loglevel", "error"]
        if mono:
            cmd.extend(["-ac", "1"])
        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        
        if format == "mp3":
            if bitrate:
                cmd.extend(["-c:a", "libmp3lame", "-b:a", bitrate])
            else:
                cmd.extend(["-c:a", "libmp3lame", "-q:a", quality_param])
        elif format == "aac":
            cmd.extend(["-c:a", "aac", "-b:a", bitrate or quality_param])
        elif format == "vorbis":
            cmd.extend(["-c:a", "libvorbis", "-b:a", bitrate or quality_param])
        elif format == "flac":
            cmd.extend(["-c:a", "flac"])
            if quality_param is not None:
                cmd.extend(["-compression_level", quality_param])
        elif format == "wav":
            cmd.extend(["-c:a", "pcm_s16le"])
        elif format == "aiff":
            cmd.extend(["-c:a", "pcm_s16be"])
        cmd.append(output_path)
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Extracted audio: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error extracting audio: {e}")



