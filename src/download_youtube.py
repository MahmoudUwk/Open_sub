#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
from typing import Optional, List

DOWNLOAD_DIR = "downloaded_videos"

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)

def find_downloader() -> List[str]:
    """Return the command list to invoke yt-dlp or youtube-dl.

    If the binary has an Exec format error (common when a script is downloaded without a proper shebang),
    wrap it with python3.
    """
    def validate(cmd: List[str]) -> Optional[List[str]]:
        try:
            subprocess.run(cmd + ["--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return cmd
        except OSError as e:
            # Exec format error -> try python wrapper
            if getattr(e, 'errno', None) == 8 or 'Exec format error' in str(e):
                try:
                    subprocess.run(["python3", cmd[0], "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return ["python3", cmd[0]]
                except Exception:
                    return None
            return None
        except Exception:
            return None

    # Prefer yt-dlp if available in PATH
    for exe in ("yt-dlp", "youtube-dl"):
        path = shutil.which(exe)
        if path:
            v = validate([path])
            if v:
                return v
    # Try common absolute paths
    candidates = [
        "/usr/local/bin/yt-dlp",
        "/usr/bin/yt-dlp",
        "/usr/local/bin/youtube-dl",
        "/usr/bin/youtube-dl",
    ]
    for path in candidates:
        if os.path.exists(path) and os.access(path, os.X_OK):
            v = validate([path])
            if v:
                return v
    raise RuntimeError("yt-dlp/youtube-dl not found or not executable. Ensure it's installed correctly.")


def get_video_id(downloader: List[str], url: str) -> str:
    cp = run(downloader + ["--no-playlist", "--get-id", url])
    vid = cp.stdout.strip().splitlines()[-1].strip()
    if not vid:
        raise RuntimeError("Failed to resolve video ID from URL")
    return vid


def _download_with_python_modules(url: str, out_dir: str) -> Optional[str]:
    """Try using yt_dlp or youtube_dl Python modules. Return path or None on failure."""
    os.makedirs(out_dir, exist_ok=True)
    # 1) Try yt_dlp
    try:
        import yt_dlp  # type: ignore

        ydl_opts = {
            'outtmpl': os.path.join(out_dir, '%(id)s.%(ext)s'),
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'format': 'bestvideo+bestaudio/best',
            'quiet': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            vid = info.get('id')
        # Check produced files
        for ext in ("mp4", "mkv", "webm"):
            candidate = os.path.join(out_dir, f"{vid}.{ext}")
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass

    # 2) Try youtube_dl
    try:
        import youtube_dl  # type: ignore

        ydl_opts = {
            'outtmpl': os.path.join(out_dir, '%(id)s.%(ext)s'),
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'format': 'bestvideo+bestaudio/best',
            'quiet': False,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            vid = info.get('id')
        for ext in ("mp4", "mkv", "webm"):
            candidate = os.path.join(out_dir, f"{vid}.{ext}")
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass

    return None


def download_youtube(url: str, out_dir: str = DOWNLOAD_DIR) -> str:
    """
    Download a single YouTube video to out_dir as MP4 and return the file path.
    Uses youtube-dl and avoids playlist downloads.
    """
    # Prefer Python modules to avoid Exec format issues
    maybe_path = _download_with_python_modules(url, out_dir)
    if maybe_path:
        return maybe_path

    downloader = find_downloader()
    os.makedirs(out_dir, exist_ok=True)

    vid = get_video_id(downloader, url)
    # Output template ensures deterministic filename
    out_tmpl = os.path.join(out_dir, "%(id)s.%(ext)s")

    # Prefer bestvideo+bestaudio merged to mp4; fallback to best
    cmd = downloader + [
        "--no-playlist",
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", out_tmpl,
        url,
    ]
    # Let it print progress to console for visibility
    cp = subprocess.run(cmd)
    if cp.returncode != 0:
        raise RuntimeError(f"youtube-dl download failed with code {cp.returncode}")

    # Resolve final path by checking for vid with common extensions
    for ext in ("mp4", "mkv", "webm", "mp4.part", "mkv.part", "webm.part"):
        candidate = os.path.join(out_dir, f"{vid}.{ext}")
        if os.path.exists(candidate):
            # If it's a partial file, consider download incomplete
            if candidate.endswith('.part'):
                raise RuntimeError("Download incomplete (.part file remains)")
            return candidate

    # If not found by ID, list recent files in out_dir and pick latest
    files = [os.path.join(out_dir, f) for f in os.listdir(out_dir)]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise RuntimeError("No downloaded file found in output directory")
    latest = max(files, key=os.path.getmtime)
    return latest


def main(argv: list) -> int:
    if len(argv) < 2:
        print("Usage: download_youtube.py <youtube_url>")
        return 2
    url = argv[1]
    try:
        path = download_youtube(url)
        print(path)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
