#!/usr/bin/env python3
"""Download audio from YouTube video for voice cloning."""

import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
RAW_AUDIO_DIR = ASSETS_DIR / "raw"

def download_audio(url: str, output_name: str = "karina_sample") -> Path:
    """Download audio from YouTube in best quality.

    Args:
        url: YouTube video URL
        output_name: Base name for output file

    Returns:
        Path to downloaded audio file
    """
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_AUDIO_DIR / f"{output_name}.%(ext)s"

    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",  # WAV format for best quality
        "--audio-quality", "0",  # Best quality
        "-o", str(output_path),
        url
    ]

    print(f"Downloading audio from: {url}")
    subprocess.run(cmd, check=True)

    # Find the downloaded file
    downloaded = list(RAW_AUDIO_DIR.glob(f"{output_name}.*"))
    if downloaded:
        print(f"Downloaded: {downloaded[0]}")
        return downloaded[0]
    raise FileNotFoundError("Download failed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_audio.py <youtube_url> [output_name]")
        sys.exit(1)

    url = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "karina_sample"
    download_audio(url, name)
