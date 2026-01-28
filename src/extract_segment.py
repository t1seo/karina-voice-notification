#!/usr/bin/env python3
"""Extract clean voice segment from audio file."""

import sys
from pathlib import Path

from pydub import AudioSegment

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
RAW_AUDIO_DIR = ASSETS_DIR / "raw"
CLEAN_AUDIO_DIR = ASSETS_DIR / "clean"


def extract_segment(
    input_file: Path, start_ms: int, end_ms: int, output_name: str = "karina_clean"
) -> Path:
    """Extract a segment from audio file.

    Args:
        input_file: Path to input audio
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        output_name: Output file name (without extension)

    Returns:
        Path to extracted segment
    """
    CLEAN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_file}")
    audio = AudioSegment.from_file(input_file)

    # Extract segment
    segment = audio[start_ms:end_ms]

    # Normalize audio levels
    segment = segment.normalize()

    # Export as WAV (16kHz mono for TTS compatibility)
    output_path = CLEAN_AUDIO_DIR / f"{output_name}.wav"
    segment = segment.set_frame_rate(16000).set_channels(1)
    segment.export(output_path, format="wav")

    duration_sec = len(segment) / 1000
    print(f"Extracted {duration_sec:.1f}s segment to: {output_path}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_segment.py <input_file> <start_ms> <end_ms> [output_name]")
        print("Example: python extract_segment.py assets/raw/karina_sample.wav 5000 15000")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    start_ms = int(sys.argv[2])
    end_ms = int(sys.argv[3])
    output_name = sys.argv[4] if len(sys.argv) > 4 else "karina_clean"

    extract_segment(input_file, start_ms, end_ms, output_name)
