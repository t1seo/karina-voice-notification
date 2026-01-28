#!/usr/bin/env python3
"""Transcribe audio using faster-whisper (GPU optimized)."""

import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CLEAN_AUDIO_DIR = ASSETS_DIR / "clean"
TRANSCRIPTS_DIR = ASSETS_DIR / "transcripts"


def transcribe_audio(audio_path: Path, language: str = "ko") -> dict:
    """Transcribe audio file using faster-whisper on GPU.

    Args:
        audio_path: Path to audio file
        language: Language code (ko for Korean)

    Returns:
        Transcription result dict with 'text' key
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper model (large-v3) on GPU...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    print(f"Transcribing: {audio_path}")
    segments, info = model.transcribe(str(audio_path), language=language)

    text = " ".join([seg.text for seg in segments])

    result = {"text": text, "language": info.language}

    # Save transcript
    output_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to: {output_path}")
    print(f"\nTranscript:\n{text}")

    return result


if __name__ == "__main__":
    audio_path = CLEAN_AUDIO_DIR / "karina_clean.wav" if len(sys.argv) < 2 else Path(sys.argv[1])

    language = sys.argv[2] if len(sys.argv) > 2 else "ko"
    transcribe_audio(audio_path, language)
