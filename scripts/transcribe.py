#!/usr/bin/env python3
"""Transcribe audio using Lightning Whisper MLX (optimized for Apple Silicon)."""

from pathlib import Path
import json
import sys

from lightning_whisper_mlx import LightningWhisperMLX

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CLEAN_AUDIO_DIR = ASSETS_DIR / "clean"
TRANSCRIPTS_DIR = ASSETS_DIR / "transcripts"

def transcribe_audio(audio_path: Path, language: str = "ko") -> dict:
    """Transcribe audio file using Lightning Whisper MLX.

    Args:
        audio_path: Path to audio file
        language: Language code (ko for Korean)

    Returns:
        Transcription result dict with 'text' key
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper model (large-v3)...")
    whisper = LightningWhisperMLX(model="large-v3", batch_size=12, quant=None)

    print(f"Transcribing: {audio_path}")
    result = whisper.transcribe(str(audio_path), language=language)

    # Save transcript
    output_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to: {output_path}")
    print(f"\nTranscript:\n{result['text']}")

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        audio_path = CLEAN_AUDIO_DIR / "karina_clean.wav"
    else:
        audio_path = Path(sys.argv[1])

    language = sys.argv[2] if len(sys.argv) > 2 else "ko"
    transcribe_audio(audio_path, language)
