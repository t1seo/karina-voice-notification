#!/usr/bin/env python3
"""Generate notification voice lines using Karina's cloned voice (GPU optimized)."""

import json
import warnings
from pathlib import Path

import soundfile as sf
import torch
import transformers
from tqdm import tqdm

# Suppress transformers warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*pad_token_id.*")

# Post-processing flag
ENABLE_POST_PROCESSING = True

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "notifications"
NOTIFICATION_LINES_FILE = PROJECT_ROOT / "notification_lines.json"


def load_model():
    """Load Qwen3-TTS 1.7B model on GPU."""
    from qwen_tts import Qwen3TTSModel

    model_path = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"

    print("Loading Qwen3-TTS 1.7B model on GPU...")
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    return model


def generate_cloned_voice(
    model,
    text: str,
    ref_audio_path: Path,
    ref_text: str,
    output_path: Path,
    post_process: bool = True,
    language: str = "korean",
):
    """Generate speech with cloned voice."""
    wavs, sr = model.generate_voice_clone(
        text=text,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        language=language,
        non_streaming_mode=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sr)

    # Apply post-processing if enabled
    if post_process and ENABLE_POST_PROCESSING:
        try:
            from post_process import post_process_file

            post_process_file(
                output_path,
                denoise=True,
                eq=True,
                dynamics=True,
                loudness_normalize=True,
                target_lufs=-14.0,
            )
        except ImportError:
            pass  # Post-processing dependencies not installed

    return output_path


def generate_all_notifications(ref_audio_path: Path, ref_text: str):
    """Generate all notification voice lines."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load notification lines
    with open(NOTIFICATION_LINES_FILE, encoding="utf-8") as f:
        notification_lines = json.load(f)

    # Load model
    model = load_model()

    # Generate each notification
    total = sum(len(lines) for lines in notification_lines.values())

    with tqdm(total=total, desc="Generating notifications") as pbar:
        for notification_type, lines in notification_lines.items():
            type_dir = OUTPUT_DIR / notification_type
            type_dir.mkdir(parents=True, exist_ok=True)

            for line in lines:
                output_path = type_dir / line["filename"]

                print(f"\nGenerating: {line['text']}")
                generate_cloned_voice(
                    model=model,
                    text=line["text"],
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    output_path=output_path,
                )
                print(f"Saved: {output_path}")
                pbar.update(1)

    print(f"\nAll notifications generated in: {OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == "__main__":
    import sys

    # Load reference audio and transcript
    ref_audio = ASSETS_DIR / "clean" / "karina_clean.wav"
    transcript_file = ASSETS_DIR / "transcripts" / "karina_clean_transcript.json"

    if not ref_audio.exists():
        print(f"Error: Reference audio not found: {ref_audio}")
        sys.exit(1)

    if not transcript_file.exists():
        print(f"Error: Transcript not found: {transcript_file}")
        sys.exit(1)

    with open(transcript_file, encoding="utf-8") as f:
        transcript_data = json.load(f)
    ref_text = transcript_data["text"]

    print(f"Reference audio: {ref_audio}")
    print(f"Reference text: {ref_text}")

    generate_all_notifications(ref_audio, ref_text)
