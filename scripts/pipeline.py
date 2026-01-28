#!/usr/bin/env python3
"""
Automated pipeline for Karina voice notification generation.
GPU environment (A100 x 4) optimized.

Usage:
    python pipeline.py <youtube_url>
"""

import subprocess
import sys
import json
import os
from pathlib import Path

import torch
import soundfile as sf
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
RAW_AUDIO_DIR = ASSETS_DIR / "raw"
CLEAN_AUDIO_DIR = ASSETS_DIR / "clean"
TRANSCRIPTS_DIR = ASSETS_DIR / "transcripts"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "notifications"

# Notification lines to generate
NOTIFICATION_LINES = {
    "permission_prompt": [
        {"text": "오빠, 잠깐! 이거 해도 돼?", "filename": "permission_prompt_1.wav"},
        {"text": "허락이 필요해요~", "filename": "permission_prompt_2.wav"},
    ],
    "idle_prompt": [
        {"text": "오빠, 다 했어! 확인해줘~", "filename": "idle_prompt_1.wav"},
        {"text": "끝났어요, 봐주세요!", "filename": "idle_prompt_2.wav"},
    ],
    "auth_success": [
        {"text": "인증 완료! 고마워요~", "filename": "auth_success_1.wav"},
    ],
    "elicitation_dialog": [
        {"text": "여기 입력이 필요해요!", "filename": "elicitation_dialog_1.wav"},
    ],
}


def check_gpu():
    """Check GPU availability."""
    print("=" * 60)
    print("GPU Environment Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    print(f"CUDA available: True")
    print(f"GPU count: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    print()
    return gpu_count


def download_audio(url: str, output_name: str = "karina_sample") -> Path:
    """Download audio from YouTube."""
    print("=" * 60)
    print("Step 1: Download Audio from YouTube")
    print("=" * 60)

    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_AUDIO_DIR / f"{output_name}.%(ext)s"

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        url
    ]

    print(f"Downloading: {url}")
    subprocess.run(cmd, check=True)

    downloaded = list(RAW_AUDIO_DIR.glob(f"{output_name}.*"))
    if downloaded:
        print(f"Downloaded: {downloaded[0]}")
        print()
        return downloaded[0]
    raise FileNotFoundError("Download failed")


def split_audio(input_file: Path, segment_duration: int = 15) -> list[Path]:
    """Split audio into segments."""
    print("=" * 60)
    print("Step 2: Split Audio into Segments")
    print("=" * 60)

    from pydub import AudioSegment

    CLEAN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_file}")
    audio = AudioSegment.from_file(input_file)
    total_duration = len(audio) / 1000  # in seconds

    print(f"Total duration: {total_duration:.1f} seconds")

    segments = []
    segment_ms = segment_duration * 1000

    # Create segments every 30 seconds
    for i, start_ms in enumerate(range(0, len(audio), 30000)):
        end_ms = min(start_ms + segment_ms, len(audio))
        if end_ms - start_ms < 5000:  # Skip segments less than 5 seconds
            continue

        segment = audio[start_ms:end_ms]
        segment = segment.normalize()
        segment = segment.set_frame_rate(16000).set_channels(1)

        start_sec = start_ms // 1000
        end_sec = end_ms // 1000
        filename = f"segment_{start_sec}s_{end_sec}s.wav"
        output_path = CLEAN_AUDIO_DIR / filename

        segment.export(output_path, format="wav")
        segments.append(output_path)

        print(f"  Created: {filename} ({(end_ms - start_ms) / 1000:.1f}s)")

    print(f"\nTotal segments: {len(segments)}")
    print()
    return segments


def select_segment(segments: list[Path]) -> Path:
    """Let user select a segment."""
    print("=" * 60)
    print("Step 3: Select Clean Voice Segment")
    print("=" * 60)

    print("\nAvailable segments:")
    for i, seg in enumerate(segments):
        print(f"  [{i}] {seg.name}")

    print("\nPlay segments with: afplay <path>")
    print("Or on Linux: aplay <path> or paplay <path>")
    print()

    while True:
        try:
            choice = input("Enter segment number to use: ").strip()
            idx = int(choice)
            if 0 <= idx < len(segments):
                selected = segments[idx]
                print(f"\nSelected: {selected.name}")

                # Copy to karina_clean.wav
                final_path = CLEAN_AUDIO_DIR / "karina_clean.wav"
                import shutil
                shutil.copy(selected, final_path)
                print(f"Copied to: {final_path}")
                print()
                return final_path
            else:
                print(f"Please enter a number between 0 and {len(segments) - 1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nAborted")
            sys.exit(1)


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio using faster-whisper (GPU)."""
    print("=" * 60)
    print("Step 4: Transcribe Audio (GPU)")
    print("=" * 60)

    from faster_whisper import WhisperModel

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper large-v3 model on GPU...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    print(f"Transcribing: {audio_path}")
    segments, info = model.transcribe(str(audio_path), language="ko")

    text = " ".join([seg.text for seg in segments])

    # Save transcript
    result = {"text": text, "language": info.language}
    output_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to: {output_path}")
    print(f"\nTranscript:\n{text}")
    print()

    return text


def setup_tts_model():
    """Download and setup Qwen3-TTS 1.7B model."""
    print("=" * 60)
    print("Step 5: Setup Qwen3-TTS 1.7B Model")
    print("=" * 60)

    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    local_dir = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"

    if not local_dir.exists():
        print(f"Downloading {model_name}...")
        snapshot_download(repo_id=model_name, local_dir=str(local_dir))
    else:
        print(f"Model already exists: {local_dir}")

    # Download tokenizer
    tokenizer_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    tokenizer_dir = MODELS_DIR / "Qwen3-TTS-Tokenizer-12Hz"

    if not tokenizer_dir.exists():
        print(f"Downloading {tokenizer_name}...")
        snapshot_download(repo_id=tokenizer_name, local_dir=str(tokenizer_dir))
    else:
        print(f"Tokenizer already exists: {tokenizer_dir}")

    print("Model setup complete!")
    print()
    return local_dir


def generate_notifications(ref_audio_path: Path, ref_text: str, model_path: Path):
    """Generate all notification voice lines using voice cloning."""
    print("=" * 60)
    print("Step 6: Generate Notification Voice Lines")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen3-TTS 1.7B model on GPU...")
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        dtype=torch.float16,  # Use float16 for GPU
        attn_implementation="flash_attention_2",  # Use FlashAttention for speed
        device_map="cuda:0",
    )

    total = sum(len(lines) for lines in NOTIFICATION_LINES.values())

    print(f"\nGenerating {total} notification voice lines...")
    print(f"Reference audio: {ref_audio_path}")
    print(f"Reference text: {ref_text[:50]}...")
    print()

    with tqdm(total=total, desc="Generating") as pbar:
        for notification_type, lines in NOTIFICATION_LINES.items():
            type_dir = OUTPUT_DIR / notification_type
            type_dir.mkdir(parents=True, exist_ok=True)

            for line in lines:
                output_path = type_dir / line["filename"]

                wavs, sr = model.generate_voice_clone(
                    text=line["text"],
                    ref_audio=str(ref_audio_path),
                    ref_text=ref_text,
                    language="korean",
                    non_streaming_mode=True,
                )

                sf.write(str(output_path), wavs[0], sr)
                pbar.update(1)

    print(f"\nAll notifications generated in: {OUTPUT_DIR}")
    print()

    # List generated files
    print("Generated files:")
    for notification_type in NOTIFICATION_LINES.keys():
        type_dir = OUTPUT_DIR / notification_type
        for f in type_dir.glob("*.wav"):
            print(f"  {f}")
    print()


DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=r96zEiIHVf4"


def main():
    if len(sys.argv) < 2:
        youtube_url = DEFAULT_YOUTUBE_URL
        print(f"Using default YouTube URL: {youtube_url}")
    else:
        youtube_url = sys.argv[1]

    print("\n" + "=" * 60)
    print("Karina Voice Notification Generator")
    print("GPU Environment (A100 x 4)")
    print("=" * 60 + "\n")

    # Check GPU
    check_gpu()

    # Step 1: Download audio
    audio_file = download_audio(youtube_url)

    # Step 2: Split into segments
    segments = split_audio(audio_file)

    # Step 3: User selects segment
    selected_segment = select_segment(segments)

    # Step 4: Transcribe
    transcript = transcribe_audio(selected_segment)

    # Step 5: Setup TTS model
    model_path = setup_tts_model()

    # Step 6: Generate notifications
    generate_notifications(selected_segment, transcript, model_path)

    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nGenerated notifications are in: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review generated audio files")
    print("2. Copy best ones to ~/.claude/hooks/karina-notification/sounds/")
    print("3. Configure Claude Code notification hooks")


if __name__ == "__main__":
    main()
