#!/usr/bin/env python3
"""
Automated pipeline for Karina voice notification generation.
GPU environment (A100 x 4) optimized.

Usage:
    python pipeline.py <youtube_url>
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from scipy import signal
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Setup
console = Console()
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

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
        {"text": "잠깐요! 이거 해도 될까요?", "filename": "permission_prompt_1.wav"},
        {"text": "허락이 필요해요~", "filename": "permission_prompt_2.wav"},
    ],
    "idle_prompt": [
        {"text": "다 했어요! 확인해주세요~", "filename": "idle_prompt_1.wav"},
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
    console.print(Panel("[bold]GPU Environment Check[/bold]", style="blue"))

    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    logger.info(f"CUDA available: True")
    logger.info(f"GPU count: {gpu_count}")

    table = Table(title="GPU Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Memory", style="yellow")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        table.add_row(str(i), props.name, f"{props.total_memory / 1024**3:.1f} GB")

    console.print(table)
    console.print()
    return gpu_count


def download_audio(url: str, output_name: str = "karina_sample") -> Path:
    """Download audio from YouTube."""
    console.print(Panel("[bold]Step 1: Download Audio from YouTube[/bold]", style="blue"))

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

    logger.info(f"Downloading: {url}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Downloading...", total=None)
        subprocess.run(cmd, check=True)

    downloaded = list(RAW_AUDIO_DIR.glob(f"{output_name}.*"))
    if downloaded:
        logger.success(f"Downloaded: {downloaded[0]}")
        console.print()
        return downloaded[0]
    raise FileNotFoundError("Download failed")


def split_audio(input_file: Path, segment_duration: int = 15) -> list[Path]:
    """Split audio into segments."""
    console.print(Panel("[bold]Step 2: Split Audio into Segments[/bold]", style="blue"))

    from pydub import AudioSegment

    CLEAN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading: {input_file}")
    audio = AudioSegment.from_file(input_file)
    total_duration = len(audio) / 1000

    logger.info(f"Total duration: {total_duration:.1f} seconds")

    segments = []
    segment_ms = segment_duration * 1000
    segment_starts = list(range(0, len(audio), 30000))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Splitting audio...", total=len(segment_starts))

        for start_ms in segment_starts:
            end_ms = min(start_ms + segment_ms, len(audio))
            if end_ms - start_ms < 5000:
                progress.advance(task)
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
            progress.advance(task)

    logger.success(f"Created {len(segments)} segments")
    console.print()
    return segments


def select_segment(segments: list[Path]) -> Path:
    """Let user select a segment."""
    console.print(Panel("[bold]Step 3: Select Clean Voice Segment[/bold]", style="blue"))

    table = Table(title="Available Segments")
    table.add_column("Index", style="cyan")
    table.add_column("Filename", style="green")

    for i, seg in enumerate(segments):
        table.add_row(str(i), seg.name)

    console.print(table)
    console.print("\n[dim]Play segments with: aplay <path> or paplay <path>[/dim]\n")

    while True:
        try:
            choice = console.input("[bold yellow]Enter segment number to use:[/bold yellow] ").strip()
            idx = int(choice)
            if 0 <= idx < len(segments):
                selected = segments[idx]
                logger.success(f"Selected: {selected.name}")

                final_path = CLEAN_AUDIO_DIR / "karina_clean.wav"
                import shutil
                shutil.copy(selected, final_path)
                logger.info(f"Copied to: {final_path}")
                console.print()
                return final_path
            else:
                logger.warning(f"Please enter a number between 0 and {len(segments) - 1}")
        except ValueError:
            logger.warning("Please enter a valid number")
        except KeyboardInterrupt:
            logger.info("Aborted")
            sys.exit(1)


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio using faster-whisper (GPU)."""
    console.print(Panel("[bold]Step 4: Transcribe Audio (GPU)[/bold]", style="blue"))

    from faster_whisper import WhisperModel

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Loading Whisper large-v3 model...", total=None)
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    logger.info(f"Transcribing: {audio_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Transcribing...", total=None)
        segments, info = model.transcribe(str(audio_path), language="ko")
        text = " ".join([seg.text for seg in segments])

    result = {"text": text, "language": info.language}
    output_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.success(f"Transcript saved to: {output_path}")
    console.print(Panel(f"[italic]{text}[/italic]", title="Transcript", style="green"))
    console.print()

    return text


def setup_tts_model():
    """Download and setup Qwen3-TTS 1.7B model."""
    console.print(Panel("[bold]Step 5: Setup Qwen3-TTS 1.7B Model[/bold]", style="blue"))

    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    local_dir = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"

    if not local_dir.exists():
        logger.info(f"Downloading {model_name}...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Downloading model...", total=None)
            snapshot_download(repo_id=model_name, local_dir=str(local_dir))
    else:
        logger.info(f"Model already exists: {local_dir}")

    tokenizer_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    tokenizer_dir = MODELS_DIR / "Qwen3-TTS-Tokenizer-12Hz"

    if not tokenizer_dir.exists():
        logger.info(f"Downloading {tokenizer_name}...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Downloading tokenizer...", total=None)
            snapshot_download(repo_id=tokenizer_name, local_dir=str(tokenizer_dir))
    else:
        logger.info(f"Tokenizer already exists: {tokenizer_dir}")

    logger.success("Model setup complete!")
    console.print()
    return local_dir


def post_process_audio(audio: np.ndarray, sr: int, silence_ms: int = 300, speed: float = 0.67) -> tuple[np.ndarray, int]:
    """Post-process audio: add silence at beginning and slow down.

    Args:
        audio: Audio waveform
        sr: Sample rate
        silence_ms: Silence to add at beginning (milliseconds)
        speed: Speed factor (0.67 = 1.5x slower)

    Returns:
        Processed audio and sample rate
    """
    # Add silence at beginning
    silence_samples = int(sr * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    audio_with_silence = np.concatenate([silence, audio])

    # Slow down audio by resampling
    if speed != 1.0:
        # Resample to slow down (lower speed = longer audio)
        num_samples = int(len(audio_with_silence) / speed)
        audio_slowed = signal.resample(audio_with_silence, num_samples)
        return audio_slowed.astype(np.float32), sr

    return audio_with_silence, sr


def generate_notifications(ref_audio_path: Path, ref_text: str, model_path: Path):
    """Generate all notification voice lines using voice cloning."""
    console.print(Panel("[bold]Step 6: Generate Notification Voice Lines[/bold]", style="blue"))

    from qwen_tts import Qwen3TTSModel

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Loading Qwen3-TTS 1.7B model on GPU...", total=None)

        # Check if flash-attn is available
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Using FlashAttention2")
        except ImportError:
            attn_impl = "sdpa"  # PyTorch native scaled dot product attention
            logger.warning("flash-attn not installed, using SDPA (slower)")

        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            dtype=torch.float16,
            attn_implementation=attn_impl,
            device_map="cuda:0",
        )

    total = sum(len(lines) for lines in NOTIFICATION_LINES.values())

    logger.info(f"Generating {total} notification voice lines...")
    logger.info(f"Reference audio: {ref_audio_path}")
    logger.info(f"Reference text: {ref_text[:50]}...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating notifications...", total=total)

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

                # Post-process: add 300ms silence, slow down to 1.5x duration
                processed_audio, sr = post_process_audio(wavs[0], sr, silence_ms=300, speed=0.67)
                sf.write(str(output_path), processed_audio, sr)
                progress.advance(task)

    logger.success(f"All notifications generated in: {OUTPUT_DIR}")

    table = Table(title="Generated Files")
    table.add_column("Type", style="cyan")
    table.add_column("File", style="green")

    for notification_type in NOTIFICATION_LINES.keys():
        type_dir = OUTPUT_DIR / notification_type
        for f in type_dir.glob("*.wav"):
            table.add_row(notification_type, f.name)

    console.print(table)
    console.print()


DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=r96zEiIHVf4"


def main():
    parser = argparse.ArgumentParser(description="Karina Voice Notification Generator")
    parser.add_argument("url", nargs="?", default=DEFAULT_YOUTUBE_URL, help="YouTube URL")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing audio in assets/raw/")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold magenta]Karina Voice Notification Generator[/bold magenta]\n"
        "[dim]GPU Environment (A100 x 4)[/dim]",
        border_style="magenta"
    ))
    console.print()

    # Check GPU
    check_gpu()

    # Step 1: Download audio or use existing
    if args.skip_download:
        console.print(Panel("[bold]Step 1: Using Existing Audio (--skip-download)[/bold]", style="blue"))
        existing = list(RAW_AUDIO_DIR.glob("*.wav")) + list(RAW_AUDIO_DIR.glob("*.mp3"))
        if not existing:
            logger.error(f"No audio files found in {RAW_AUDIO_DIR}")
            logger.error("Please upload audio file to assets/raw/ first")
            sys.exit(1)
        audio_file = existing[0]
        logger.info(f"Using: {audio_file}")
        console.print()
    else:
        audio_file = download_audio(args.url)

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

    console.print(Panel.fit(
        "[bold green]Pipeline Complete![/bold green]\n\n"
        f"Generated notifications are in: [cyan]{OUTPUT_DIR}[/cyan]\n\n"
        "[dim]Next steps:[/dim]\n"
        "1. Review generated audio files\n"
        "2. Copy best ones to ~/.claude/sounds/\n"
        "3. Configure Claude Code notification hooks",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
