#!/usr/bin/env python3
"""
Automated pipeline for Karina voice notification generation.
Cross-platform: Linux (CUDA) / Mac (MPS) / CPU

Usage:
    python pipeline.py [--skip-download] [url]
"""

import argparse
import subprocess
import sys
import json
import os
import shutil
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich import box
import readchar

from device_utils import detect_device, print_device_info, DeviceType, DeviceInfo

# Setup
console = Console()
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_AUDIO_DIR = OUTPUT_DIR / "raw"
CLEAN_AUDIO_DIR = OUTPUT_DIR / "clean"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
NOTIFICATIONS_DIR = OUTPUT_DIR / "notifications"
MODELS_DIR = PROJECT_ROOT / "models"

# Notification lines to generate
NOTIFICATION_LINES = {
    "permission_prompt": [
        {"text": "잠깐만요! 이거 실행해도 괜찮을까요? 허락해주세요~", "filename": "permission_prompt_1.wav"},
        {"text": "잠시만요, 이 작업을 하려면 허락이 필요해요~", "filename": "permission_prompt_2.wav"},
    ],
    "idle_prompt": [
        {"text": "다 끝났어요! 결과 한번 확인해주세요~", "filename": "idle_prompt_1.wav"},
        {"text": "작업이 완료되었어요, 한번 봐주시겠어요?", "filename": "idle_prompt_2.wav"},
    ],
    "auth_success": [
        {"text": "인증이 완료되었어요! 도와주셔서 정말 고마워요~", "filename": "auth_success_1.wav"},
    ],
    "elicitation_dialog": [
        {"text": "여기에 입력이 필요해요! 작성해주시겠어요?", "filename": "elicitation_dialog_1.wav"},
    ],
}


# ============== Interactive Menu ==============

class InteractiveMenu:
    """Beautiful interactive menu with arrow key navigation."""

    def __init__(self, title: str, options: list[dict], subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.options = options
        self.selected = 0

    def _render(self) -> Panel:
        """Render the menu."""
        menu_text = Text()
        
        for i, opt in enumerate(self.options):
            if i == self.selected:
                menu_text.append("  ▸ ", style="bold cyan")
                menu_text.append(f"{opt['label']}\n", style="bold white on blue")
                if opt.get('desc'):
                    menu_text.append(f"    {opt['desc']}\n", style="dim cyan")
            else:
                menu_text.append("    ", style="dim")
                menu_text.append(f"{opt['label']}\n", style="white")
                if opt.get('desc'):
                    menu_text.append(f"    {opt['desc']}\n", style="dim")
            
            if i < len(self.options) - 1:
                menu_text.append("\n")

        footer = Text("\n  ↑↓ 이동  •  Enter 선택  •  q 종료", style="dim")
        menu_text.append(footer)

        return Panel(
            Align.left(menu_text),
            title=f"[bold magenta]✨ {self.title}[/bold magenta]",
            subtitle=f"[dim]{self.subtitle}[/dim]" if self.subtitle else None,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def run(self) -> int | None:
        """Run the interactive menu. Returns selected index or None if cancelled."""
        with Live(self._render(), console=console, refresh_per_second=30, transient=True) as live:
            while True:
                key = readchar.readkey()
                
                if key == readchar.key.UP:
                    self.selected = (self.selected - 1) % len(self.options)
                elif key == readchar.key.DOWN:
                    self.selected = (self.selected + 1) % len(self.options)
                elif key == readchar.key.ENTER:
                    return self.selected
                elif key.lower() == 'q' or key == readchar.key.ESCAPE:
                    return None
                
                live.update(self._render())


def show_main_menu() -> str | None:
    """Show main menu and return selected action."""
    options = [
        {"label": "🚀 전체 파이프라인 실행", "desc": "다운로드 → 분할 → 전사 → TTS 생성", "action": "full"},
        {"label": "📥 음성 다운로드 & 추출", "desc": "YouTube에서 음성 다운로드 후 세그먼트 분할", "action": "download"},
        {"label": "📝 전사(Transcribe)부터 시작", "desc": "기존 오디오로 전사 → TTS 생성", "action": "transcribe"},
        {"label": "🎤 음성 생성만", "desc": "기존 전사 결과로 TTS 음성만 생성", "action": "generate"},
        {"label": "❌ 종료", "desc": "", "action": "exit"},
    ]

    menu = InteractiveMenu(
        title="Karina Voice Generator",
        subtitle="aespa 카리나 음성으로 Claude Code 알림음 생성",
        options=options
    )

    result = menu.run()
    if result is None:
        return None
    return options[result]["action"]


def show_segment_menu(segments: list[Path]) -> int | None:
    """Show segment selection menu."""
    options = [{"label": f"🎵 {seg.name}", "desc": ""} for seg in segments]
    options.append({"label": "❌ 취소", "desc": ""})

    menu = InteractiveMenu(
        title="세그먼트 선택",
        subtitle="깨끗한 음성 구간을 선택하세요 (afplay로 미리듣기 가능)",
        options=options
    )

    result = menu.run()
    if result is None or result == len(segments):
        return None
    return result


# ============== Device Check ==============

def check_device() -> DeviceInfo:
    """Check and display device information."""
    console.print(Panel("[bold]Device Environment Check[/bold]", style="blue"))
    device_info = detect_device()
    print_device_info(device_info)
    if not device_info.is_gpu:
        logger.warning("No GPU detected, using CPU (will be slow)")
    console.print()
    return device_info


# ============== Pipeline Steps ==============

def download_audio(url: str, output_name: str = "karina_sample") -> Path:
    """Download audio from YouTube."""
    console.print(Panel("[bold]Step 1: Download Audio from YouTube[/bold]", style="blue"))

    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_AUDIO_DIR / f"{output_name}.%(ext)s"

    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0", "-o", str(output_path), url]
    logger.info(f"Downloading: {url}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
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

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
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


def select_segment(segments: list[Path]) -> Path | None:
    """Let user select a segment using interactive menu."""
    console.print(Panel("[bold]Step 3: Select Clean Voice Segment[/bold]", style="blue"))

    # Show segments info
    table = Table(title="Available Segments", box=box.ROUNDED)
    table.add_column("Index", style="cyan")
    table.add_column("Filename", style="green")
    for i, seg in enumerate(segments):
        table.add_row(str(i), seg.name)
    console.print(table)
    console.print("\n[dim]💡 Tip: 터미널에서 'afplay <path>' (Mac) 또는 'aplay <path>' (Linux)로 미리듣기[/dim]\n")

    # Interactive selection
    idx = show_segment_menu(segments)
    if idx is None:
        return None

    selected = segments[idx]
    logger.success(f"Selected: {selected.name}")

    final_path = CLEAN_AUDIO_DIR / "karina_clean.wav"
    shutil.copy(selected, final_path)
    logger.info(f"Copied to: {final_path}")
    console.print()
    return final_path


def transcribe_audio(audio_path: Path, device_info: DeviceInfo) -> str:
    """Transcribe audio using the appropriate backend for the platform."""
    console.print(Panel(f"[bold]Step 4: Transcribe Audio ({device_info.whisper_backend})[/bold]", style="blue"))

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_info.whisper_backend == "faster-whisper":
        from faster_whisper import WhisperModel
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Loading Whisper large-v3 model...", total=None)
            compute_type = "float16" if device_info.dtype == torch.float16 else "float32"
            model = WhisperModel("large-v3", device=device_info.device_type.value, compute_type=compute_type)

        logger.info(f"Transcribing: {audio_path}")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Transcribing...", total=None)
            segments, info = model.transcribe(str(audio_path), language="ko")
            text = " ".join([seg.text for seg in segments])
    else:
        import mlx_whisper
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Loading Whisper large-v3 model (MLX)...", total=None)
        logger.info(f"Transcribing: {audio_path}")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Transcribing...", total=None)
            result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo="mlx-community/whisper-large-v3-mlx", language="ko")
            text = result["text"]
            info = type("Info", (), {"language": "ko"})()

    result_data = {"text": text, "language": info.language}
    output_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

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
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Downloading model...", total=None)
            snapshot_download(repo_id=model_name, local_dir=str(local_dir))
    else:
        logger.info(f"Model already exists: {local_dir}")

    tokenizer_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    tokenizer_dir = MODELS_DIR / "Qwen3-TTS-Tokenizer-12Hz"

    if not tokenizer_dir.exists():
        logger.info(f"Downloading {tokenizer_name}...")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Downloading tokenizer...", total=None)
            snapshot_download(repo_id=tokenizer_name, local_dir=str(tokenizer_dir))
    else:
        logger.info(f"Tokenizer already exists: {tokenizer_dir}")

    logger.success("Model setup complete!")
    console.print()
    return local_dir


def post_process_audio(audio: np.ndarray, sr: int, silence_ms: int = 300) -> tuple[np.ndarray, int]:
    """Post-process audio: add silence at beginning."""
    silence_samples = int(sr * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    return np.concatenate([silence, audio]), sr


def generate_notifications(ref_audio_path: Path, ref_text: str, model_path: Path, device_info: DeviceInfo):
    """Generate all notification voice lines using voice cloning."""
    console.print(Panel(f"[bold]Step 6: Generate Notification Voice Lines ({device_info.device_type.value.upper()})[/bold]", style="blue"))

    from qwen_tts import Qwen3TTSModel

    NOTIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(f"Loading Qwen3-TTS 1.7B model on {device_info.device_type.value.upper()}...", total=None)
        model = Qwen3TTSModel.from_pretrained(str(model_path), dtype=device_info.dtype, attn_implementation=device_info.attn_implementation, device_map=device_info.torch_device)

    if device_info.device_type == DeviceType.MPS:
        torch.mps.synchronize()

    total = sum(len(lines) for lines in NOTIFICATION_LINES.values())
    logger.info(f"Generating {total} notification voice lines...")
    logger.info(f"Reference audio: {ref_audio_path}")
    logger.info(f"Reference text: {ref_text[:50]}...")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Generating notifications...", total=total)

        for notification_type, lines in NOTIFICATION_LINES.items():
            type_dir = NOTIFICATIONS_DIR / notification_type
            type_dir.mkdir(parents=True, exist_ok=True)

            for line in lines:
                output_path = type_dir / line["filename"]
                wavs, sr = model.generate_voice_clone(text=line["text"], ref_audio=str(ref_audio_path), ref_text=ref_text, language="korean", non_streaming_mode=True)

                if device_info.device_type == DeviceType.MPS:
                    torch.mps.synchronize()

                processed_audio, sr = post_process_audio(wavs[0], sr, silence_ms=300)
                sf.write(str(output_path), processed_audio, sr)
                progress.advance(task)

    logger.success(f"All notifications generated in: {NOTIFICATIONS_DIR}")

    table = Table(title="Generated Files", box=box.ROUNDED)
    table.add_column("Type", style="cyan")
    table.add_column("File", style="green")
    for notification_type in NOTIFICATION_LINES.keys():
        type_dir = OUTPUT_DIR / notification_type
        for f in type_dir.glob("*.wav"):
            table.add_row(notification_type, f.name)
    console.print(table)
    console.print()


def show_completion():
    """Show completion message."""
    console.print(Panel.fit(
        "[bold green]✨ Pipeline Complete![/bold green]\n\n"
        f"Generated notifications are in: [cyan]{NOTIFICATIONS_DIR}[/cyan]\n\n"
        "[dim]Next steps:[/dim]\n"
        "1. Review generated audio files\n"
        "2. Copy best ones to ~/.claude/sounds/\n"
        "3. Configure Claude Code notification hooks",
        border_style="green"
    ))


# ============== Main Entry Points ==============

DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=r96zEiIHVf4"


def run_full_pipeline(url: str, device_info: DeviceInfo):
    """Run the complete pipeline."""
    audio_file = download_audio(url)
    segments = split_audio(audio_file)
    selected_segment = select_segment(segments)
    if selected_segment is None:
        logger.info("Cancelled")
        return
    transcript = transcribe_audio(selected_segment, device_info)
    model_path = setup_tts_model()
    generate_notifications(selected_segment, transcript, model_path, device_info)
    show_completion()


def run_download_only(url: str):
    """Download and split audio only."""
    audio_file = download_audio(url)
    segments = split_audio(audio_file)
    selected = select_segment(segments)
    if selected:
        console.print(Panel.fit(f"[bold green]✅ 오디오 준비 완료![/bold green]\n\nSelected: [cyan]{selected}[/cyan]", border_style="green"))


def run_from_transcribe(device_info: DeviceInfo):
    """Run from transcribe step using existing audio."""
    clean_audio = CLEAN_AUDIO_DIR / "karina_clean.wav"
    if not clean_audio.exists():
        existing = list(CLEAN_AUDIO_DIR.glob("segment_*.wav"))
        if not existing:
            logger.error(f"No audio files found in {CLEAN_AUDIO_DIR}")
            logger.error("Please run 'Download & Extract' first")
            return
        selected = select_segment(existing)
        if selected is None:
            return
        clean_audio = selected

    transcript = transcribe_audio(clean_audio, device_info)
    model_path = setup_tts_model()
    generate_notifications(clean_audio, transcript, model_path, device_info)
    show_completion()


def run_generate_only(device_info: DeviceInfo):
    """Generate notifications using existing transcript."""
    clean_audio = CLEAN_AUDIO_DIR / "karina_clean.wav"
    if not clean_audio.exists():
        logger.error(f"No clean audio found: {clean_audio}")
        logger.error("Please run 'Transcribe' first")
        return

    transcript_files = list(TRANSCRIPTS_DIR.glob("*_transcript.json"))
    if not transcript_files:
        logger.error(f"No transcript found in {TRANSCRIPTS_DIR}")
        logger.error("Please run 'Transcribe' first")
        return

    with open(transcript_files[0], "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    transcript = transcript_data["text"]

    logger.info(f"Using transcript: {transcript[:50]}...")
    model_path = setup_tts_model()
    generate_notifications(clean_audio, transcript, model_path, device_info)
    show_completion()


def main():
    parser = argparse.ArgumentParser(description="Karina Voice Notification Generator")
    parser.add_argument("url", nargs="?", default=DEFAULT_YOUTUBE_URL, help="YouTube URL")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing audio")
    parser.add_argument("--no-menu", action="store_true", help="Skip menu, run full pipeline")
    args = parser.parse_args()

    # Show banner
    console.print(Panel.fit(
        "[bold magenta]🎤 Karina Voice Notification Generator[/bold magenta]\n"
        "[dim]Cross-platform (CUDA / MPS / CPU)[/dim]",
        border_style="magenta"
    ))
    console.print()

    # Check device
    device_info = check_device()

    # If --no-menu or --skip-download, run full pipeline directly
    if args.no_menu or args.skip_download:
        if args.skip_download:
            run_from_transcribe(device_info)
        else:
            run_full_pipeline(args.url, device_info)
        return

    # Show interactive menu
    while True:
        action = show_main_menu()

        if action is None or action == "exit":
            console.print("\n[dim]👋 Bye![/dim]")
            break
        elif action == "full":
            url = console.input("\n[bold yellow]YouTube URL[/bold yellow] (Enter for default): ").strip() or args.url
            run_full_pipeline(url, device_info)
        elif action == "download":
            url = console.input("\n[bold yellow]YouTube URL[/bold yellow] (Enter for default): ").strip() or args.url
            run_download_only(url)
        elif action == "transcribe":
            run_from_transcribe(device_info)
        elif action == "generate":
            run_generate_only(device_info)

        console.print("\n")


if __name__ == "__main__":
    main()
