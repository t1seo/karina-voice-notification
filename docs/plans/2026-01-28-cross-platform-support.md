# Cross-Platform Support (Linux GPU + Mac Apple Silicon) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 기존 Linux GPU 환경에 더해 Mac Apple Silicon (64GB RAM)에서도 TTS 파이프라인이 동작하도록 수정

**Architecture:** 플랫폼 감지 유틸리티를 통해 CUDA/MPS/CPU 중 최적 디바이스 자동 선택. Whisper는 Mac에서 mlx-whisper 사용, Qwen3-TTS는 MPS 백엔드로 float32 dtype 사용.

**Tech Stack:** PyTorch (CUDA/MPS), mlx-whisper (Mac), faster-whisper (Linux), Qwen3-TTS, pixi (cross-platform)

**Sources:**
- [Qwen3-TTS Mac Test Report](https://github.com/tumf/2026-01-23-Qwen3-TTS)
- [ComfyUI-Qwen3-TTS MPS Support](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS)
- [insanely-fast-whisper MPS](https://github.com/Vaibhavs10/insanely-fast-whisper)

---

### Task 1: pixi.toml에 Mac 플랫폼 추가

**Files:**
- Modify: `pixi.toml`

**Step 1: 플랫폼 및 feature 기반 의존성 구조로 변경**

```toml
[workspace]
name = "karina-tts-notification"
version = "0.1.0"
description = "Karina voice notification for Claude Code"
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
# Common dependencies
python = "3.12.*"
ffmpeg = ">=6.0"
yt-dlp = ">=2024.1.0"
sox = ">=14.4.2"
pip = ">=24.0"

[feature.linux.dependencies]
# Linux CUDA dependencies
cuda-toolkit = ">=12.0"
cudnn = ">=8.0"

[feature.linux]
platforms = ["linux-64"]

[feature.mac]
platforms = ["osx-arm64"]

[pypi-dependencies]
# Common Python packages
pydub = ">=0.25.1"
torch = ">=2.0.0"
soundfile = ">=0.12.0"
huggingface-hub = ">=0.20.0"
scipy = ">=1.10.0"
loguru = ">=0.7.0"
rich = ">=13.0.0"
qwen-tts = "*"

[feature.linux.pypi-dependencies]
# Linux: faster-whisper for GPU
faster-whisper = ">=1.0.0"

[feature.mac.pypi-dependencies]
# Mac: mlx-whisper for Apple Silicon
mlx = ">=0.5.0"
mlx-whisper = ">=0.4.0"

[environments]
linux = ["linux"]
mac = ["mac"]

[tasks]
pipeline = "python scripts/pipeline.py"
```

**Step 2: 변경 검증**

Run (Linux): `pixi run -e linux python -c "import torch; print(torch.cuda.is_available())"`
Expected: `True`

Run (Mac): `pixi run -e mac python -c "import torch; print(torch.backends.mps.is_available())"`
Expected: `True`

**Step 3: Commit**

```bash
git add pixi.toml
git commit -m "build: add osx-arm64 platform support with feature-based deps"
```

---

### Task 2: 디바이스 감지 유틸리티 생성

**Files:**
- Create: `scripts/device_utils.py`

**Step 1: 디바이스 감지 유틸리티 작성**

```python
#!/usr/bin/env python3
"""Device detection utilities for cross-platform support."""

import sys
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch


class DeviceType(Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    device_type: DeviceType
    device_name: str
    dtype: torch.dtype
    attn_implementation: str
    whisper_backend: Literal["faster-whisper", "mlx-whisper"]

    @property
    def torch_device(self) -> str:
        if self.device_type == DeviceType.CUDA:
            return "cuda:0"
        elif self.device_type == DeviceType.MPS:
            return "mps"
        return "cpu"

    @property
    def is_gpu(self) -> bool:
        return self.device_type in (DeviceType.CUDA, DeviceType.MPS)


def detect_device() -> DeviceInfo:
    """Detect the best available device for inference."""

    # Check CUDA first (Linux/Windows with NVIDIA GPU)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Check for FlashAttention support
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        return DeviceInfo(
            device_type=DeviceType.CUDA,
            device_name=f"{gpu_name} ({gpu_memory:.1f}GB)",
            dtype=torch.float16,
            attn_implementation=attn_impl,
            whisper_backend="faster-whisper",
        )

    # Check MPS (Apple Silicon Mac)
    if torch.backends.mps.is_available():
        # Get Mac model info
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            cpu_name = result.stdout.strip()
        except Exception:
            cpu_name = "Apple Silicon"

        return DeviceInfo(
            device_type=DeviceType.MPS,
            device_name=cpu_name,
            dtype=torch.float32,  # MPS requires float32 for Qwen3-TTS
            attn_implementation="sdpa",  # FlashAttention not available on Mac
            whisper_backend="mlx-whisper",
        )

    # Fallback to CPU
    return DeviceInfo(
        device_type=DeviceType.CPU,
        device_name=platform.processor() or "CPU",
        dtype=torch.float32,
        attn_implementation="sdpa",
        whisper_backend="mlx-whisper",  # mlx-whisper also works on CPU
    )


def print_device_info(device_info: DeviceInfo) -> None:
    """Print device information in a formatted way."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Device Configuration")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Device Type", device_info.device_type.value.upper())
    table.add_row("Device Name", device_info.device_name)
    table.add_row("PyTorch Device", device_info.torch_device)
    table.add_row("Data Type", str(device_info.dtype))
    table.add_row("Attention Implementation", device_info.attn_implementation)
    table.add_row("Whisper Backend", device_info.whisper_backend)

    console.print(table)


if __name__ == "__main__":
    device = detect_device()
    print_device_info(device)
```

**Step 2: 테스트 실행**

Run: `pixi run python scripts/device_utils.py`
Expected: Device Configuration 테이블 출력

**Step 3: Commit**

```bash
git add scripts/device_utils.py
git commit -m "feat: add cross-platform device detection utility"
```

---

### Task 3: Whisper 전사 함수 크로스플랫폼 지원

**Files:**
- Modify: `scripts/pipeline.py` (transcribe_audio 함수)

**Step 1: 전사 함수에 플랫폼별 분기 추가**

기존 `transcribe_audio` 함수를 다음으로 교체:

```python
def transcribe_audio(audio_path: Path, device_info: "DeviceInfo") -> str:
    """Transcribe audio using the appropriate backend for the platform."""
    console.print(Panel(f"[bold]Step 4: Transcribe Audio ({device_info.whisper_backend})[/bold]", style="blue"))

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_info.whisper_backend == "faster-whisper":
        # Linux with CUDA: use faster-whisper
        from faster_whisper import WhisperModel

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading Whisper large-v3 model...", total=None)
            compute_type = "float16" if device_info.dtype == torch.float16 else "float32"
            model = WhisperModel("large-v3", device=device_info.device_type.value, compute_type=compute_type)

        logger.info(f"Transcribing: {audio_path}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Transcribing...", total=None)
            segments, info = model.transcribe(str(audio_path), language="ko")
            text = " ".join([seg.text for seg in segments])

    else:
        # Mac/CPU: use mlx-whisper
        import mlx_whisper

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading Whisper large-v3 model (MLX)...", total=None)
            # mlx-whisper automatically uses the best available backend

        logger.info(f"Transcribing: {audio_path}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Transcribing...", total=None)
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                language="ko",
            )
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
```

**Step 2: Commit**

```bash
git add scripts/pipeline.py
git commit -m "feat: add mlx-whisper support for Mac transcription"
```

---

### Task 4: TTS 생성 함수 크로스플랫폼 지원

**Files:**
- Modify: `scripts/pipeline.py` (generate_notifications 함수)

**Step 1: TTS 함수에 MPS 지원 추가**

기존 `generate_notifications` 함수를 다음으로 교체:

```python
def generate_notifications(ref_audio_path: Path, ref_text: str, model_path: Path, device_info: "DeviceInfo"):
    """Generate all notification voice lines using voice cloning."""
    console.print(Panel(f"[bold]Step 6: Generate Notification Voice Lines ({device_info.device_type.value.upper()})[/bold]", style="blue"))

    from qwen_tts import Qwen3TTSModel

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task(f"Loading Qwen3-TTS 1.7B model on {device_info.device_type.value.upper()}...", total=None)

        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            dtype=device_info.dtype,
            attn_implementation=device_info.attn_implementation,
            device_map=device_info.torch_device,
        )

    # Synchronize MPS if needed
    if device_info.device_type == DeviceType.MPS:
        torch.mps.synchronize()

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

                # Synchronize MPS after generation
                if device_info.device_type == DeviceType.MPS:
                    torch.mps.synchronize()

                # Post-process: add 300ms silence at beginning
                processed_audio, sr = post_process_audio(wavs[0], sr, silence_ms=300)
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
```

**Step 2: Commit**

```bash
git add scripts/pipeline.py
git commit -m "feat: add MPS support for Qwen3-TTS on Mac"
```

---

### Task 5: 메인 파이프라인 통합

**Files:**
- Modify: `scripts/pipeline.py` (imports, check_gpu -> check_device, main)

**Step 1: imports 및 check_device 함수 수정**

파일 상단 imports에 추가:
```python
from device_utils import detect_device, print_device_info, DeviceType, DeviceInfo
```

`check_gpu()` 함수를 `check_device()`로 교체:
```python
def check_device() -> DeviceInfo:
    """Check and display device information."""
    console.print(Panel("[bold]Device Environment Check[/bold]", style="blue"))

    device_info = detect_device()
    print_device_info(device_info)

    if not device_info.is_gpu:
        logger.warning("No GPU detected, using CPU (will be slow)")

    console.print()
    return device_info
```

**Step 2: main() 함수 수정**

```python
def main():
    parser = argparse.ArgumentParser(description="Karina Voice Notification Generator")
    parser.add_argument("url", nargs="?", default=DEFAULT_YOUTUBE_URL, help="YouTube URL")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing audio in assets/raw/")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold magenta]Karina Voice Notification Generator[/bold magenta]\n"
        "[dim]Cross-platform (CUDA / MPS / CPU)[/dim]",
        border_style="magenta"
    ))
    console.print()

    # Check device (replaces check_gpu)
    device_info = check_device()

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

    # Step 4: Transcribe (pass device_info)
    transcript = transcribe_audio(selected_segment, device_info)

    # Step 5: Setup TTS model
    model_path = setup_tts_model()

    # Step 6: Generate notifications (pass device_info)
    generate_notifications(selected_segment, transcript, model_path, device_info)

    console.print(Panel.fit(
        "[bold green]Pipeline Complete![/bold green]\n\n"
        f"Generated notifications are in: [cyan]{OUTPUT_DIR}[/cyan]\n\n"
        "[dim]Next steps:[/dim]\n"
        "1. Review generated audio files\n"
        "2. Copy best ones to ~/.claude/sounds/\n"
        "3. Configure Claude Code notification hooks",
        border_style="green"
    ))
```

**Step 3: Commit**

```bash
git add scripts/pipeline.py
git commit -m "feat: integrate cross-platform device detection in pipeline"
```

---

### Task 6: README 및 CLAUDE.md 업데이트

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: README.md 업데이트**

Requirements 섹션 수정:
```markdown
## Requirements

### Linux (NVIDIA GPU)
- GPU: NVIDIA A100 (권장) 또는 CUDA 지원 GPU
- CUDA 12.0+
- pixi 패키지 매니저

### Mac (Apple Silicon)
- Mac mini / MacBook with M1/M2/M3/M4 chip
- RAM: 64GB 권장 (32GB 최소)
- pixi 패키지 매니저

## Setup

```bash
# pixi 설치
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc  # or ~/.zshrc on Mac

# 의존성 설치
cd karina-tts-notification

# Linux
pixi install -e linux

# Mac
pixi install -e mac
```

## Usage

```bash
# Linux
pixi run -e linux pipeline

# Mac
pixi run -e mac pipeline

# 또는 직접 실행 (pixi가 환경 자동 감지)
pixi run pipeline
```
```

**Step 2: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update for cross-platform support"
```

---

### Task 7: 테스트 및 최종 검증

**Step 1: Linux에서 테스트**

```bash
pixi run -e linux python scripts/pipeline.py --skip-download
```

Expected: CUDA 감지, faster-whisper 사용, float16으로 TTS 실행

**Step 2: Mac에서 테스트**

```bash
pixi run -e mac python scripts/pipeline.py --skip-download
```

Expected: MPS 감지, mlx-whisper 사용, float32로 TTS 실행

**Step 3: 최종 커밋 및 푸시**

```bash
git push origin main
```

---

## 요약

| 구성 요소 | Linux (CUDA) | Mac (MPS) |
|----------|--------------|-----------|
| Device | `cuda:0` | `mps` |
| dtype | `float16` | `float32` |
| Attention | FlashAttention2 / SDPA | SDPA |
| Whisper | faster-whisper | mlx-whisper |
| 예상 속도 | 빠름 | 적당함 (64GB RAM 활용) |
