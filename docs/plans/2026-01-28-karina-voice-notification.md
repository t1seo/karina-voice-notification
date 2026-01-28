# Karina Voice Notification for Claude Code

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** aespa 카리나의 목소리를 클로닝하여 Claude Code 알림 사운드로 사용

**Architecture:** YouTube에서 카리나 음성 샘플 다운로드 → MLX-Whisper로 transcribe → Qwen3-TTS로 voice cloning하여 알림 음성 생성 → Claude Code notification hook 설정

**Tech Stack:** pixi (dependency management), yt-dlp, ffmpeg, lightning-whisper-mlx, Qwen3-TTS (MPS), Python 3.12

---

## Prerequisites

시스템 요구사항:
- macOS with Apple Silicon (M1/M2/M3/M4)
- 약 10GB 디스크 공간 (모델 저장용)

---

## Task 1: Environment Setup with Pixi

**Files:**
- Create: `~/Documents/Github/karina-tts-notification/pixi.toml`
- Create: `~/Documents/Github/karina-tts-notification/assets/` (directory)
- Create: `~/.claude/hooks/karina-notification/` (directory)

**Step 1: Install pixi**

Run:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Expected: pixi installed, restart terminal or run `source ~/.bashrc` / `source ~/.zshrc`

**Step 2: Verify pixi installation**

Run:
```bash
pixi --version
```

Expected: pixi version displayed (e.g., `pixi 0.40.x`)

**Step 3: Create project directories**

Run:
```bash
mkdir -p ~/Documents/Github/karina-tts-notification/{assets,scripts,output}
mkdir -p ~/.claude/hooks/karina-notification
```

Expected: Directories created

**Step 4: Initialize pixi project**

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
pixi init
```

Expected: `pixi.toml` created

**Step 5: Create pixi.toml with all dependencies**

```toml
[project]
name = "karina-tts-notification"
version = "0.1.0"
description = "Karina voice notification for Claude Code"
channels = ["conda-forge", "pytorch"]
platforms = ["osx-arm64"]

[dependencies]
# Python
python = ">=3.12"

# System dependencies (conda-forge)
ffmpeg = ">=6.0"
yt-dlp = ">=2024.1.0"

# Python packages
pip = ">=24.0"

[pypi-dependencies]
# Audio processing
pydub = ">=0.25.1"

# Transcription (MLX-based for Apple Silicon)
lightning-whisper-mlx = ">=0.0.8"

# TTS - Qwen3-TTS
qwen-tts = ">=0.1.0"
torch = ">=2.0.0"
soundfile = ">=0.12.0"
huggingface-hub = ">=0.20.0"

# Utilities
tqdm = ">=4.66.0"

[tasks]
download = "python scripts/download_audio.py"
extract = "python scripts/extract_segment.py"
transcribe = "python scripts/transcribe.py"
setup-tts = "python scripts/setup_qwen_tts.py"
generate = "python scripts/generate_notifications.py"
```

**Step 6: Install all dependencies**

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
pixi install
```

Expected: All conda and PyPI dependencies installed

**Step 7: Verify installation**

Run:
```bash
pixi run python --version
pixi run which ffmpeg
pixi run which yt-dlp
```

Expected:
- Python 3.12.x
- ffmpeg path in `.pixi/envs/`
- yt-dlp path in `.pixi/envs/`

**Step 8: Commit environment setup**

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
git init
echo ".pixi/" >> .gitignore
git add pixi.toml pixi.lock .gitignore
git commit -m "chore: add pixi project configuration"
```

Note: `pixi.lock`은 재현 가능한 환경을 위해 커밋합니다. `.pixi/` 디렉토리는 실제 설치된 환경이므로 gitignore 처리합니다.

---

## Task 2: Download Karina Voice Sample from YouTube

**Files:**
- Create: `~/Documents/Github/karina-tts-notification/scripts/download_audio.py`

**Step 1: Create download script**

```python
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
```

**Step 2: Find and download Karina voice sample**

카리나의 목소리가 명확하게 들리는 인터뷰 또는 말하는 영상을 찾아 다운로드합니다.

추천 영상 유형:
- aespa 멤버 인터뷰 (카리나 솔로 파트)
- V-log 또는 브이라이브
- 팬미팅 멘트

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
# 예시 URL - 실제 카리나 인터뷰 영상 URL로 교체 필요
pixi run python scripts/download_audio.py "https://www.youtube.com/watch?v=KARINA_VIDEO_ID" karina_sample
```

Expected: `assets/raw/karina_sample.wav` 생성됨

**Step 3: Commit download script**

Run:
```bash
git add scripts/download_audio.py
git commit -m "feat: add YouTube audio download script"
```

---

## Task 3: Extract Clean Voice Segment

**Files:**
- Create: `~/Documents/Github/karina-tts-notification/scripts/extract_segment.py`

**Step 1: Create segment extraction script**

```python
#!/usr/bin/env python3
"""Extract clean voice segment from audio file."""

from pathlib import Path
from pydub import AudioSegment
import sys

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
RAW_AUDIO_DIR = ASSETS_DIR / "raw"
CLEAN_AUDIO_DIR = ASSETS_DIR / "clean"

def extract_segment(
    input_file: Path,
    start_ms: int,
    end_ms: int,
    output_name: str = "karina_clean"
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
```

**Step 2: Extract clean 10-15 second segment**

원본 오디오를 듣고 카리나 목소리만 깨끗하게 나오는 구간(10-15초)을 찾아 추출합니다.

Run:
```bash
# 먼저 오디오 파일 재생하여 좋은 구간 찾기
afplay assets/raw/karina_sample.wav

# 적절한 구간 추출 (예: 5초~18초)
pixi run python scripts/extract_segment.py assets/raw/karina_sample.wav 5000 18000 karina_clean
```

Expected: `assets/clean/karina_clean.wav` (약 10-15초, 16kHz mono)

**Step 3: Commit extraction script**

Run:
```bash
git add scripts/extract_segment.py
git commit -m "feat: add audio segment extraction script"
```

---

## Task 4: Transcribe Voice with Lightning-Whisper-MLX

**Files:**
- Create: `~/Documents/Github/karina-tts-notification/scripts/transcribe.py`

**Step 1: Create transcription script**

```python
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

    print(f"Loading Whisper model (distil-large-v3)...")
    whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12, quant=None)

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
```

**Step 2: Run transcription**

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
pixi run python scripts/transcribe.py assets/clean/karina_clean.wav ko
```

Expected:
- `assets/transcripts/karina_clean_transcript.json` 생성
- 콘솔에 한국어 텍스트 출력

**Step 3: Verify transcript accuracy**

생성된 transcript를 확인하고 필요시 수동으로 수정합니다.

**Step 4: Commit transcription script**

Run:
```bash
git add scripts/transcribe.py
git commit -m "feat: add MLX-Whisper transcription script"
```

---

## Task 5: Setup Qwen3-TTS for Voice Cloning

**Files:**
- Create: `~/Documents/Github/karina-tts-notification/scripts/setup_qwen_tts.py`

**Step 1: Create TTS setup script**

```python
#!/usr/bin/env python3
"""Setup and test Qwen3-TTS model for Apple Silicon."""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def check_mps_support():
    """Check if MPS (Metal Performance Shaders) is available."""
    print("Checking MPS support...")
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")

    if mps_available:
        print(f"MPS device: {torch.device('mps')}")
    else:
        print("Warning: MPS not available, will use CPU (slower)")

    return mps_available

def download_model():
    """Download Qwen3-TTS model for voice cloning."""
    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # Smaller model for faster inference
    local_dir = MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-Base"

    if local_dir.exists():
        print(f"Model already downloaded: {local_dir}")
        return local_dir

    print(f"Downloading {model_name}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False
    )

    # Also download tokenizer
    tokenizer_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    tokenizer_dir = MODELS_DIR / "Qwen3-TTS-Tokenizer-12Hz"

    if not tokenizer_dir.exists():
        print(f"Downloading {tokenizer_name}...")
        snapshot_download(
            repo_id=tokenizer_name,
            local_dir=str(tokenizer_dir),
            local_dir_use_symlinks=False
        )

    print("Model download complete!")
    return local_dir

def test_model():
    """Test the TTS model with a simple generation."""
    from qwen_tts import Qwen3TTSModel
    import soundfile as sf

    model_path = MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-Base"

    print("Loading Qwen3-TTS model...")
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,  # float32 required for MPS voice clone
        attn_implementation="sdpa",  # Use SDPA for Mac (not FlashAttention)
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
    )

    # Simple test generation (voice design mode, not clone)
    print("Testing basic TTS generation...")
    test_text = "안녕하세요, 테스트입니다."

    wavs, sr = model.generate(
        text=test_text,
        voice="A young Korean female voice, clear and friendly"
    )

    output_path = PROJECT_ROOT / "output" / "test_output.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sr)

    print(f"Test audio saved to: {output_path}")
    return True

if __name__ == "__main__":
    check_mps_support()
    download_model()
    test_model()
```

**Step 2: Run setup**

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
pixi run python scripts/setup_qwen_tts.py
```

Expected:
- MPS support confirmed
- Model downloaded to `models/` directory
- Test audio generated at `output/test_output.wav`

**Step 3: Verify test audio**

Run:
```bash
afplay output/test_output.wav
```

Expected: Korean TTS audio plays successfully

**Step 4: Commit setup script**

Run:
```bash
git add scripts/setup_qwen_tts.py
echo "models/" >> .gitignore
git add .gitignore
git commit -m "feat: add Qwen3-TTS setup script"
```

---

## Task 6: Generate Notification Voice Lines

**Files:**
- Create: `~/Documents/Github/karina-tts-notification/scripts/generate_notifications.py`
- Create: `~/Documents/Github/karina-tts-notification/notification_lines.json`

**Step 1: Define notification lines**

```json
{
  "permission_prompt": [
    {
      "text": "오빠, 잠깐! 이거 해도 돼?",
      "filename": "permission_prompt_1.wav"
    },
    {
      "text": "허락이 필요해요~",
      "filename": "permission_prompt_2.wav"
    }
  ],
  "idle_prompt": [
    {
      "text": "오빠, 다 했어! 확인해줘~",
      "filename": "idle_prompt_1.wav"
    },
    {
      "text": "끝났어요, 봐주세요!",
      "filename": "idle_prompt_2.wav"
    }
  ],
  "auth_success": [
    {
      "text": "인증 완료! 고마워요~",
      "filename": "auth_success_1.wav"
    }
  ],
  "elicitation_dialog": [
    {
      "text": "여기 입력이 필요해요!",
      "filename": "elicitation_dialog_1.wav"
    }
  ]
}
```

**Step 2: Create voice generation script**

```python
#!/usr/bin/env python3
"""Generate notification voice lines using Karina's cloned voice."""

import json
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "notifications"
NOTIFICATION_LINES_FILE = PROJECT_ROOT / "notification_lines.json"

def load_model():
    """Load Qwen3-TTS model."""
    from qwen_tts import Qwen3TTSModel

    model_path = MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-Base"

    print("Loading Qwen3-TTS model...")
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,  # Required for MPS voice clone
        attn_implementation="sdpa",
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
    )
    return model

def generate_cloned_voice(
    model,
    text: str,
    ref_audio_path: Path,
    ref_text: str,
    output_path: Path
):
    """Generate speech with cloned voice.

    Args:
        model: Loaded Qwen3-TTS model
        text: Text to synthesize
        ref_audio_path: Reference audio for voice cloning
        ref_text: Transcript of reference audio
        output_path: Where to save the output
    """
    wavs, sr = model.generate_voice_clone(
        text=text,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sr)

    return output_path

def generate_all_notifications(
    ref_audio_path: Path,
    ref_text: str
):
    """Generate all notification voice lines.

    Args:
        ref_audio_path: Path to Karina's reference audio
        ref_text: Transcript of reference audio
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load notification lines
    with open(NOTIFICATION_LINES_FILE, "r", encoding="utf-8") as f:
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
                    output_path=output_path
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

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    ref_text = transcript_data["text"]

    print(f"Reference audio: {ref_audio}")
    print(f"Reference text: {ref_text}")

    generate_all_notifications(ref_audio, ref_text)
```

**Step 3: Run voice generation**

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
pixi run python scripts/generate_notifications.py
```

Expected:
- `output/notifications/permission_prompt/*.wav`
- `output/notifications/idle_prompt/*.wav`
- `output/notifications/auth_success/*.wav`
- `output/notifications/elicitation_dialog/*.wav`

**Step 4: Preview generated audio**

Run:
```bash
afplay output/notifications/idle_prompt/idle_prompt_1.wav
afplay output/notifications/permission_prompt/permission_prompt_1.wav
```

Expected: Karina-like voice plays the notification lines

**Step 5: Commit generation scripts**

Run:
```bash
git add scripts/generate_notifications.py notification_lines.json
git commit -m "feat: add notification voice generation with Qwen3-TTS"
```

---

## Task 7: Backup Assets and Install Notifications

**Files:**
- Modify: `~/Documents/Github/karina-tts-notification/assets/`
- Create: `~/.claude/hooks/karina-notification/sounds/`

**Step 1: Copy best notification sounds to assets (backup)**

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
mkdir -p assets/final_notifications

# Copy the best version of each notification type
cp output/notifications/permission_prompt/permission_prompt_1.wav assets/final_notifications/
cp output/notifications/idle_prompt/idle_prompt_1.wav assets/final_notifications/
cp output/notifications/auth_success/auth_success_1.wav assets/final_notifications/
cp output/notifications/elicitation_dialog/elicitation_dialog_1.wav assets/final_notifications/
```

**Step 2: Install to Claude hooks directory**

Run:
```bash
mkdir -p ~/.claude/hooks/karina-notification/sounds

# Copy all notification sounds
cp assets/final_notifications/*.wav ~/.claude/hooks/karina-notification/sounds/
```

**Step 3: Verify installation**

Run:
```bash
ls -la ~/.claude/hooks/karina-notification/sounds/
```

Expected: All .wav files present

**Step 4: Commit assets**

Run:
```bash
git add assets/final_notifications/
git commit -m "feat: add final notification sound assets"
```

---

## Task 8: Configure Claude Code Notification Hooks

**Files:**
- Create: `~/.claude/hooks/karina-notification/play_notification.sh`
- Modify: `~/.claude/settings.json` (hooks configuration)

**Step 1: Create notification player script**

```bash
#!/bin/bash
# Play Karina notification sound based on notification type
# Usage: play_notification.sh <notification_type>

SOUNDS_DIR="$HOME/.claude/hooks/karina-notification/sounds"
NOTIFICATION_TYPE="${1:-idle_prompt}"

case "$NOTIFICATION_TYPE" in
    "permission_prompt")
        SOUND_FILE="$SOUNDS_DIR/permission_prompt_1.wav"
        ;;
    "idle_prompt")
        SOUND_FILE="$SOUNDS_DIR/idle_prompt_1.wav"
        ;;
    "auth_success")
        SOUND_FILE="$SOUNDS_DIR/auth_success_1.wav"
        ;;
    "elicitation_dialog")
        SOUND_FILE="$SOUNDS_DIR/elicitation_dialog_1.wav"
        ;;
    *)
        SOUND_FILE="$SOUNDS_DIR/idle_prompt_1.wav"
        ;;
esac

if [ -f "$SOUND_FILE" ]; then
    afplay "$SOUND_FILE" &
fi
```

Run:
```bash
cat > ~/.claude/hooks/karina-notification/play_notification.sh << 'EOF'
#!/bin/bash
SOUNDS_DIR="$HOME/.claude/hooks/karina-notification/sounds"
NOTIFICATION_TYPE="${1:-idle_prompt}"

case "$NOTIFICATION_TYPE" in
    "permission_prompt")
        SOUND_FILE="$SOUNDS_DIR/permission_prompt_1.wav"
        ;;
    "idle_prompt")
        SOUND_FILE="$SOUNDS_DIR/idle_prompt_1.wav"
        ;;
    "auth_success")
        SOUND_FILE="$SOUNDS_DIR/auth_success_1.wav"
        ;;
    "elicitation_dialog")
        SOUND_FILE="$SOUNDS_DIR/elicitation_dialog_1.wav"
        ;;
    *)
        SOUND_FILE="$SOUNDS_DIR/idle_prompt_1.wav"
        ;;
esac

if [ -f "$SOUND_FILE" ]; then
    afplay "$SOUND_FILE" &
fi
EOF

chmod +x ~/.claude/hooks/karina-notification/play_notification.sh
```

**Step 2: Create hook handler script**

```bash
#!/bin/bash
# Claude Code Notification Hook Handler
# Reads notification type from stdin and plays corresponding sound

# Read JSON input from stdin
read -r INPUT

# Extract notification_type using jq or simple parsing
NOTIFICATION_TYPE=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('notification_type', 'idle_prompt'))")

# Play the notification
~/.claude/hooks/karina-notification/play_notification.sh "$NOTIFICATION_TYPE"
```

Run:
```bash
cat > ~/.claude/hooks/karina-notification/hook_handler.sh << 'EOF'
#!/bin/bash
read -r INPUT
NOTIFICATION_TYPE=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('notification_type', 'idle_prompt'))")
~/.claude/hooks/karina-notification/play_notification.sh "$NOTIFICATION_TYPE"
EOF

chmod +x ~/.claude/hooks/karina-notification/hook_handler.sh
```

**Step 3: Update Claude Code settings**

현재 Claude Code 설정 파일을 확인하고 Notification hooks를 추가합니다.

Run:
```bash
# Backup current settings
cp ~/.claude/settings.json ~/.claude/settings.json.backup

# View current hooks
cat ~/.claude/settings.json | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin).get('hooks', {}), indent=2))"
```

Claude Code settings.json에 다음 hooks 설정 추가/수정:

```json
{
  "hooks": {
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/karina-notification/hook_handler.sh",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

**Step 4: Test notification hook**

Run:
```bash
# Test the hook manually
echo '{"notification_type": "permission_prompt"}' | ~/.claude/hooks/karina-notification/hook_handler.sh
echo '{"notification_type": "idle_prompt"}' | ~/.claude/hooks/karina-notification/hook_handler.sh
```

Expected: Karina's voice notification plays

**Step 5: Commit hook scripts to project**

Run:
```bash
cd ~/Documents/Github/karina-tts-notification
mkdir -p hooks
cp ~/.claude/hooks/karina-notification/*.sh hooks/
git add hooks/
git commit -m "feat: add Claude Code notification hook scripts"
```

---

## Task 9: Final Verification and Documentation

**Files:**
- Create: `~/Documents/Github/karina-tts-notification/README.md`

**Step 1: Create README**

```markdown
# Karina Voice Notification for Claude Code

aespa 카리나의 목소리를 클로닝하여 Claude Code 알림음으로 사용하는 프로젝트

## Features

- YouTube에서 음성 샘플 다운로드
- MLX-Whisper로 한국어 음성 인식
- Qwen3-TTS로 voice cloning
- Claude Code notification hook 연동

## Setup

1. **Install pixi and dependencies**
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   pixi install
   ```

2. **Download voice sample**
   ```bash
   pixi run python scripts/download_audio.py "YOUTUBE_URL"
   # Or use the task shortcut:
   pixi run download "YOUTUBE_URL"
   ```

3. **Extract clean segment**
   ```bash
   pixi run python scripts/extract_segment.py assets/raw/karina_sample.wav START_MS END_MS
   ```

4. **Transcribe**
   ```bash
   pixi run python scripts/transcribe.py assets/clean/karina_clean.wav ko
   ```

5. **Generate notifications**
   ```bash
   pixi run python scripts/generate_notifications.py
   # Or use the task shortcut:
   pixi run generate
   ```

6. **Install hooks**
   ```bash
   cp -r hooks/* ~/.claude/hooks/karina-notification/
   cp assets/final_notifications/*.wav ~/.claude/hooks/karina-notification/sounds/
   ```

## Notification Types

| Type | 대사 | 파일 |
|------|------|------|
| permission_prompt | "오빠, 잠깐! 이거 해도 돼?" | permission_prompt_1.wav |
| idle_prompt | "오빠, 다 했어! 확인해줘~" | idle_prompt_1.wav |
| auth_success | "인증 완료! 고마워요~" | auth_success_1.wav |
| elicitation_dialog | "여기 입력이 필요해요!" | elicitation_dialog_1.wav |

## Tech Stack

- pixi: Python + 시스템 의존성 관리
- yt-dlp: YouTube 다운로드
- ffmpeg: 오디오 처리
- lightning-whisper-mlx: Apple Silicon 최적화 음성인식
- Qwen3-TTS: Voice cloning TTS
- Claude Code hooks: 알림 연동
```

**Step 2: Final commit**

Run:
```bash
git add README.md
git commit -m "docs: add project README"
```

**Step 3: Verify everything works**

Claude Code에서 작업 후 알림이 울리는지 확인합니다.

Run:
```bash
# Restart Claude Code to apply hook changes
# Then perform any action that triggers a notification
```

Expected: 카리나 목소리 알림이 재생됨

---

## Summary

| Task | Description | Output |
|------|-------------|--------|
| 1 | Environment Setup with Pixi | pixi.toml, .pixi/ environment |
| 2 | Download YouTube Audio | assets/raw/karina_sample.wav |
| 3 | Extract Clean Segment | assets/clean/karina_clean.wav |
| 4 | Transcribe with MLX-Whisper | assets/transcripts/*.json |
| 5 | Setup Qwen3-TTS | models/ directory |
| 6 | Generate Notification Lines | output/notifications/*.wav |
| 7 | Backup and Install | ~/.claude/hooks/karina-notification/ |
| 8 | Configure Hooks | settings.json updated |
| 9 | Documentation | README.md |

## References

- [Pixi - Fast Package Manager](https://pixi.sh)
- [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Q3-TTS for Apple Silicon](https://github.com/esendjer/Q3-TTS)
- [Claude Code Hooks Documentation](https://docs.anthropic.com/claude-code/hooks)
