# Karina Voice Notification Generator

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/platform-Linux%20|%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-CUDA%2012.0%2B%20|%20Apple%20Silicon-green" alt="GPU">
  <img src="https://img.shields.io/badge/TTS-Qwen3--TTS%201.7B-orange" alt="TTS Model">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <a href="README.ko.md"><img src="https://img.shields.io/badge/README-한국어-red" alt="Korean"></a>
</p>

<p align="center">
  <img src="assets/karina.jpg" alt="Karina" width="800">
</p>

Generate Claude Code notification sounds with **any voice** using AI voice cloning technology.

> **Use Any Voice!** Clone your favorite idol, voice actor, streamer, or even your own voice from any YouTube video!

---

## Features

- **Voice Cloning** - Clone any voice from YouTube videos using Qwen3-TTS 1.7B
- **BGM Removal (Recommended)** - Automatically separate vocals from background music using Demucs AI
- **Audio Normalization** - Automatically normalize audio to prevent clipping
- **Audio Post-processing** - Optional audio enhancement (noise reduction, EQ, compression, loudness normalization)
- **Multi-language Support** - 10 languages for TTS (Korean, English, Chinese, Japanese, etc.)
- **Interactive Menus** - Easy-to-use keyboard navigation for all options
- **Output Cleanup** - Option to delete previous outputs when starting a new pipeline
- **Code Quality** - Pre-commit hooks with ruff linter/formatter

---

## How It Works

### Pipeline Overview

```
YouTube Video
     ↓
Audio Download (yt-dlp)
     ↓
[Optional] BGM Removal (Demucs AI)
     ↓
Segment Split (15s clips)
     ↓
Segment Selection (interactive)
     ↓
Transcription (Whisper large-v3)
     ↓
Voice Cloning (Qwen3-TTS 1.7B)
     ↓
[Optional] Post-processing
     ↓
Notification Sounds
```

| Step | What Happens | Technology |
|------|--------------|------------|
| **Audio Download** | Extract audio from YouTube | yt-dlp |
| **BGM Removal** | Separate vocals from background music | Demucs (Meta AI) |
| **Segment Split** | Split into 15-second clips | pydub |
| **Transcription** | Convert speech to text | Whisper large-v3 |
| **Voice Cloning** | Generate new speech in cloned voice | Qwen3-TTS 1.7B |
| **Post-processing** | Enhance audio quality | noisereduce, pedalboard, pyloudnorm |

### Post-processing Pipeline

```
Raw TTS Output
     ↓
Noise Reduction (spectral gating)
     ↓
Voice EQ (80Hz highpass, 12kHz lowpass)
     ↓
Dynamics (compression + limiting)
     ↓
Loudness Normalization (-14 LUFS)
     ↓
Final Output
```

---

## Requirements

### Linux (NVIDIA GPU)
- NVIDIA GPU with CUDA support (A100 recommended)
- CUDA 12.0+
- [pixi](https://pixi.sh) package manager

### macOS (Apple Silicon)
- Mac with M1/M2/M3/M4 chip
- 64GB RAM recommended (32GB minimum)
- [pixi](https://pixi.sh) package manager

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/t1seo/karina-voice-notification.git
cd project-karina-voice

# Install dependencies
pixi install
pixi run install-deps-mac    # macOS
pixi run install-deps-linux  # Linux

# Run the pipeline
pixi run pipeline
```

---

## Supported Languages

### TTS (Voice Generation)
| Language | Code |
|----------|------|
| Korean | korean |
| English | english |
| Chinese | chinese |
| Japanese | japanese |
| French | french |
| German | german |
| Spanish | spanish |
| Italian | italian |
| Portuguese | portuguese |
| Russian | russian |

### Transcription (Whisper)
Supports all languages above with automatic detection or manual selection.

---

## Choosing a Good Voice Sample

> **Critical**: Voice cloning quality depends entirely on your source audio.

**Good sources** (clean, isolated vocals):
- Interview clips
- Solo speaking segments
- Behind-the-scenes talking moments
- Podcast recordings

**Avoid** (will produce poor results):
- Music videos (use BGM Removal option)
- Noisy environments
- Multiple speakers

The reference audio should be **5-15 seconds** of clear speech.

---

## Customization

### Custom Notification Phrases

Edit `notification_lines.json` to change what the cloned voice says:

```json
{
  "permission_prompt": [
    {"text": "Your custom phrase here", "filename": "permission_prompt_1.wav"}
  ]
}
```

### Post-processing Settings

Adjust in `src/post_process.py`:
- `target_lufs`: Target loudness (-14 LUFS default, streaming standard)
- `denoise_strength`: Noise reduction intensity (0.0-1.0)
- `highpass_freq`: Remove low rumble (80Hz default)
- `lowpass_freq`: Remove high-freq noise (12kHz default)

---

## Claude Code Integration

After generating your notification sounds:

```bash
# Copy audio files
mkdir -p ~/.claude/sounds
cp output/notifications/*/*.wav ~/.claude/sounds/

# Install hook script
mkdir -p ~/.claude/hooks
cp .claude/skills/setup-notifications/scripts/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

Then add the hook configuration to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Notification": [
      {
        "hooks": [{"type": "command", "command": "python3 ~/.claude/hooks/claude_notification_hook.py", "timeout": 10}]
      }
    ],
    "Stop": [
      {
        "hooks": [{"type": "command", "command": "python3 ~/.claude/hooks/claude_notification_hook.py", "timeout": 10}]
      }
    ]
  }
}
```

---

## Troubleshooting

### Poor voice quality
- Use a cleaner voice sample without background music
- Enable BGM Removal (Demucs) in the pipeline menu
- Ensure the reference audio is 5-15 seconds
- Try enabling post-processing for enhanced output

### Post-processing not working
```bash
# Install dependencies in the correct pixi environment
pixi run -e mac pip install noisereduce pedalboard pyloudnorm  # macOS
pixi run -e linux pip install noisereduce pedalboard pyloudnorm  # Linux
```

### Hook not playing sounds
- Check that audio files exist in `~/.claude/sounds/`
- Verify the hook script has execute permissions
- Check `~/.claude/hooks/hook_debug.log` for errors

---

## License

MIT License
