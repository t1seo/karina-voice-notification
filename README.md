# Karina Voice Notification Generator

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/platform-Linux%20|%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-CUDA%2012.0%2B%20|%20Apple%20Silicon-green" alt="GPU">
  <img src="https://img.shields.io/badge/TTS-Qwen3--TTS%201.7B-orange" alt="TTS Model">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <br>
  <a href="README.ko.md"><img src="https://img.shields.io/badge/lang-한국어-red" alt="Korean"></a>
</p>

<p align="center">
  <img src="assets/karina.jpg" alt="Karina" width="800">
</p>

Generate Claude Code notification sounds with **aespa Karina's voice** using AI voice cloning technology.

> **What is this?** This pipeline extracts a voice sample from YouTube, transcribes it, and uses Qwen3-TTS to generate custom notification sounds that play when Claude Code needs your attention.

---

## Features

- **Voice Cloning**: Generate notifications in Karina's voice using Qwen3-TTS 1.7B
- **Cross-Platform**: Works on Linux (CUDA) and macOS (Apple Silicon)
- **Interactive Menu**: Easy-to-use arrow key navigation for segment selection
- **Customizable**: Edit notification phrases and regenerate audio anytime

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

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc  # or ~/.zshrc on Mac
```

### 2. Clone and Setup

```bash
git clone https://github.com/t1seo/project-karina-voice.git
cd project-karina-voice

# For Linux
pixi install -e linux
pixi run -e linux install-deps-linux

# For macOS
pixi install -e mac
pixi run -e mac install-deps-mac
```

### 3. Run the Pipeline

```bash
# Linux
pixi run -e linux pipeline

# macOS
pixi run -e mac pipeline
```

The interactive menu will guide you through:
1. Downloading audio from YouTube
2. Selecting a clean voice segment
3. Transcribing the audio
4. Generating notification sounds

---

## Choosing a Good Voice Sample

> **Important**: For best results, choose a YouTube video with **voice only** (no background music). Clean, isolated vocals produce much better voice cloning results than audio mixed with music or sound effects.

**Good sources:**
- Interview clips
- Solo speaking segments
- Behind-the-scenes talking moments

**Avoid:**
- Music videos
- Clips with background music
- Noisy environments

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1. GPU Check | Detect CUDA/MPS availability |
| 2. Download Audio | Extract audio from YouTube using yt-dlp |
| 3. Split Segments | Create 15-second clips at 30-second intervals |
| 4. Select Segment | Choose a clean voice segment (manual or auto) |
| 5. Transcribe | Convert speech to text using Whisper |
| 6. Setup TTS | Download Qwen3-TTS 1.7B model |
| 7. Generate | Create notification sounds via voice cloning |

---

## Generated Notifications

| Type | Phrase | File |
|------|--------|------|
| permission_prompt | Wait! Is it okay to run this? | `permission_prompt_1.wav` |
| idle_prompt | All done! Please check the results~ | `idle_prompt_1.wav` |
| auth_success | Authentication complete! | `auth_success_1.wav` |
| elicitation_dialog | Input needed here! | `elicitation_dialog_1.wav` |

Output files are saved to `output/notifications/`

---

## Claude Code Integration

### 1. Copy Audio Files

```bash
mkdir -p ~/.claude/sounds
cp output/notifications/*/*.wav ~/.claude/sounds/
```

### 2. Install Hook Script

```bash
mkdir -p ~/.claude/hooks
cp src/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

### 3. Configure Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/claude_notification_hook.py",
            "timeout": 10
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/claude_notification_hook.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

### Notification Types

| Type | When it plays |
|------|---------------|
| `permission_prompt` | Claude asks permission before risky commands |
| `idle_prompt` | Task complete, waiting 60+ seconds |
| `auth_success` | Authentication successful |
| `elicitation_dialog` | User input required |
| Stop event | Claude response complete |

---

## Customization

### Change Notification Phrases

Edit `notification_lines.json`:

```json
{
  "permission_prompt": [
    {"text": "Your custom phrase here", "filename": "permission_prompt_1.wav"}
  ]
}
```

### Regenerate Audio

```bash
pixi run -e mac pipeline --skip-download
```

---

## Project Structure

```
project-karina-voice/
├── src/
│   ├── pipeline.py              # Main pipeline
│   ├── device_utils.py          # GPU detection
│   ├── download_audio.py        # YouTube download
│   ├── transcribe.py            # Whisper transcription
│   ├── generate_notifications.py # Voice generation
│   └── claude_notification_hook.py # Claude hook
├── output/
│   ├── raw/                     # Downloaded audio
│   ├── clean/                   # Processed segments
│   └── notifications/           # Generated sounds
├── notification_lines.json      # Phrase configuration
└── pixi.toml                    # Dependencies
```

---

## Troubleshooting

### "No module named 'xxx'"
Run the dependency installation:
```bash
pixi run -e mac install-deps-mac  # or install-deps-linux
```

### Poor voice quality
- Use a cleaner voice sample without background music
- Ensure the reference audio is 5-15 seconds
- Try a different segment from the source video

### Hook not playing sounds
- Check that audio files exist in `~/.claude/sounds/`
- Verify the hook script has execute permissions
- Check `~/.claude/hooks/hook_debug.log` for errors

---

## License

MIT License
