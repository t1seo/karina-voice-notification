# Karina Voice Notification Generator

<p align="center">
  <a href="README.ko.md"><img src="https://img.shields.io/badge/README-한국어-red" alt="Korean"></a>
</p>

<p align="center">
  <img src="assets/karina.jpg" alt="Karina" width="600">
</p>

Generate Claude Code notification sounds with **any voice** from YouTube videos.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/t1seo/karina-voice-notification.git
cd karina-voice-notification

pixi install
pixi run install-deps-mac    # macOS (Apple Silicon)
pixi run install-deps-linux  # Linux (NVIDIA GPU)
```

### 2. Run

```bash
pixi run pipeline
```

Follow the interactive menu to:
1. Paste a YouTube URL with clear voice
2. Select a clean voice segment (5-15 seconds)
3. Generate notification sounds

### 3. Use with Claude Code

In Claude Code, run:

```
/setup-notifications
```

This skill automatically copies sounds and configures hooks for you.

---

## Requirements

| Platform | Requirements |
|----------|-------------|
| **macOS** | Apple Silicon (M1+), 32GB+ RAM, [pixi](https://pixi.sh) |
| **Linux** | NVIDIA GPU, CUDA 12.0+, [pixi](https://pixi.sh) |

---

## Tips for Best Results

**Good voice sources:**
- Interview clips, solo speaking, podcasts
- Enable "BGM Removal" for music videos

**Avoid:**
- Noisy environments, multiple speakers
- Clips shorter than 5 seconds

---

## Customization

Edit `notification_lines.json` to change notification phrases:

```json
{"text": "Your custom phrase here", "filename": "permission_prompt_1.wav"}
```

---

## How It Works

```
YouTube → Download → [BGM Removal] → Split → Select → Transcribe → Voice Clone → Output
```

| Step | Technology |
|------|------------|
| Download | yt-dlp |
| BGM Removal | Demucs (Meta AI) |
| Transcription | Whisper large-v3 |
| Voice Cloning | Qwen3-TTS 1.7B |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Poor voice quality | Use cleaner source, enable BGM Removal |
| Hook not playing | Check `~/.claude/sounds/` exists, verify permissions |
| Missing dependencies | Run `pixi run install-deps-mac` or `install-deps-linux` |

---

## License

MIT License
