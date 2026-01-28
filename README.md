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

## How It Works

This project uses a **voice cloning pipeline** to generate custom notification sounds. Give it any YouTube video with clear vocals, and it will clone that voice to speak your notification phrases.

### Pipeline Overview

```
YouTube Video → Audio Extraction → Voice Segment Selection → Transcription → Voice Cloning → Notification Sounds
```

| Step | What Happens | Why |
|------|--------------|-----|
| **1. Audio Download** | Extract audio from YouTube using yt-dlp | Get raw voice material from any source |
| **2. Segment Split** | Split into 15-second clips at 30-second intervals | Find clean voice segments without music/noise |
| **3. Segment Selection** | Interactive menu to pick the best clip | Human ear picks cleaner audio than algorithms |
| **4. Transcription** | Convert speech to text using Whisper large-v3 | TTS needs reference text to match voice patterns |
| **5. Voice Cloning** | Generate new speech using Qwen3-TTS 1.7B | Clone the voice to speak custom notification phrases |

### Voice Cloning Technology

The pipeline uses **Qwen3-TTS 1.7B**, a state-of-the-art text-to-speech model that can clone voices from short audio samples. It analyzes the reference audio's:
- Tone and pitch patterns
- Speaking rhythm and pace
- Voice characteristics and timbre

Then generates new speech that sounds like the same person speaking your custom phrases.

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

## Choosing a Good Voice Sample

> **Critical**: Voice cloning quality depends entirely on your source audio.

**Good sources** (clean, isolated vocals):
- Interview clips
- Solo speaking segments
- Behind-the-scenes talking moments
- Podcast recordings

**Avoid** (will produce poor results):
- Music videos
- Clips with background music
- Noisy environments
- Multiple speakers

The reference audio should be **5-15 seconds** of clear speech.

---

## Generated Notifications

The pipeline generates notification sounds for Claude Code events:

| Event | When it plays |
|-------|---------------|
| `permission_prompt` | Claude asks permission before risky commands |
| `idle_prompt` | Task complete, waiting for response |
| `auth_success` | Authentication successful |
| `elicitation_dialog` | User input required |

Each notification type generates **10 voice variations** that play randomly for variety.

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

---

## Claude Code Integration

After generating your notification sounds, copy them to Claude Code:

```bash
# Copy audio files
mkdir -p ~/.claude/sounds
cp output/notifications/*/*.wav ~/.claude/sounds/

# Install hook script
mkdir -p ~/.claude/hooks
cp src/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

Then add the hook configuration to `~/.claude/settings.json`. See [CLAUDE.md](CLAUDE.md) for detailed setup instructions.

---

## Project Structure

```
project-karina-voice/
├── src/
│   ├── pipeline.py              # Main pipeline orchestrator
│   ├── download_audio.py        # YouTube audio extraction
│   ├── transcribe.py            # Whisper transcription
│   ├── generate_notifications.py # Qwen3-TTS voice cloning
│   └── claude_notification_hook.py # Claude Code hook
├── output/
│   ├── raw/                     # Downloaded audio
│   ├── clean/                   # Selected voice segments
│   └── notifications/           # Generated notification sounds
├── notification_lines.json      # Customizable phrases
└── pixi.toml                    # Dependencies
```

---

## Troubleshooting

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
