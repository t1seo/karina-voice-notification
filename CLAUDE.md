# Karina Voice Notification - Claude Code Guide

This project generates Claude Code notification sounds using aespa Karina's voice.

## Quick Setup (Using Pre-generated Audio)

### 1. Copy Audio Files

```bash
mkdir -p ~/.claude/sounds
cp output/*/*.wav ~/.claude/sounds/
```

### 2. Install Hook Script (Recommended)

> **Note**: Claude Code's `matcher`-based Notification hooks may not work reliably. Use this unified Python script approach instead, which parses the `notification_type` field to play the appropriate sound.

```bash
mkdir -p ~/.claude/hooks
cp scripts/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

Add to the `hooks` section in `~/.claude/settings.json`:

```json
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
```

## Hook Types and Audio Files

| Hook Matcher | Trigger | Audio File | Dialog |
|--------------|---------|------------|--------|
| `permission_prompt` | Before executing risky commands | `permission_prompt_1.wav` | Wait! Is it okay to run this? Please allow~ |
| | | `permission_prompt_2.wav` | Hold on, I need permission for this task~ |
| `idle_prompt` | After task completion, waiting for response | `idle_prompt_1.wav` | All done! Please check the results~ |
| | | `idle_prompt_2.wav` | Task completed, would you take a look? |
| `auth_success` | Authentication success | `auth_success_1.wav` | Authentication complete! Thanks for your help~ |
| `elicitation_dialog` | User input required | `elicitation_dialog_1.wav` | Input needed here! Could you fill this in? |

## Customizing Dialog

### 1. Edit notification_lines.json

```json
{
  "permission_prompt": [
    {"text": "Enter your new dialog here", "filename": "permission_prompt_1.wav"}
  ]
}
```

### 2. Regenerate Audio

```bash
# GPU environment required
pixi run python scripts/pipeline.py --skip-download
```

### 3. Copy New Audio

```bash
cp output/*/*.wav ~/.claude/sounds/
```

## Platform-specific Playback Commands

| Platform | Command |
|----------|---------|
| macOS | `afplay ~/.claude/sounds/filename.wav` |
| Linux (ALSA) | `aplay ~/.claude/sounds/filename.wav` |
| Linux (PulseAudio) | `paplay ~/.claude/sounds/filename.wav` |

## Generating New Audio (GPU Required)

```bash
# 1. Download voice sample from YouTube and generate
pixi run python scripts/pipeline.py "https://youtube.com/watch?v=..."

# 2. Regenerate with existing sample
pixi run python scripts/pipeline.py --skip-download
```

## Notes

- Qwen3-TTS reference audio: 5-15 seconds recommended
- Clean, noise-free voice samples required
- Generation only possible in GPU (CUDA 12.0+ or Apple Silicon MPS) environment
