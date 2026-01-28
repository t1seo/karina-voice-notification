---
name: setup-notifications
description: Set up Karina voice notification sounds for Claude Code. Use when user asks to "install notifications", "setup sounds", "configure voice alerts", or mentions "Karina voice notifications".
disable-model-invocation: true
---

# Setup Karina Voice Notifications

This skill helps you install custom voice notification sounds for Claude Code events.

## What Gets Installed

| Event | Sound | When It Plays |
|-------|-------|---------------|
| `permission_prompt` | Permission request voice | Before executing risky commands |
| `idle_prompt` | Task complete voice | After task completion |
| `auth_success` | Authentication success voice | After successful auth |
| `elicitation_dialog` | Input needed voice | When user input is required |

## Installation Steps

### 1. Copy Audio Files

```bash
mkdir -p ~/.claude/sounds
cp output/notifications/*/*.wav ~/.claude/sounds/
```

### 2. Copy Hook Script

```bash
mkdir -p ~/.claude/hooks
cp .claude/skills/setup-notifications/scripts/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

### 3. Configure Claude Code Settings

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

## Verification

After installation, test by:
1. Running any Claude Code task
2. When the task completes, you should hear the idle_prompt sound
3. When permission is required, you should hear the permission_prompt sound

## Troubleshooting

If sounds don't play:
- Check files exist: `ls ~/.claude/sounds/`
- Check hook permissions: `ls -la ~/.claude/hooks/`
- Check debug log: `cat ~/.claude/hooks/hook_debug.log`

## Platform-Specific Notes

| Platform | Audio Command |
|----------|---------------|
| macOS | `afplay` (default) |
| Linux (ALSA) | Change to `aplay` in hook script |
| Linux (PulseAudio) | Change to `paplay` in hook script |
