---
name: setup-voice-notifications
description: Install Karina voice notification system for Claude Code. Use when setting up notification sounds, installing voice alerts, or configuring notification hooks.
disable-model-invocation: true
allowed-tools: Bash, Read, Edit, Write
---

# Karina Voice Notification Setup

This skill installs the Karina voice notification system for Claude Code.

## Installation Steps

### Step 1: Copy sound files

```bash
mkdir -p ~/.claude/sounds
cp "$ARGUMENTS/output/notifications/"*/*.wav ~/.claude/sounds/
```

> If `$ARGUMENTS` is empty, use current directory (`$(pwd)`).

### Step 2: Install hook script

```bash
mkdir -p ~/.claude/hooks
cp "$ARGUMENTS/src/claude_notification_hook.py" ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

### Step 3: Configure settings.json

Read `~/.claude/settings.json` and add/merge the hooks configuration below. If a hooks section already exists, merge the Notification and Stop hooks:

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

### Step 4: Verify installation

Check installed files and report results:

```bash
echo "Sound files:"
ls -la ~/.claude/sounds/*.wav 2>/dev/null || echo "No sound files found"

echo "Hook script:"
ls -la ~/.claude/hooks/claude_notification_hook.py 2>/dev/null || echo "Hook script not found"
```

## Notification Types

| Type | Sound Files | Trigger |
|------|-------------|---------|
| `permission_prompt` | permission_prompt_*.wav | Permission request |
| `idle_prompt` | idle_prompt_*.wav | Task complete / Stop event |
| `auth_success` | auth_success_*.wav | Authentication success |
| `elicitation_dialog` | elicitation_dialog_*.wav | User input required |

**Note:** When multiple sound files exist for a notification type (e.g., `_1.wav`, `_2.wav`), a random one is played each time.
