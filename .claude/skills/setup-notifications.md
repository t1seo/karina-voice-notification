# Setup Karina Voice Notifications

This skill sets up Karina voice notifications for Claude Code.

## Usage

Run `/setup-notifications` to install the voice notification system.

## What it does

1. Copies notification sound files to `~/.claude/sounds/`
2. Installs the notification hook script to `~/.claude/hooks/`
3. Configures `~/.claude/settings.json` with the necessary hooks

## Instructions

Execute the following steps:

### Step 1: Copy sound files

```bash
mkdir -p ~/.claude/sounds
cp "$(pwd)/output/notifications/"*/*.wav ~/.claude/sounds/
```

### Step 2: Install hook script

```bash
mkdir -p ~/.claude/hooks
cp "$(pwd)/src/claude_notification_hook.py" ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

### Step 3: Configure settings.json

Read `~/.claude/settings.json` and add/merge the following hooks configuration. If hooks section already exists, merge the Notification and Stop hooks:

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

### Step 4: Verify and report

List the installed files and confirm setup is complete:

```bash
echo "Sound files:"
ls -la ~/.claude/sounds/*.wav

echo "Hook script:"
ls -la ~/.claude/hooks/claude_notification_hook.py
```

Report success with the list of notification types:

| Type | Sound | Trigger |
|------|-------|---------|
| `permission_prompt` | permission_prompt_1.wav | Permission request |
| `idle_prompt` | idle_prompt_1.wav | Task complete / Stop |
| `auth_success` | auth_success_1.wav | Auth success |
| `elicitation_dialog` | elicitation_dialog_1.wav | User input needed |
