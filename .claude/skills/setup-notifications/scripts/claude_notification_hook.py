#!/usr/bin/env python3
"""Claude Code notification hook with sound playback.

This script plays Karina voice notifications for Claude Code events.

Usage:
    1. Copy to ~/.claude/hooks/claude_notification_hook.py
    2. Add to ~/.claude/settings.json hooks section
"""

import glob
import json
import os
import random
import subprocess
import sys
from datetime import datetime

# Debug log file
DEBUG_LOG = os.path.expanduser("~/.claude/hooks/hook_debug.log")

# Sound directory
SOUND_DIR = os.path.expanduser("~/.claude/sounds")


def debug_log(msg: str):
    """Write debug message to log file."""
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


def get_random_sound(notification_type: str):
    """Get a random sound file for the given notification type."""
    pattern = os.path.join(SOUND_DIR, f"{notification_type}_*.wav")
    sound_files = glob.glob(pattern)
    if sound_files:
        return random.choice(sound_files)
    return None


def play_notification_sound(notification_type: str):
    """Play a random sound for the given notification type."""
    sound_file = get_random_sound(notification_type)
    if sound_file:
        try:
            # Use nohup to ensure sound plays even after hook exits
            subprocess.Popen(
                f'nohup afplay "{sound_file}" &',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            debug_log(f"Playing sound for {notification_type}: {sound_file}")
        except Exception as e:
            debug_log(f"Error playing sound: {e}")


def main():
    debug_log("=== Hook started ===")

    # Read hook input from stdin
    try:
        stdin_data = sys.stdin.read()
        hook_input = json.loads(stdin_data) if stdin_data else {}
    except json.JSONDecodeError as e:
        debug_log(f"JSON decode error: {e}")
        print(json.dumps({"continue": True}))
        sys.exit(0)

    event = hook_input.get("hook_event_name", "")
    debug_log(f"Event: {event}")

    # Handle Stop event - play completion sound
    if event == "Stop":
        play_notification_sound("idle_prompt")

    # Handle Notification event - play sound based on type
    elif event == "Notification":
        notification_type = hook_input.get("notification_type", "")
        if notification_type:
            play_notification_sound(notification_type)

    # Output continue (don't block)
    debug_log("Hook completed")
    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
