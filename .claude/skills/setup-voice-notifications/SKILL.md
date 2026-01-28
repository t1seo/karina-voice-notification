---
name: setup-voice-notifications
description: Karina 음성 알림 시스템을 설치합니다. Claude Code 알림음 설정, 카리나 음성 설치, notification hook 설정 시 사용합니다.
disable-model-invocation: true
allowed-tools: Bash, Read, Edit, Write
---

# Karina Voice Notification 설치

이 스킬은 Karina 음성 알림 시스템을 Claude Code에 설치합니다.

## 설치 단계

### Step 1: 사운드 파일 복사

```bash
mkdir -p ~/.claude/sounds
cp "$ARGUMENTS/output/notifications/"*/*.wav ~/.claude/sounds/
```

> `$ARGUMENTS`가 없으면 현재 디렉토리(`$(pwd)`)를 사용합니다.

### Step 2: Hook 스크립트 설치

```bash
mkdir -p ~/.claude/hooks
cp "$ARGUMENTS/src/claude_notification_hook.py" ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

### Step 3: settings.json 설정

`~/.claude/settings.json`을 읽고 아래 hooks 설정을 추가/병합합니다. 기존 hooks 섹션이 있으면 Notification과 Stop hooks를 병합합니다:

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

### Step 4: 설치 확인

설치된 파일을 확인하고 결과를 보고합니다:

```bash
echo "Sound files:"
ls -la ~/.claude/sounds/*.wav 2>/dev/null || echo "No sound files found"

echo "Hook script:"
ls -la ~/.claude/hooks/claude_notification_hook.py 2>/dev/null || echo "Hook script not found"
```

## 알림 유형

| Type | Sound Files | Trigger |
|------|-------------|---------|
| `permission_prompt` | permission_prompt_*.wav | 권한 요청 시 |
| `idle_prompt` | idle_prompt_*.wav | 작업 완료 / Stop 이벤트 |
| `auth_success` | auth_success_*.wav | 인증 성공 시 |
| `elicitation_dialog` | elicitation_dialog_*.wav | 사용자 입력 필요 시 |

**참고:** 각 알림 유형에 여러 사운드 파일이 있으면 (`_1.wav`, `_2.wav` 등) 매번 랜덤으로 재생됩니다.
