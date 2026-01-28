# Karina Voice Notification - Claude Code 가이드

이 프로젝트는 aespa 카리나 음성으로 Claude Code 알림음을 생성합니다.

## 빠른 설정 (이미 생성된 음성 사용)

### 1. 음성 파일 복사

```bash
mkdir -p ~/.claude/sounds
cp output/*/*.wav ~/.claude/sounds/
```

### 2. Hook 스크립트 설치 (권장)

> **Note**: Claude Code의 `matcher` 기반 Notification hook이 안정적으로 작동하지 않아서, 통합 Python 스크립트 방식을 사용합니다. 이 방식은 `notification_type` 필드를 파싱하여 해당 소리를 재생합니다.

```bash
mkdir -p ~/.claude/hooks
cp src/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

`~/.claude/settings.json`의 `hooks` 섹션에 추가:

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

## Hook 유형 및 음성 파일

| Hook Matcher | 발생 시점 | 음성 파일 | 대사 |
|--------------|----------|----------|------|
| `permission_prompt` | 위험한 명령 실행 전 허락 요청 | `permission_prompt_1.wav` | 잠깐만요! 이거 실행해도 괜찮을까요? 허락해주세요~ |
| | | `permission_prompt_2.wav` | 잠시만요, 이 작업을 하려면 허락이 필요해요~ |
| `idle_prompt` | 작업 완료 후 응답 대기 | `idle_prompt_1.wav` | 다 끝났어요! 결과 한번 확인해주세요~ |
| | | `idle_prompt_2.wav` | 작업이 완료되었어요, 한번 봐주시겠어요? |
| `auth_success` | 인증 성공 | `auth_success_1.wav` | 인증이 완료되었어요! 도와주셔서 정말 고마워요~ |
| `elicitation_dialog` | 사용자 입력 필요 | `elicitation_dialog_1.wav` | 여기에 입력이 필요해요! 작성해주시겠어요? |

## 대사 변경하기

### 1. notification_lines.json 수정

```json
{
  "permission_prompt": [
    {"text": "원하는 새 대사를 입력하세요", "filename": "permission_prompt_1.wav"}
  ]
}
```

### 2. 음성 재생성

```bash
# GPU 환경 필요
pixi run python src/pipeline.py --skip-download
```

### 3. 새 음성 복사

```bash
cp output/*/*.wav ~/.claude/sounds/
```

## 플랫폼별 재생 명령어

| 플랫폼 | 명령어 |
|--------|--------|
| macOS | `afplay ~/.claude/sounds/파일명.wav` |
| Linux (ALSA) | `aplay ~/.claude/sounds/파일명.wav` |
| Linux (PulseAudio) | `paplay ~/.claude/sounds/파일명.wav` |

## 새 음성 생성하기 (GPU 필요)

```bash
# 1. YouTube에서 음성 샘플 다운로드 및 생성
pixi run python src/pipeline.py "https://youtube.com/watch?v=..."

# 2. 이미 다운로드된 샘플로 재생성
pixi run python src/pipeline.py --skip-download
```

## 주의사항

- Qwen3-TTS 레퍼런스 오디오: 5-15초 권장
- 깨끗하고 노이즈 없는 음성 샘플 필요
- GPU (CUDA 12.0+ 또는 Apple Silicon MPS) 환경에서만 생성 가능
