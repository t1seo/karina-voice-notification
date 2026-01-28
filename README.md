# Karina Voice Notification Generator

Claude Code 알림음을 aespa 카리나 음성으로 생성하는 파이프라인.

## Requirements

- GPU: NVIDIA A100 (권장) 또는 CUDA 지원 GPU
- CUDA 12.0+
- pixi 패키지 매니저

## Setup

```bash
# pixi 설치
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# 의존성 설치
cd karina-tts-notification
pixi install
```

## Usage

```bash
# 기본 YouTube URL로 실행
pixi run python scripts/pipeline.py

# 다른 YouTube URL로 실행
pixi run python scripts/pipeline.py "https://youtube.com/watch?v=..."

# 이미 다운로드된 오디오 사용
pixi run python scripts/pipeline.py --skip-download
```

## Pipeline Steps

1. **GPU Check** - CUDA 및 GPU 확인
2. **Download Audio** - YouTube에서 오디오 추출 (yt-dlp)
3. **Split Segments** - 30초 간격으로 15초 클립 생성
4. **Select Segment** - 깨끗한 음성 구간 선택
5. **Transcribe** - faster-whisper large-v3로 전사
6. **Setup TTS** - Qwen3-TTS 1.7B 모델 다운로드
7. **Generate** - 음성 복제로 알림음 생성

## Generated Notifications

| 유형 | 문구 | 파일 |
|------|------|------|
| permission_prompt | 잠깐만요! 이거 실행해도 괜찮을까요? 허락해주세요~ | `permission_prompt_1.wav` |
| | 잠시만요, 이 작업을 하려면 허락이 필요해요~ | `permission_prompt_2.wav` |
| idle_prompt | 다 끝났어요! 결과 한번 확인해주세요~ | `idle_prompt_1.wav` |
| | 작업이 완료되었어요, 한번 봐주시겠어요? | `idle_prompt_2.wav` |
| auth_success | 인증이 완료되었어요! 도와주셔서 정말 고마워요~ | `auth_success_1.wav` |
| elicitation_dialog | 여기에 입력이 필요해요! 작성해주시겠어요? | `elicitation_dialog_1.wav` |

생성된 파일: `output/`

---

## Claude Code 알림 설정

### 1. 음성 파일 복사

```bash
mkdir -p ~/.claude/sounds
cp output/*/*.wav ~/.claude/sounds/
```

### 2. Hook 스크립트 설치 (권장)

> **Note**: Claude Code의 `matcher` 기반 Notification hook이 안정적으로 작동하지 않는 경우가 있어, 통합 Python 스크립트 방식을 권장합니다.

```bash
# hook 스크립트 복사
mkdir -p ~/.claude/hooks
cp scripts/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

`~/.claude/settings.json`의 `hooks` 섹션에 추가:

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

이 방식은 `notification_type` 필드를 파싱하여 해당 소리를 자동으로 재생합니다.

### 3. 알림 유형 설명

| notification_type | 발생 시점 | 소리 파일 |
|-------------------|----------|----------|
| `permission_prompt` | Claude가 위험한 명령 실행 전 허락을 구할 때 | `permission_prompt_1.wav` |
| `idle_prompt` | 작업 완료 후 60초 이상 대기할 때 | `idle_prompt_1.wav` |
| `auth_success` | 인증 성공 시 | `auth_success_1.wav` |
| `elicitation_dialog` | 사용자 입력이 필요할 때 (AskUserQuestion) | `elicitation_dialog_1.wav` |
| Stop 이벤트 | Claude 응답 완료 시 | `idle_prompt_1.wav` |

### 4. Linux에서 사용

`scripts/claude_notification_hook.py`의 `afplay` 명령어를 수정:

```python
# macOS
subprocess.Popen(["afplay", sound_file], ...)

# Linux (ALSA)
subprocess.Popen(["aplay", sound_file], ...)

# Linux (PulseAudio)
subprocess.Popen(["paplay", sound_file], ...)
```

### 5. Slack 연동 (선택)

Slack 알림도 함께 받으려면 `~/.claude/settings.json`의 `env` 섹션에 추가:

```json
{
  "env": {
    "SLACK_BOT_TOKEN": "xoxb-your-token",
    "SLACK_CHANNEL_ID": "C0123456789"
  }
}
```

---

## 알림 메시지 커스터마이징

### 문구 변경

`notification_lines.json` 파일 수정:

```json
{
  "permission_prompt": [
    {"text": "새로운 문구를 여기에 입력하세요", "filename": "permission_prompt_1.wav"}
  ]
}
```

### 새 음성 생성

문구 수정 후 파이프라인 재실행:

```bash
pixi run python scripts/pipeline.py --skip-download
```

---

## Project Structure

```
karina-tts-notification/
├── scripts/
│   ├── pipeline.py           # 메인 파이프라인
│   ├── download_audio.py     # YouTube 다운로드
│   ├── extract_segment.py    # 오디오 분할
│   ├── transcribe.py         # Whisper 전사
│   ├── setup_qwen_tts.py     # TTS 모델 설정
│   └── generate_notifications.py  # 알림음 생성
├── assets/
│   ├── raw/                  # 원본 오디오
│   ├── clean/                # 정제된 오디오
│   └── transcripts/          # 전사 결과
├── models/                   # TTS 모델 (자동 다운로드)
├── output/                   # 생성된 알림음
├── notification_lines.json   # 알림 문구 설정
├── pixi.toml                 # 의존성 설정
└── pixi.lock
```
