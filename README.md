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
| permission_prompt | 잠깐요! 이거 해도 될까요? | `permission_prompt_1.wav` |
| | 허락이 필요해요~ | `permission_prompt_2.wav` |
| idle_prompt | 다 했어요! 확인해주세요~ | `idle_prompt_1.wav` |
| | 끝났어요, 봐주세요! | `idle_prompt_2.wav` |
| auth_success | 인증 완료! 고마워요~ | `auth_success_1.wav` |
| elicitation_dialog | 여기 입력이 필요해요! | `elicitation_dialog_1.wav` |

생성된 파일: `output/notifications/`

## Claude Code 설정

```bash
# 알림음 복사
mkdir -p ~/.claude/sounds
cp output/notifications/*/*.wav ~/.claude/sounds/

# hooks 설정 (~/.claude/settings.json)
{
  "hooks": {
    "notification": [
      {
        "event": "permission_prompt",
        "command": "afplay ~/.claude/sounds/permission_prompt_1.wav"
      },
      {
        "event": "idle_prompt",
        "command": "afplay ~/.claude/sounds/idle_prompt_1.wav"
      }
    ]
  }
}
```

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
├── output/notifications/     # 생성된 알림음
├── notification_lines.json   # 알림 문구 설정
├── pixi.toml                 # 의존성 설정
└── pixi.lock
```

## Customization

알림 문구 수정: `notification_lines.json` 편집

```json
{
  "permission_prompt": [
    {"text": "새로운 문구", "filename": "permission_prompt_1.wav"}
  ]
}
```
