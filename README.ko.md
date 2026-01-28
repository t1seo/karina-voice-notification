# Karina Voice Notification Generator

<p align="center">
  <img src="assets/karina.jpg" alt="Karina" width="800">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/platform-Linux%20|%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-CUDA%2012.0%2B%20|%20Apple%20Silicon-green" alt="GPU">
  <img src="https://img.shields.io/badge/TTS-Qwen3--TTS%201.7B-orange" alt="TTS Model">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

AI 음성 복제 기술을 사용하여 **aespa 카리나 음성**으로 Claude Code 알림음을 생성합니다.

> **이게 뭐예요?** YouTube에서 음성 샘플을 추출하고, 전사한 뒤 Qwen3-TTS를 사용해 Claude Code가 사용자의 주의가 필요할 때 재생되는 커스텀 알림음을 생성하는 파이프라인입니다.

---

## 주요 기능

- **음성 복제**: Qwen3-TTS 1.7B를 사용해 카리나 음성으로 알림 생성
- **크로스 플랫폼**: Linux (CUDA)와 macOS (Apple Silicon) 지원
- **인터랙티브 메뉴**: 화살표 키로 쉽게 세그먼트 선택
- **커스터마이징**: 알림 문구를 수정하고 언제든 재생성 가능

---

## 요구 사항

### Linux (NVIDIA GPU)
- CUDA 지원 NVIDIA GPU (A100 권장)
- CUDA 12.0+
- [pixi](https://pixi.sh) 패키지 매니저

### macOS (Apple Silicon)
- M1/M2/M3/M4 칩 탑재 Mac
- 64GB RAM 권장 (최소 32GB)
- [pixi](https://pixi.sh) 패키지 매니저

---

## 빠른 시작

### 1. pixi 설치

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc  # Mac에서는 ~/.zshrc
```

### 2. 클론 및 설정

```bash
git clone https://github.com/t1seo/project-karina-voice.git
cd project-karina-voice

# Linux
pixi install -e linux
pixi run -e linux install-deps-linux

# macOS
pixi install -e mac
pixi run -e mac install-deps-mac
```

### 3. 파이프라인 실행

```bash
# Linux
pixi run -e linux pipeline

# macOS
pixi run -e mac pipeline
```

인터랙티브 메뉴가 다음 과정을 안내합니다:
1. YouTube에서 오디오 다운로드
2. 깨끗한 음성 구간 선택
3. 오디오 전사
4. 알림음 생성

---

## 좋은 음성 샘플 선택하기

> **중요**: 최상의 결과를 위해 **배경음악 없이 목소리만** 있는 YouTube 영상을 선택하세요. 깨끗하고 분리된 음성이 음악이나 효과음이 섞인 오디오보다 훨씬 좋은 음성 복제 결과를 만들어냅니다.

**좋은 소스:**
- 인터뷰 영상
- 단독 발화 장면
- 비하인드 토킹 영상

**피해야 할 것:**
- 뮤직비디오
- 배경음악이 있는 클립
- 시끄러운 환경

---

## 파이프라인 단계

| 단계 | 설명 |
|------|------|
| 1. GPU 확인 | CUDA/MPS 사용 가능 여부 감지 |
| 2. 오디오 다운로드 | yt-dlp로 YouTube에서 오디오 추출 |
| 3. 세그먼트 분할 | 30초 간격으로 15초 클립 생성 |
| 4. 세그먼트 선택 | 깨끗한 음성 구간 선택 (수동 또는 자동) |
| 5. 전사 | Whisper로 음성을 텍스트로 변환 |
| 6. TTS 설정 | Qwen3-TTS 1.7B 모델 다운로드 |
| 7. 생성 | 음성 복제로 알림음 생성 |

---

## 생성되는 알림

| 유형 | 문구 | 파일 |
|------|------|------|
| permission_prompt | 잠깐만요! 이거 실행해도 괜찮을까요? 허락해주세요~ | `permission_prompt_1.wav` |
| idle_prompt | 다 끝났어요! 결과 한번 확인해주세요~ | `idle_prompt_1.wav` |
| auth_success | 인증이 완료되었어요! 도와주셔서 정말 고마워요~ | `auth_success_1.wav` |
| elicitation_dialog | 여기에 입력이 필요해요! 작성해주시겠어요? | `elicitation_dialog_1.wav` |

출력 파일은 `output/notifications/`에 저장됩니다.

---

## Claude Code 연동

### 1. 오디오 파일 복사

```bash
mkdir -p ~/.claude/sounds
cp output/notifications/*/*.wav ~/.claude/sounds/
```

### 2. Hook 스크립트 설치

```bash
mkdir -p ~/.claude/hooks
cp src/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

### 3. Claude Code 설정

`~/.claude/settings.json`에 추가:

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

### 알림 유형

| 유형 | 재생 시점 |
|------|----------|
| `permission_prompt` | Claude가 위험한 명령 실행 전 허락을 구할 때 |
| `idle_prompt` | 작업 완료 후 60초 이상 대기할 때 |
| `auth_success` | 인증 성공 시 |
| `elicitation_dialog` | 사용자 입력이 필요할 때 |
| Stop 이벤트 | Claude 응답 완료 시 |

---

## 커스터마이징

### 알림 문구 변경

`notification_lines.json` 수정:

```json
{
  "permission_prompt": [
    {"text": "원하는 문구를 여기에 입력하세요", "filename": "permission_prompt_1.wav"}
  ]
}
```

### 오디오 재생성

```bash
pixi run -e mac pipeline --skip-download
```

---

## 프로젝트 구조

```
project-karina-voice/
├── src/
│   ├── pipeline.py              # 메인 파이프라인
│   ├── device_utils.py          # GPU 감지
│   ├── download_audio.py        # YouTube 다운로드
│   ├── transcribe.py            # Whisper 전사
│   ├── generate_notifications.py # 음성 생성
│   └── claude_notification_hook.py # Claude 훅
├── output/
│   ├── raw/                     # 다운로드된 오디오
│   ├── clean/                   # 처리된 세그먼트
│   └── notifications/           # 생성된 알림음
├── notification_lines.json      # 문구 설정
└── pixi.toml                    # 의존성
```

---

## 문제 해결

### "No module named 'xxx'"
의존성 설치 실행:
```bash
pixi run -e mac install-deps-mac  # 또는 install-deps-linux
```

### 음성 품질이 안 좋아요
- 배경음악 없는 깨끗한 음성 샘플 사용
- 레퍼런스 오디오가 5-15초인지 확인
- 원본 영상에서 다른 구간 시도

### Hook에서 소리가 안 나요
- `~/.claude/sounds/`에 오디오 파일이 있는지 확인
- Hook 스크립트에 실행 권한이 있는지 확인
- `~/.claude/hooks/hook_debug.log`에서 에러 확인

---

## 라이선스

MIT License
