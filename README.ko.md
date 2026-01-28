# Karina Voice Notification Generator

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/platform-Linux%20|%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-CUDA%2012.0%2B%20|%20Apple%20Silicon-green" alt="GPU">
  <img src="https://img.shields.io/badge/TTS-Qwen3--TTS%201.7B-orange" alt="TTS Model">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <br>
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue" alt="English"></a>
</p>

<p align="center">
  <img src="assets/karina.jpg" alt="Karina" width="800">
</p>

AI 음성 복제 기술을 사용하여 **aespa 카리나 음성**으로 Claude Code 알림음을 생성합니다.

> **이게 뭐예요?** Claude Code가 사용자의 주의가 필요할 때 재생되는 커스텀 알림음입니다 - 권한 요청, 작업 완료 등.

> **어떤 음성이든 OK!** 기본값은 카리나 음성이지만, **어떤 YouTube 영상이든** 음성 소스로 사용할 수 있어요. 좋아하는 아이돌, 성우, 스트리머, 심지어 본인 목소리도 복제 가능!

---

## 빠른 설정 (사전 생성된 오디오 사용)

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

---

## 알림 유형

| 유형 | 재생 시점 |
|------|----------|
| `permission_prompt` | Claude가 위험한 명령 실행 전 허락을 구할 때 |
| `idle_prompt` | 작업 완료 후 대기 중일 때 |
| `auth_success` | 인증 성공 시 |
| `elicitation_dialog` | 사용자 입력이 필요할 때 |
| Stop 이벤트 | Claude 응답 완료 시 |

각 알림 유형마다 10개의 음성 변형이 있어 랜덤으로 재생됩니다.

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

---

## 새 오디오 생성하기 (GPU 필요)

### 요구 사항

**Linux (NVIDIA GPU)**
- CUDA 지원 NVIDIA GPU (A100 권장)
- CUDA 12.0+
- [pixi](https://pixi.sh) 패키지 매니저

**macOS (Apple Silicon)**
- M1/M2/M3/M4 칩 탑재 Mac
- 64GB RAM 권장 (최소 32GB)
- [pixi](https://pixi.sh) 패키지 매니저

### 좋은 음성 샘플 선택하기

> **중요**: 최상의 결과를 위해 **배경음악 없이 목소리만** 있는 YouTube 영상을 선택하세요.

**좋은 소스:**
- 인터뷰 영상
- 단독 발화 장면
- 비하인드 토킹 영상

**피해야 할 것:**
- 뮤직비디오
- 배경음악이 있는 클립
- 시끄러운 환경

---

## 문제 해결

### Hook에서 소리가 안 나요
- `~/.claude/sounds/`에 오디오 파일이 있는지 확인
- Hook 스크립트에 실행 권한이 있는지 확인
- `~/.claude/hooks/hook_debug.log`에서 에러 확인

### 음성 품질이 안 좋아요
- 배경음악 없는 깨끗한 음성 샘플 사용
- 레퍼런스 오디오가 5-15초인지 확인
- 원본 영상에서 다른 구간 시도

---

## 프로젝트 구조

```
project-karina-voice/
├── src/
│   ├── pipeline.py              # 메인 파이프라인
│   ├── claude_notification_hook.py # Claude 훅
│   └── ...
├── output/
│   └── notifications/           # 생성된 알림음
├── notification_lines.json      # 문구 설정
└── pixi.toml                    # 의존성
```

---

## 라이선스

MIT License
