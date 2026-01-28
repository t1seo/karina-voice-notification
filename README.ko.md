# Karina Voice Notification Generator

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/platform-Linux%20|%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-CUDA%2012.0%2B%20|%20Apple%20Silicon-green" alt="GPU">
  <img src="https://img.shields.io/badge/TTS-Qwen3--TTS%201.7B-orange" alt="TTS Model">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <a href="README.md"><img src="https://img.shields.io/badge/README-English-blue" alt="English"></a>
</p>

<p align="center">
  <img src="assets/karina.jpg" alt="Karina" width="800">
</p>

AI 음성 복제 기술을 사용하여 **어떤 목소리로든** Claude Code 알림음을 생성합니다.

> **어떤 음성이든 OK!** 좋아하는 아이돌, 성우, 스트리머, 심지어 본인 목소리도 YouTube 영상에서 복제 가능!

---

## 작동 원리

이 프로젝트는 **음성 복제 파이프라인**을 사용하여 커스텀 알림음을 생성합니다. 깨끗한 음성이 담긴 YouTube 영상을 주면, 해당 목소리를 복제하여 알림 문구를 말하게 합니다.

### 파이프라인 개요

```
YouTube 영상 → 오디오 추출 → 음성 구간 선택 → 전사 → 음성 복제 → 알림음
```

| 단계 | 수행 작업 | 이유 |
|------|----------|------|
| **1. 오디오 다운로드** | yt-dlp로 YouTube에서 오디오 추출 | 어떤 소스에서든 원본 음성 확보 |
| **2. 세그먼트 분할** | 30초 간격으로 15초 클립 생성 | 음악/노이즈 없는 깨끗한 구간 찾기 |
| **3. 세그먼트 선택** | 인터랙티브 메뉴로 최적 클립 선택 | 사람 귀가 알고리즘보다 깨끗한 오디오를 잘 구분함 |
| **4. 전사** | Whisper large-v3로 음성을 텍스트로 변환 | TTS가 음성 패턴을 매칭하려면 참조 텍스트 필요 |
| **5. 음성 복제** | Qwen3-TTS 1.7B로 새 음성 생성 | 목소리를 복제하여 커스텀 알림 문구를 말하게 함 |

### 음성 복제 기술

파이프라인은 짧은 오디오 샘플에서 목소리를 복제할 수 있는 최신 TTS 모델 **Qwen3-TTS 1.7B**를 사용합니다. 참조 오디오의 다음 요소를 분석합니다:
- 톤과 피치 패턴
- 말하는 리듬과 속도
- 목소리 특성과 음색

그런 다음 같은 사람이 커스텀 문구를 말하는 것처럼 들리는 새 음성을 생성합니다.

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

## 좋은 음성 샘플 선택하기

> **중요**: 음성 복제 품질은 전적으로 소스 오디오에 달려 있습니다.

**좋은 소스** (깨끗하고 분리된 음성):
- 인터뷰 영상
- 단독 발화 장면
- 비하인드 토킹 영상
- 팟캐스트 녹음

**피해야 할 것** (품질 저하 원인):
- 뮤직비디오
- 배경음악이 있는 클립
- 시끄러운 환경
- 여러 명이 말하는 영상

참조 오디오는 **5-15초**의 깨끗한 음성이어야 합니다.

---

## 생성되는 알림

파이프라인은 Claude Code 이벤트에 대한 알림음을 생성합니다:

| 이벤트 | 재생 시점 |
|--------|----------|
| `permission_prompt` | Claude가 위험한 명령 실행 전 허락을 구할 때 |
| `idle_prompt` | 작업 완료 후 대기 중일 때 |
| `auth_success` | 인증 성공 시 |
| `elicitation_dialog` | 사용자 입력이 필요할 때 |

각 알림 유형마다 **10개의 음성 변형**이 생성되어 랜덤으로 재생됩니다.

---

## 커스터마이징

### 커스텀 알림 문구

`notification_lines.json`을 수정하여 복제된 목소리가 말할 내용을 변경합니다:

```json
{
  "permission_prompt": [
    {"text": "원하는 문구를 여기에 입력하세요", "filename": "permission_prompt_1.wav"}
  ]
}
```

---

## Claude Code 연동

알림음을 생성한 후 Claude Code에 복사합니다:

```bash
# 오디오 파일 복사
mkdir -p ~/.claude/sounds
cp output/notifications/*/*.wav ~/.claude/sounds/

# Hook 스크립트 설치
mkdir -p ~/.claude/hooks
cp src/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

그런 다음 `~/.claude/settings.json`에 hook 설정을 추가합니다. 자세한 설정 방법은 [CLAUDE.md](CLAUDE.md)를 참조하세요.

---

## 프로젝트 구조

```
project-karina-voice/
├── src/
│   ├── pipeline.py              # 메인 파이프라인 오케스트레이터
│   ├── download_audio.py        # YouTube 오디오 추출
│   ├── transcribe.py            # Whisper 전사
│   ├── generate_notifications.py # Qwen3-TTS 음성 복제
│   └── claude_notification_hook.py # Claude Code 훅
├── output/
│   ├── raw/                     # 다운로드된 오디오
│   ├── clean/                   # 선택된 음성 세그먼트
│   └── notifications/           # 생성된 알림음
├── notification_lines.json      # 커스터마이징 가능한 문구
└── pixi.toml                    # 의존성
```

---

## 문제 해결

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
