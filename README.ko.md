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

## 주요 기능

- **음성 복제** - Qwen3-TTS 1.7B를 사용하여 YouTube 영상에서 목소리 복제
- **BGM 제거 (권장)** - Demucs AI로 배경음악에서 보컬 자동 분리
- **오디오 정규화** - 클리핑 방지를 위한 자동 오디오 정규화
- **오디오 후처리** - 선택적 오디오 향상 (노이즈 제거, EQ, 컴프레서, 음량 정규화)
- **다국어 지원** - TTS 10개 언어 지원 (한국어, 영어, 중국어, 일본어 등)
- **인터랙티브 메뉴** - 키보드로 쉽게 조작하는 메뉴 시스템
- **출력 정리** - 새 파이프라인 시작 시 이전 출력 삭제 옵션
- **코드 품질** - Pre-commit hooks와 ruff 린터/포맷터

---

## 작동 원리

### 파이프라인 개요

```
YouTube 영상
     ↓
오디오 다운로드 (yt-dlp)
     ↓
[선택] BGM 제거 (Demucs AI)
     ↓
세그먼트 분할 (15초 클립)
     ↓
세그먼트 선택 (인터랙티브)
     ↓
전사 (Whisper large-v3)
     ↓
음성 복제 (Qwen3-TTS 1.7B)
     ↓
[선택] 후처리
     ↓
알림음
```

| 단계 | 수행 작업 | 기술 |
|------|----------|------|
| **오디오 다운로드** | YouTube에서 오디오 추출 | yt-dlp |
| **BGM 제거** | 배경음악에서 보컬 분리 | Demucs (Meta AI) |
| **세그먼트 분할** | 15초 클립으로 분할 | pydub |
| **전사** | 음성을 텍스트로 변환 | Whisper large-v3 |
| **음성 복제** | 복제된 목소리로 새 음성 생성 | Qwen3-TTS 1.7B |
| **후처리** | 오디오 품질 향상 | noisereduce, pedalboard, pyloudnorm |

### 후처리 파이프라인

```
Raw TTS 출력
     ↓
노이즈 제거 (spectral gating)
     ↓
보이스 EQ (80Hz 하이패스, 12kHz 로우패스)
     ↓
다이내믹스 (컴프레서 + 리미터)
     ↓
음량 정규화 (-14 LUFS)
     ↓
최종 출력
```

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

```bash
# 저장소 클론
git clone https://github.com/t1seo/karina-voice-notification.git
cd project-karina-voice

# 의존성 설치
pixi install
pixi run install-deps-mac    # macOS
pixi run install-deps-linux  # Linux

# 파이프라인 실행
pixi run pipeline
```

---

## 지원 언어

### TTS (음성 생성)
| 언어 | 코드 |
|------|------|
| 한국어 | korean |
| 영어 | english |
| 중국어 | chinese |
| 일본어 | japanese |
| 프랑스어 | french |
| 독일어 | german |
| 스페인어 | spanish |
| 이탈리아어 | italian |
| 포르투갈어 | portuguese |
| 러시아어 | russian |

### 전사 (Whisper)
위의 모든 언어를 자동 감지 또는 수동 선택으로 지원합니다.

---

## 좋은 음성 샘플 선택하기

> **중요**: 음성 복제 품질은 전적으로 소스 오디오에 달려 있습니다.

**좋은 소스** (깨끗하고 분리된 음성):
- 인터뷰 영상
- 단독 발화 장면
- 비하인드 토킹 영상
- 팟캐스트 녹음

**피해야 할 것** (품질 저하 원인):
- 뮤직비디오 (BGM 제거 옵션 사용)
- 시끄러운 환경
- 여러 명이 말하는 영상

참조 오디오는 **5-15초**의 깨끗한 음성이어야 합니다.

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

### 후처리 설정

`src/post_process.py`에서 조정:
- `target_lufs`: 목표 음량 (-14 LUFS 기본값, 스트리밍 표준)
- `denoise_strength`: 노이즈 제거 강도 (0.0-1.0)
- `highpass_freq`: 저음 럼블 제거 (80Hz 기본값)
- `lowpass_freq`: 고주파 노이즈 제거 (12kHz 기본값)

---

## Claude Code 연동

알림음을 생성한 후 Claude Code에 복사합니다:

```bash
# 오디오 파일 복사
mkdir -p ~/.claude/sounds
cp output/notifications/*/*.wav ~/.claude/sounds/

# Hook 스크립트 설치
mkdir -p ~/.claude/hooks
cp .claude/skills/setup-notifications/scripts/claude_notification_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/claude_notification_hook.py
```

그런 다음 `~/.claude/settings.json`에 hook 설정을 추가합니다:

```json
{
  "hooks": {
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
  }
}
```

---

## 문제 해결

### 음성 품질이 안 좋아요
- 배경음악 없는 깨끗한 음성 샘플 사용
- 파이프라인 메뉴에서 BGM 제거 (Demucs) 활성화
- 참조 오디오가 5-15초인지 확인
- 후처리를 활성화하여 출력 품질 향상

### 후처리가 작동 안 해요
```bash
# 올바른 pixi 환경에 의존성 설치
pixi run -e mac pip install noisereduce pedalboard pyloudnorm  # macOS
pixi run -e linux pip install noisereduce pedalboard pyloudnorm  # Linux
```

### Hook에서 소리가 안 나요
- `~/.claude/sounds/`에 오디오 파일이 있는지 확인
- Hook 스크립트에 실행 권한이 있는지 확인
- `~/.claude/hooks/hook_debug.log`에서 에러 확인

---

## 라이선스

MIT License
