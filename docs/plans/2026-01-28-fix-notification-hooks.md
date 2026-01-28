# Fix Claude Code Notification Hooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Slack 메시지와 카리나 음성 알림이 모든 Notification 이벤트에서 정상 작동하도록 수정

**Architecture:** 통합 Python 스크립트(`claude_slack_hook.py`)에서 Slack 전송과 소리 재생을 모두 처리. matcher 기반 별도 hook은 Claude Code 버그로 작동하지 않으므로, catchall hook에서 `notification_type` 필드를 파싱하여 해당 소리 재생.

**Tech Stack:** Python 3, afplay (macOS), Slack API

---

### Task 1: Slack Hook에 소리 재생 호출 추가

**Files:**
- Modify: `/Users/cillian/.claude/hooks/claude_slack_hook.py:396-405`

**Step 1: Notification 이벤트 핸들러에 소리 재생 추가**

현재 코드:
```python
    # Handle Notification event
    elif event == "Notification":
        message = hook_input.get("message", "")

        blocks = build_notification_blocks(
            project_name, user_request, message, session_id
        )

        fallback = f"⏳ Claude 입력 대기 - {project_name}"
        result = send_slack_message(fallback, blocks)
        debug_log(f"Slack send result: {result}")
```

수정 코드:
```python
    # Handle Notification event
    elif event == "Notification":
        message = hook_input.get("message", "")
        notification_type = hook_input.get("notification_type", "")

        # Play notification sound based on type
        if notification_type:
            play_notification_sound(notification_type)

        blocks = build_notification_blocks(
            project_name, user_request, message, session_id
        )

        fallback = f"⏳ Claude 입력 대기 - {project_name}"
        result = send_slack_message(fallback, blocks)
        debug_log(f"Slack send result: {result}")
```

**Step 2: 스크립트 직접 테스트**

Run:
```bash
echo '{"hook_event_name":"Notification","notification_type":"elicitation_dialog","message":"test"}' | python3 ~/.claude/hooks/claude_slack_hook.py
```

Expected: 소리 재생 + Slack 메시지 전송

**Step 3: 변경 확인**

Run: `tail -5 ~/.claude/hooks/hook_debug.log`
Expected: `Playing sound for elicitation_dialog` 로그 출력

---

### Task 2: Stop 이벤트에도 완료 소리 추가 (선택)

**Files:**
- Modify: `/Users/cillian/.claude/hooks/claude_slack_hook.py:381-393`

**Step 1: idle_prompt 소리를 Stop 이벤트에도 재생**

수정 코드 (Stop 핸들러 시작 부분에 추가):
```python
    # Handle Stop event
    if event == "Stop":
        # Play completion sound
        play_notification_sound("idle_prompt")

        assistant_msg = get_last_assistant_message(transcript_path)
        ...
```

---

### Task 3: Claude Code에서 실제 테스트

**Step 1: AskUserQuestion 도구로 elicitation_dialog 테스트**

Claude가 AskUserQuestion 도구 사용 시 `elicitation_dialog` 소리가 나야 함

**Step 2: 60초 대기 후 idle_prompt 테스트**

Claude 응답 완료 후 60초 대기 시 `idle_prompt` 소리가 나야 함

**Step 3: 성공 확인**

Expected:
- Slack 메시지: ✅
- 카리나 소리: ✅

---

## 진단 정보

### 현재 상태 (debug 로그 기반)
- `Slack send result: True` - Slack은 정상 작동
- `Playing sound` 로그 없음 - 소리 함수 미호출
- `notification_type: "idle_prompt"` 수신됨 - 데이터는 있음

### 왜 matcher 기반 hook이 안 되나?
Claude Code의 Notification hook에서 `matcher` 필드가 제대로 작동하지 않는 것으로 보임. catchall hook(matcher 없음)만 실행됨. 이는 Claude Code 버그일 가능성 있음.

### 해결책
catchall hook에서 `notification_type` 필드를 직접 파싱하여 소리 재생
