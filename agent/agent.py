"""
⑥ LLM 에이전트
- Claude API에 도구 목록을 주고, 도구 호출 요청이 오면 실행해 결과를 돌려주는 루프
- LLM을 호출하는 코드는 이 파일에만 둠
"""
import json

import anthropic
import pandas as pd

from config import AGENT_EFFORT, AGENT_MAX_TOKENS, AGENT_MAX_TURNS, AGENT_MODEL, OUTPUT_DIR
from review_tools import TOOL_SCHEMAS

# 에이전트에게 주는 역할 설명 (모든 거래에 똑같이 쓰임)
SYSTEM_PROMPT = """너는 신용카드 이상거래 탐지(FDS) 팀의 검토 보조 에이전트다.
RF 모델이 사기 확률 0.15 이상으로 걸러 낸 거래 1건을 받아, 사람 검토자가 빨리 판단할 수 있도록 근거를 모으고 요약한다.

역할 분담
- 판정(사기 확률)은 RF 모델이 이미 했다. 너는 점수를 다시 매기지 않는다.
- 너는 도구로 근거를 모으고, 근거끼리 맞는지 어긋나는지 해석해 권고와 우선순위를 정한다.
- 최종 결정과 라벨 확정은 사람이 한다. 너는 정답 라벨을 볼 수 없다.

조사 방법
- 도구 결과에 있는 사실만 근거로 쓴다. V1~V28은 익명화된 변수이므로 의미를 지어내지 않는다.
- 점수가 0.7 이상인 확신 구간은 짧게 확인해도 된다. 0.7 미만인 애매한 구간은 과거 실적상 사기와 정상이 반반이므로,
  SHAP 방향과 비슷한 과거 사례가 서로 같은 쪽을 가리키는지 꼼꼼히 본다.
- 조사가 끝나면 반드시 submit_review 도구로 결과를 제출한다. 요약과 근거는 한국어로 쓴다."""

# 최종 결과를 정해진 형식으로 받기 위한 "제출" 도구
# (도구 4개는 계산용, 이 도구는 LLM이 답을 내는 통로. 실행할 코드는 없음)
SUBMIT_TOOL = {
    "name": "submit_review",
    "description": "조사를 마친 뒤 검토 결과를 제출한다. 거래 1건당 마지막에 한 번만 호출한다.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "recommendation": {
                "type": "string",
                "enum": ["fraud_likely", "uncertain", "legit_likely"],
                "description": "fraud_likely=사기 가능성 높음(차단 검토), uncertain=추가 확인 필요, legit_likely=정상 가능성 높음",
            },
            "priority": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "사람이 먼저 봐야 하는 정도",
            },
            "summary": {"type": "string", "description": "검토자용 요약 2~3문장"},
            "key_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "권고의 핵심 근거 (도구 결과의 숫자를 포함해 2~4개)",
            },
        },
        "required": ["recommendation", "priority", "summary", "key_evidence"],
        "additionalProperties": False,
    },
}
ALL_TOOLS = TOOL_SCHEMAS + [SUBMIT_TOOL]


def _call_claude(client, messages):
    """Claude API 1회 호출 (설정은 config.py에서)"""
    return client.beta.messages.create(
        model=AGENT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        messages=messages,
        output_config={"effort": AGENT_EFFORT},
        # 안전 분류기가 요청을 거절하면 서버가 다른 모델로 자동으로 다시 시도 (거절로 멈추는 일 방지)
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def _failed(test_row, fraud_prob, n_tool_calls, reason):
    """에이전트가 결과를 못 낸 경우의 기록 (사람이 직접 봐야 하므로 uncertain / high로 둠)"""
    return {"test_row": test_row, "fraud_prob": fraud_prob, "recommendation": "uncertain",
            "priority": "high", "summary": f"에이전트 실패: {reason}", "key_evidence": [],
            "n_tool_calls": n_tool_calls, "error": reason}


def review_transaction(client, toolkit, test_row, fraud_prob):
    """거래 1건을 에이전트가 조사해 {요약, 권고, 근거} 딕셔너리로 반환"""
    # 대화 기록: 이 리스트에 질문 → 도구 요청 → 도구 결과 → ... 가 차례로 쌓임
    # (API는 기억이 없어서 매번 전체 기록을 다시 보냄)
    messages = [{
        "role": "user",
        "content": f"리뷰 큐 거래 test_row={test_row} (RF 사기 확률 {fraud_prob:.3f})를 조사하고 "
                   "submit_review로 결과를 제출해 줘.",
    }]
    n_tool_calls = 0

    for _ in range(AGENT_MAX_TURNS):                       # 최대 왕복 횟수 제한 (무한 루프 방지)
        response = _call_claude(client, messages)

        # 1. Claude의 응답을 기록에 그대로 추가 (텍스트만 꺼내지 말고 content 전체를 넣어야 함)
        messages.append({"role": "assistant", "content": response.content})

        # 2. 정상적으로 끝나지 않은 경우
        if response.stop_reason in ("refusal", "max_tokens"):
            return _failed(test_row, fraud_prob, n_tool_calls, response.stop_reason)

        # 3. 응답 안에서 도구 호출 요청만 골라냄
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            # 도구 없이 글로만 답하고 끝낸 경우 → 제출 도구를 쓰라고 한 번 더 요청
            messages.append({"role": "user", "content": "조사가 끝났다면 submit_review 도구로 결과를 제출해 줘."})
            continue

        # 4. 요청받은 도구를 실행하고 결과를 모음
        tool_results = []
        for block in tool_uses:
            if block.name == "submit_review":              # 최종 제출 → 이 거래는 끝
                return {"test_row": test_row, "fraud_prob": fraud_prob, **block.input,
                        "n_tool_calls": n_tool_calls, "error": None}

            n_tool_calls += 1
            try:
                result = toolkit.run(block.name, block.input)
                content, is_error = json.dumps(result, ensure_ascii=False), False
            except Exception as e:                         # 잘못된 행 번호 등 → 에러를 LLM에게 알려 스스로 고치게 함
                content, is_error = f"오류: {e}", True

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,                   # 어느 요청에 대한 결과인지 짝을 맞춤
                "content": content,
                "is_error": is_error,
            })

        # 5. 도구 결과를 모두 한 메시지로 돌려보냄 → 다음 반복에서 Claude가 이어서 판단
        messages.append({"role": "user", "content": tool_results})

    return _failed(test_row, fraud_prob, n_tool_calls, f"{AGENT_MAX_TURNS}회 안에 제출하지 않음")


def run_agent(queue, toolkit, limit=None):
    """리뷰 큐 전체(또는 앞에서 limit건)를 처리해 결과 표로 반환"""
    client = anthropic.Anthropic()                         # API 키는 환경 변수 ANTHROPIC_API_KEY에서 읽음
    targets = queue if limit is None else queue.head(limit)

    rows = []
    for i, item in enumerate(targets.itertuples(index=False), start=1):
        test_row, fraud_prob = int(item.test_row), float(item.fraud_prob)
        try:
            result = review_transaction(client, toolkit, test_row, fraud_prob)
        except anthropic.AuthenticationError:
            raise                                          # 키 문제는 모든 거래에서 똑같이 실패하므로 바로 멈춤
        except anthropic.APIError as e:                    # 네트워크·서버 오류 → 이 거래만 실패로 기록하고 계속
            result = _failed(test_row, fraud_prob, 0, f"API 오류: {type(e).__name__}")

        rows.append(result)
        print(f"[{i}/{len(targets)}] row {test_row} (p={fraud_prob:.3f}) → "
              f"{result['recommendation']} / {result['priority']} (도구 {result['n_tool_calls']}회)")

    # 결과 표: 우선순위 → 점수 순으로 정렬 (사람이 위에서부터 검토)
    results = pd.DataFrame(rows)
    results["priority_rank"] = results["priority"].map({"high": 0, "medium": 1, "low": 2})
    results = results.sort_values(["priority_rank", "fraud_prob"], ascending=[True, False]).drop(columns="priority_rank")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = results.assign(key_evidence=results["key_evidence"].apply(" | ".join))   # 리스트 → 문자열 (CSV 저장용)
    saved.to_csv(OUTPUT_DIR / "agent_results.csv", index=False, encoding="utf-8-sig")

    return results.reset_index(drop=True)
