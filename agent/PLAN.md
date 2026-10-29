# 검토 보조 에이전트 계획

## 목표
RF 모델(임계값 0.15)이 Test에서 걸러 낸 **리뷰 큐 68건**에 대해, 에이전트가 근거를 모아 요약하고 검토 우선순위를 정한다.

| 역할 | 담당 |
|---|---|
| 판정 (사기 확률) | RF 모델 |
| 근거 수집·요약, 우선순위 | 에이전트 (도구 + LLM) |
| 최종 결정, 라벨 확정 | 사람 |

## 구조
```
credit-card-fds/agent/
├── CLAUDE.md          작업 규칙 (Claude Code가 자동으로 읽음)
├── PLAN.md            이 문서
├── requirements.txt   필요한 패키지
├── config.py          ① 경로, RF 설정, 임계값, 비용 가정
├── data.py            ② 데이터 로드, RF 학습·저장·불러오기
├── queue_builder.py   ③ 리뷰 큐 만들기 (라벨 없음)
├── review_tools.py    ④ 에이전트 도구 4개
├── agent.py           ⑤ LLM 에이전트 루프
├── evaluate.py        ⑥ 재현성 확인, 권고 vs 실제 라벨 평가
├── main.py            ⑦ 실행 진입점
├── models/            rf_final.joblib (실행 시 생성)
└── outputs/           리뷰 큐, 에이전트 결과 (실행 시 생성)
```

## 데이터 흐름
```
../data/processed/*.csv (과제 저장소)
        │
   ② data.py ──────────► X_train, y_train, X_test, y_test, rf_model
        │
   ③ queue_builder.py ─► 리뷰 큐 68건 (행 번호 + 점수만)
        │
   ⑤ agent.py ◄──────── ④ review_tools.py
        │                  get_transaction / explain_score /
        │                  find_similar_cases / priority_score
        ▼
   에이전트 권고 (요약, 권고, 근거)
        │
   ⑥ evaluate.py ◄───── y_test (여기서만 사용)
```

## 1단계에서 확인한 사실 (이전 시도)
- 리뷰 큐 68건 중 실제 사기 57건 / 정상 11건 (`final_report`와 동일)
- 점수 0.7 이상 53건 중 사기 50건 → 짧은 확인으로 충분
- 점수 0.7 미만 15건 중 사기 7건 / 정상 8건 → **에이전트가 깊게 조사해야 할 구간**
- processed 데이터의 `Amount`는 log1p + RobustScaler 변환값 → 원본(`data/raw`)이 없어 유로 금액은 복원하지 않음. 도구 1은 정상 거래 대비 백분위로 크기를 알려 줌 (변환이 단조 증가라 백분위는 원래 금액과 같음)

## 한계 (미탐) — 사후 분석은 하지 않기로 함
- Test 사기 72건 중 미탐 15건. 점수: 0.10, 0.06, 0.02, 0.01, 0.005×3, 0×8 → 8건이 0점
- 임계값을 낮춰도 거의 못 잡음: 0.05 → 경보 83건 / 사기 59건, 0.01 → 경보 438건 / 사기 61건
- 결론: 임계값 문제가 아니라 **익명화된 특성 30개로는 정상과 구별되지 않는 사기**. 위치·고객 패턴 같은 정보가 필요함
- 에이전트도 같은 데이터를 보므로 이 한계를 넘지 못함 (경보 68건 검토 보조에만 집중)

## 다음 단계 (이번 범위 밖)
- 위도·경도 등 위치 정보가 있는 데이터셋(사용자 보유)으로 같은 구조를 테스트해 이번 결과와 비교

## 체크리스트
- [x] 0. 뼈대 생성 (CLAUDE.md, PLAN.md, 빈 .py 파일)
- [x] 1. `config.py` (+ 재현성 기준, 에이전트 설정 추가)
- [x] 2. `data.py` — 과제와 같은 RF 재현
- [x] 3. `queue_builder.py` — 68건 큐 생성
- [x] 4. `evaluate.py` 1부 — 재현성 확인 (68건 / 사기 57건 일치)
- [x] 5. `review_tools.py` — 도구 4개, 단독 테스트 통과
- [x] 6. `agent.py` — 도구 호출 루프 (가짜 응답으로 루프 검증 완료)
- [ ] 6-1. API 키 준비 후 실제 실행 (`python main.py --agent --limit 3`)
- [x] 7. `evaluate.py` 2부 — 권고 평가 (가짜 권고로 동작 확인)
- [ ] 7-1. 실제 권고 평가, 특히 0.7 미만 15건 (`--only-uncertain`)
- [x] 8. `main.py` 완성, `requirements.txt` 정리
