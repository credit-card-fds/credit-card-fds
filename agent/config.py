"""
① 설정 파일
- 경로, 모델 설정, 운영 임계값, 비용 가정을 한곳에 모음
- 다른 파일에서는 `from config import DATA_DIR` 처럼 가져다 씀
"""
from pathlib import Path

# ── 1. 경로 ──────────────────────────────────────────────
# __file__  : 지금 이 파일(config.py)의 경로
# .resolve(): 상대 경로를 절대 경로로 바꿈
# .parent   : 그 파일이 들어 있는 폴더
AGENT_ROOT = Path(__file__).resolve().parent             # .../credit-card-fds/agent
FDS_ROOT = AGENT_ROOT.parent                              # .../credit-card-fds (과제 저장소 최상위)

# 데이터는 과제 쪽 eda_and_split.ipynb가 만든 분할 데이터를 그대로 읽음 (복사하지 않음)
DATA_DIR = FDS_ROOT / "data" / "processed"                # train/validation/test_processed.csv 위치

MODEL_DIR = AGENT_ROOT / "models"                         # 학습한 RF 모델 저장 위치 (*.joblib은 Git 제외)
MODEL_PATH = MODEL_DIR / "rf_final.joblib"                # 모델 파일 경로
OUTPUT_DIR = AGENT_ROOT / "outputs"                       # 리뷰 큐, 에이전트 결과 저장 위치 (*.csv는 Git 제외)

# 과제 노트북들이 만든 결과표 (app.py 모델 비교 탭에서 읽기만 함, 노트북 실행 후 생성)
REPORT_TABLE_DIR = FDS_ROOT / "reports" / "tables"

# ── 2. 모델 설정 ──────────────────────────────────────────
RANDOM_STATE = 42                                         # 재현성: 과제 노트북과 같은 값

# 과제 final_report.ipynb의 최종 모델과 똑같은 설정이어야 같은 리뷰 큐가 나옴
RF_PARAMS = {
    "n_estimators": 200,                                  # 트리 200개
    "class_weight": "balanced",                           # 사기(소수 클래스)에 더 큰 가중치
    "random_state": RANDOM_STATE,                         # 트리를 만들 때의 무작위성 고정
    "n_jobs": -1,                                         # CPU 코어 전부 사용
}

# ── 3. 운영 기준 ─────────────
# Validation에서 "Recall ≥ 0.80이면서 검토 건수 최소" 규칙으로 고른 임계값
# 사기 확률이 이 값 이상이면 리뷰 큐에 올림 (사람이 검토할 대상)
FINAL_THRESHOLD = 0.15

# 과제 final_report의 비용 가정 (도구 4 우선순위 점수에서 사용)
FRAUD_LOSS_PER_CASE = 122      # 놓친 사기 1건당 평균 손실 (EUR)
REVIEW_COST_PER_CASE = 5       # 사람이 1건 검토하는 비용 (EUR)

# ── 4. 재현성 기준 ────────────────────────────────────────
# 과제 final_report에서 RF @0.15가 Test에서 걸러 낸 건수 (evaluate.py 1부에서 비교)
EXPECTED_QUEUE_SIZE = 68       # 리뷰 큐 전체 건수
EXPECTED_QUEUE_FRAUD = 57      # 그중 실제 사기 건수

# 점수가 이 값 이상이면 "확신 구간", 미만이면 "애매한 구간" (PLAN.md 1단계에서 확인한 경계)
HIGH_SCORE_CUTOFF = 0.7

# ── 5. 에이전트 (LLM) 설정 ─────────────────────────────────
AGENT_MODEL = "claude-opus-5-5"   # 사용할 Claude 모델
AGENT_EFFORT = "medium"           # 생각 깊이: low / medium / high / xhigh / max
AGENT_MAX_TOKENS = 16000          # 응답 1번의 최대 길이 (토큰)
AGENT_MAX_TURNS = 10              # 거래 1건당 LLM 왕복 최대 횟수 (무한 루프 방지)
