"""
⑤ 에이전트 도구
- LLM이 골라서 호출하는 순수 파이썬 함수 모음 (LLM 없이 단독 실행 가능)
- 계산은 여기서, 해석과 요약은 LLM이 맡음
- y_test(정답)는 받지 않음. y_train은 "과거 사례"로 사용
"""
import numpy as np
import shap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import FRAUD_LOSS_PER_CASE, HIGH_SCORE_CUTOFF, REVIEW_COST_PER_CASE


class ReviewToolkit:
    """도구 4개가 공유하는 데이터(학습 데이터, 모델, SHAP 설명기)를 한 번만 준비해 두는 클래스"""

    def __init__(self, model, X_train, y_train, X_test):
        # 1. 받은 재료 보관 (y_test는 일부러 받지 않음 → 도구가 정답을 볼 방법이 없음)
        self.model = model
        self.X_test = X_test
        self.y_train = y_train
        self.feature_names = list(X_train.columns)

        # 2. Test 전체 사기 확률을 한 번만 계산해 둠 (도구마다 다시 계산하지 않도록)
        self.test_probs = model.predict_proba(X_test)[:, 1]

        # 3. 도구 1용: 정상 거래(Train, Class=0)의 특성별 값을 정렬해 둠 → 백분위를 빠르게 구함
        normal = X_train[y_train == 0]
        self.normal_sorted = {c: np.sort(normal[c].values) for c in self.feature_names}

        # 4. 도구 2용: SHAP 설명기 (트리 모델 전용, 계산이 빠름)
        self.explainer = shap.TreeExplainer(model)

        # 5. 도구 3용: 비슷한 과거 거래 찾기
        #    특성마다 범위가 달라서(Hour는 0~23, V1~V28은 대략 -5~5) 표준화한 뒤 거리를 잼
        self.scaler = StandardScaler().fit(X_train)
        self.neighbors = NearestNeighbors().fit(self.scaler.transform(X_train))

    # ── 공통 도우미 ─────────────────────────────────────────
    def _row(self, test_row):
        """행 번호를 검사하고 해당 거래 1행(DataFrame)을 돌려줌"""
        test_row = int(test_row)
        if not 0 <= test_row < len(self.X_test):
            # 에러 메시지는 LLM에게 그대로 전달되므로, 고칠 방법을 알 수 있게 씀
            raise ValueError(f"test_row는 0 ~ {len(self.X_test) - 1} 사이여야 합니다 (받은 값: {test_row})")
        return self.X_test.iloc[[test_row]]                # [[ ]] : 1행짜리 DataFrame으로 꺼냄 (모델 입력 형태 유지)

    def _normal_percentile(self, feature, value):
        """정상 거래 중 이 값보다 작은 거래의 비율 (%)"""
        arr = self.normal_sorted[feature]
        # searchsorted: 정렬된 배열에서 value가 들어갈 위치 = value보다 작은 값의 개수
        return round(100 * np.searchsorted(arr, value) / len(arr), 1)

    # ── 도구 1 ─────────────────────────────────────────────
    def get_transaction(self, test_row):
        """도구 1: 거래 특성값과 사기 확률, 정상 거래 분포 대비 백분위"""
        row = self._row(test_row).iloc[0]                   # 1행 DataFrame → Series

        features = {}
        for c in self.feature_names:
            pct = self._normal_percentile(c, row[c])
            features[c] = {"value": round(float(row[c]), 4), "normal_percentile": pct}

        # 정상 거래에서 거의 안 나오는 값 (하위·상위 1%) → 조사 단서
        unusual = [c for c, f in features.items()
                   if c != "Hour" and (f["normal_percentile"] <= 1 or f["normal_percentile"] >= 99)]

        return {
            "test_row": int(test_row),
            "fraud_prob": round(float(self.test_probs[int(test_row)]), 4),
            "features": features,
            "unusual_features": unusual,
            "note": "Amount는 log1p+RobustScaler 변환값이라 유로 금액이 아님. 백분위로 크기를 판단할 것. "
                    "V1~V28은 익명화된 PCA 변수라 의미를 추측하지 말 것.",
        }

    # ── 도구 2 ─────────────────────────────────────────────
    def explain_score(self, test_row, top_k=5):
        """도구 2: SHAP 기여도 상위 변수 (사기 쪽 / 정상 쪽 방향 포함)"""
        row = self._row(test_row)
        sv = self.explainer(row)                           # SHAP 값 계산 (1행)

        # RF는 [정상, 사기] 클래스별 값을 주므로 사기(1) 쪽만 사용 (과제 final_report와 같은 처리)
        values = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
        base = sv.base_values[0, 1] if np.ndim(sv.base_values) == 2 else sv.base_values[0]

        # 영향이 큰 순서(절댓값 기준)로 top_k개
        order = np.argsort(-np.abs(values))[:int(top_k)]
        contributions = [{
            "feature": self.feature_names[i],
            "value": round(float(row.iloc[0, i]), 4),
            "shap": round(float(values[i]), 4),
            "direction": "사기 쪽" if values[i] > 0 else "정상 쪽",
        } for i in order]

        return {
            "test_row": int(test_row),
            "base_value": round(float(base), 4),            # 아무 정보가 없을 때의 평균 예측값
            "fraud_prob": round(float(self.test_probs[int(test_row)]), 4),
            "top_contributions": contributions,
            "note": "base_value + 모든 shap 합 = fraud_prob",
        }

    # ── 도구 3 ─────────────────────────────────────────────
    def find_similar_cases(self, test_row, k=5):
        """도구 3: Train에서 특성이 가장 비슷한 과거 거래 k건과 그 라벨"""
        row = self._row(test_row)
        dist, idx = self.neighbors.kneighbors(self.scaler.transform(row), n_neighbors=int(k))

        cases = [{
            "train_row": int(i),
            "distance": round(float(d), 3),
            "label": "사기" if self.y_train[i] == 1 else "정상",   # 과거 사례의 확정 라벨 (Train이라 사용 가능)
        } for d, i in zip(dist[0], idx[0])]

        n_fraud = sum(c["label"] == "사기" for c in cases)
        return {
            "test_row": int(test_row),
            "k": int(k),
            "similar_fraud_count": n_fraud,
            "cases": cases,
            "note": "distance는 표준화한 30개 특성 공간의 거리. 작을수록 비슷함. "
                    "참고로 Train 전체의 사기 비율은 약 0.17%.",
        }

    # ── 도구 4 ─────────────────────────────────────────────
    def priority_score(self, test_row):
        """도구 4: 기대 손실 기반 검토 우선순위 점수"""
        self._row(test_row)                                # 행 번호 검사만
        p = float(self.test_probs[int(test_row)])

        # 기대 손실 = 사기일 확률 × 사기 1건 손실 / 검토하면 이 손실을 막을 수 있음
        expected_loss = p * FRAUD_LOSS_PER_CASE
        net_benefit = expected_loss - REVIEW_COST_PER_CASE  # 검토로 얻는 순이익 (양수면 검토할 가치 있음)

        return {
            "test_row": int(test_row),
            "fraud_prob": round(p, 4),
            "expected_loss_eur": round(expected_loss, 2),
            "review_cost_eur": REVIEW_COST_PER_CASE,
            "net_benefit_eur": round(net_benefit, 2),
            "score_band": "확신 구간" if p >= HIGH_SCORE_CUTOFF else "애매한 구간",
            "note": f"비용은 과제 가정값 (사기 1건 {FRAUD_LOSS_PER_CASE} EUR, 검토 1건 {REVIEW_COST_PER_CASE} EUR). "
                    f"애매한 구간(<{HIGH_SCORE_CUTOFF})은 과거 실적상 사기와 정상이 반반 정도.",
        }

    # ── LLM이 고른 도구 실행 ────────────────────────────────
    def run(self, name, args):
        """도구 이름과 입력(dict)을 받아 해당 메서드를 실행 (agent.py에서 사용)"""
        tools = {
            "get_transaction": self.get_transaction,
            "explain_score": self.explain_score,
            "find_similar_cases": self.find_similar_cases,
            "priority_score": self.priority_score,
        }
        if name not in tools:
            raise ValueError(f"알 수 없는 도구: {name}")
        return tools[name](**args)                         # ** : {"test_row": 3} → test_row=3


# LLM에게 알려 줄 도구 설명서 (이름, 설명, 입력 형식)
# - description은 LLM이 "언제 이 도구를 쓸지" 판단하는 유일한 근거라 구체적으로 씀
# - strict + additionalProperties=False: LLM이 형식에 맞는 입력만 보내도록 보장
TOOL_SCHEMAS = [
    {
        "name": "get_transaction",
        "description": "거래 1건의 특성값 30개와 RF 사기 확률을 가져온다. 각 특성이 정상 거래 분포에서 "
                       "몇 백분위인지, 정상 거래에서 드문 값(상·하위 1%)이 무엇인지도 알려 준다.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"test_row": {"type": "integer", "description": "리뷰 큐의 거래 행 번호"}},
            "required": ["test_row"],
            "additionalProperties": False,
        },
    },
    {
        "name": "explain_score",
        "description": "SHAP으로 RF 점수를 설명한다. 점수를 사기 쪽으로 올린 변수와 정상 쪽으로 내린 변수를 "
                       "영향이 큰 순서로 돌려준다.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "test_row": {"type": "integer", "description": "리뷰 큐의 거래 행 번호"},
                "top_k": {"type": "integer", "description": "보여 줄 변수 개수 (보통 5)"},
            },
            "required": ["test_row", "top_k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_similar_cases",
        "description": "학습 데이터에서 이 거래와 가장 비슷한 과거 거래 k건과 그 확정 라벨(사기/정상)을 찾는다. "
                       "비슷한 과거 사례 중 사기가 많으면 사기 근거가 된다.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "test_row": {"type": "integer", "description": "리뷰 큐의 거래 행 번호"},
                "k": {"type": "integer", "description": "찾을 과거 거래 수 (보통 5~10)"},
            },
            "required": ["test_row", "k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "priority_score",
        "description": "사기 확률과 비용 가정으로 기대 손실과 검토 순이익(EUR)을 계산하고, "
                       "확신 구간인지 애매한 구간인지 알려 준다.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"test_row": {"type": "integer", "description": "리뷰 큐의 거래 행 번호"}},
            "required": ["test_row"],
            "additionalProperties": False,
        },
    },
]


# ── 단독 테스트: python review_tools.py (API 키 불필요) ───────
if __name__ == "__main__":
    import json

    from data import get_model, load_train_test
    from queue_builder import build_review_queue

    X_train, y_train, X_test, _ = load_train_test()
    model = get_model(X_train, y_train)
    queue = build_review_queue(model, X_test)
    toolkit = ReviewToolkit(model, X_train, y_train, X_test)

    # 애매한 구간에서 첫 거래 하나로 도구 4개를 모두 실행해 봄
    row = int(queue[queue["fraud_prob"] < HIGH_SCORE_CUTOFF].iloc[0]["test_row"])
    print("테스트 거래:", row)
    for name, args in [("get_transaction", {"test_row": row}),
                       ("explain_score", {"test_row": row, "top_k": 5}),
                       ("find_similar_cases", {"test_row": row, "k": 5}),
                       ("priority_score", {"test_row": row})]:
        result = toolkit.run(name, args)
        if name == "get_transaction":                      # 특성 30개는 길어서 일부만 출력
            result = {**result, "features": dict(list(result["features"].items())[:3])}
        print(f"\n== {name}\n{json.dumps(result, ensure_ascii=False, indent=1)}")
