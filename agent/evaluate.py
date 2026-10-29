"""
④·⑦ 평가
- 정답 라벨(y_test)을 쓰는 유일한 파일
- 1부: 리뷰 큐가 과제 final_report와 같은지 확인 (재현성)
- 2부: 에이전트 권고가 실제 라벨과 얼마나 맞는지 평가
"""
import pandas as pd

from config import EXPECTED_QUEUE_FRAUD, EXPECTED_QUEUE_SIZE, HIGH_SCORE_CUTOFF, OUTPUT_DIR


def check_reproducibility(queue, y_test):
    """큐가 68건이고 그중 사기 57건인지 확인 (다르면 모델/데이터가 바뀐 것)"""
    # queue["test_row"]는 X_test 안의 위치 번호 → 같은 위치의 정답을 y_test에서 꺼냄
    labels = y_test[queue["test_row"].values]
    n_queue, n_fraud = len(queue), int(labels.sum())

    ok = (n_queue == EXPECTED_QUEUE_SIZE) and (n_fraud == EXPECTED_QUEUE_FRAUD)
    print(f"[재현성] 리뷰 큐 {n_queue}건 / 사기 {n_fraud}건 "
          f"(기준 {EXPECTED_QUEUE_SIZE}건 / {EXPECTED_QUEUE_FRAUD}건) → {'일치' if ok else '불일치 ⚠'}")
    if not ok:
        # 멈추지는 않고 경고만: 라이브러리 버전이 달라도 에이전트 실습 자체는 가능하므로
        print("  → 데이터를 다시 만들었거나 scikit-learn 버전이 다를 수 있습니다. "
              "models/ 폴더를 지우고 다시 실행해 보세요.")
    return ok


def evaluate_recommendations(results, y_test):
    """에이전트 권고와 실제 라벨 비교표 반환 (점수 구간별)"""
    # 1. 에이전트 결과에 정답과 점수 구간을 붙임 (원본 results는 건드리지 않도록 복사)
    df = results.copy()
    df["actual"] = y_test[df["test_row"].values]
    df["actual"] = df["actual"].map({1: "사기", 0: "정상"})
    df["score_band"] = df["fraud_prob"].apply(
        lambda p: f"확신 구간 (≥{HIGH_SCORE_CUTOFF})" if p >= HIGH_SCORE_CUTOFF else f"애매한 구간 (<{HIGH_SCORE_CUTOFF})"
    )

    # 2. 구간 × 권고 × 실제 라벨 교차표
    #    pd.crosstab: 두 열의 값 조합별 건수를 세어 표로 만듦
    table = pd.crosstab(
        [df["score_band"], df["recommendation"]], df["actual"],
        margins=True, margins_name="합계",               # 행·열 합계 추가
    )

    # 3. 핵심 지표 두 가지
    #    - 위험한 실수: 실제 사기인데 에이전트가 "정상 가능성 높음"이라고 한 건 (놓치면 손실)
    #    - 아낀 검토: 실제 정상인데 "정상 가능성 높음"이라고 한 건 (사람이 빨리 넘길 수 있음)
    legit_rec = df["recommendation"] == "legit_likely"
    dangerous = df[legit_rec & (df["actual"] == "사기")]
    saved = df[legit_rec & (df["actual"] == "정상")]

    print(f"[평가] 에이전트가 검토한 거래: {len(df)}건")
    print(table.to_string())
    print(f"\n- 위험한 실수 (사기를 정상으로 권고): {len(dangerous)}건 {dangerous['test_row'].tolist()}")
    print(f"- 아낀 검토 (정상을 정상으로 권고): {len(saved)}건 / 큐 안의 실제 정상 {int((df['actual'] == '정상').sum())}건")

    # 4. 거래별 상세 결과 저장 (정답이 붙은 표는 evaluate 결과로만 남김)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "agent_evaluation.csv", index=False, encoding="utf-8-sig")

    return table


# ── 단독 테스트: python evaluate.py (1부만, API 키 불필요) ────
if __name__ == "__main__":
    from data import get_model, load_train_test
    from queue_builder import build_review_queue

    X_train, y_train, X_test, y_test = load_train_test()
    model = get_model(X_train, y_train)
    queue = build_review_queue(model, X_test)
    check_reproducibility(queue, y_test)

    # 구간별로 나눠 보기 (PLAN.md "1단계에서 확인한 사실"과 같은지)
    labels = y_test[queue["test_row"].values]
    high = queue["fraud_prob"].values >= HIGH_SCORE_CUTOFF
    print(f"  확신 구간 {high.sum()}건 중 사기 {labels[high].sum()}건 / "
          f"애매한 구간 {(~high).sum()}건 중 사기 {labels[~high].sum()}건")
