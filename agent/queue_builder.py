"""
③ 리뷰 큐 만들기
- Test 거래에 사기 확률을 매기고, 운영 임계값 이상인 거래만 큐로 보냄
- 큐에는 행 번호와 점수만 담음 (정답 라벨은 넣지 않음)
"""
import pandas as pd

from config import FINAL_THRESHOLD, OUTPUT_DIR


def build_review_queue(model, X_test):
    """임계값 이상 거래의 (test_row, fraud_prob) 표를 점수 높은 순으로 반환"""
    # 1. Test 전체에 사기 확률 매기기
    #    predict_proba는 [정상 확률, 사기 확률] 2열을 돌려주므로 [:, 1]로 사기 확률만 꺼냄
    probs = model.predict_proba(X_test)[:, 1]

    # 2. 표로 만들기: test_row는 X_test 안의 "위치 번호" (0, 1, 2, ...)
    #    도구들이 X_test.iloc[test_row]로 같은 거래를 찾을 때 이 번호를 씀
    scores = pd.DataFrame({"test_row": range(len(X_test)), "fraud_prob": probs})

    # 3. 임계값 이상만 남기고, 점수 높은 순으로 정렬 (사람이 위에서부터 검토)
    queue = scores[scores["fraud_prob"] >= FINAL_THRESHOLD]
    queue = queue.sort_values("fraud_prob", ascending=False).reset_index(drop=True)

    # 4. 결과 저장 (CSV는 .gitignore로 Git에서 제외됨)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    queue.to_csv(OUTPUT_DIR / "review_queue.csv", index=False, encoding="utf-8-sig")

    return queue


# ── 단독 테스트: python queue_builder.py ─────────────────────
if __name__ == "__main__":
    from data import get_model, load_train_test

    X_train, y_train, X_test, _ = load_train_test()       # _ : y_test는 여기서 쓰지 않음 (라벨 숨김)
    model = get_model(X_train, y_train)
    queue = build_review_queue(model, X_test)

    print(f"리뷰 큐: {len(queue)}건 (임계값 {FINAL_THRESHOLD} 이상)")
    print(queue.head(10).round(4).to_string(index=False))
