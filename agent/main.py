"""
⑦ 실행 진입점
- `python main.py` 한 번으로 데이터 준비 → 리뷰 큐 → 에이전트 → 평가를 순서대로 실행
- 에이전트(유료 API 호출)는 --agent를 붙였을 때만 실행

사용 예
  python main.py                              # 1~4단계만 (API 키 불필요)
  python main.py --agent --limit 3            # 점수 높은 3건만 에이전트 실행 (시험용)
  python main.py --agent --only-uncertain     # 애매한 구간(<0.7) 15건만
  python main.py --agent                      # 68건 전체
"""
import argparse

from config import HIGH_SCORE_CUTOFF
from data import get_model, load_train_test
from evaluate import check_reproducibility, evaluate_recommendations
from queue_builder import build_review_queue
from review_tools import ReviewToolkit


def parse_args():
    """명령줄 옵션 읽기"""
    parser = argparse.ArgumentParser(description="FDS 검토 보조 에이전트")
    parser.add_argument("--agent", action="store_true", help="에이전트 실행 (Claude API 호출, 비용 발생)")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N건만 처리")
    parser.add_argument("--only-uncertain", action="store_true", help=f"점수 {HIGH_SCORE_CUTOFF} 미만 거래만 처리")
    return parser.parse_args()


def main():
    args = parse_args()

    # ① ~ ② 데이터와 모델
    X_train, y_train, X_test, y_test = load_train_test()
    model = get_model(X_train, y_train)

    # ③ 리뷰 큐 (라벨 없음)
    queue = build_review_queue(model, X_test)
    print(f"리뷰 큐: {len(queue)}건")

    # ④ 재현성 확인 (y_test는 evaluate에만 전달)
    check_reproducibility(queue, y_test)

    if not args.agent:
        print("\n에이전트를 실행하려면 --agent를 붙이세요 (예: python main.py --agent --limit 3)")
        return

    # ⑤ 도구 준비 (y_test는 넘기지 않음)
    toolkit = ReviewToolkit(model, X_train, y_train, X_test)

    # ⑥ 에이전트 실행 (LLM 코드는 agent.py에만 있으므로 여기서 불러옴 → --agent 없이는 anthropic 불필요)
    from agent import run_agent
    targets = queue[queue["fraud_prob"] < HIGH_SCORE_CUTOFF] if args.only_uncertain else queue
    results = run_agent(targets, toolkit, limit=args.limit)

    # ⑦ 권고 평가 (여기서 처음으로 정답과 비교)
    print()
    evaluate_recommendations(results, y_test)


if __name__ == "__main__":
    main()
