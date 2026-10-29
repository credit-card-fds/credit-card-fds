"""
⑧ 시연 화면 (Streamlit)
- 실행: agent/ 폴더에서 `streamlit run app.py` → 브라우저에서 http://localhost:8501
- 탭 1 리뷰 큐: 거래를 고르면 도구 4개를 그 자리에서 실행하고, 저장된 에이전트 권고를 함께 보여 줌
- 탭 2 모델 비교: 과제 노트북이 만든 결과표(reports/tables)를 읽어 성능·비용을 비교
- 탭 3 에이전트 평가: 권고 vs 실제 라벨 (사이드바의 "정답 공개"를 켜야 보임)
- 탭 4 시스템 구조: 역할 분담과 한계
- y_test는 직접 읽지 않음. 정답은 evaluate.py가 만든 agent_evaluation.csv로만 봄
"""
import importlib.util
import os

import altair as alt
import pandas as pd
import streamlit as st

from config import (FINAL_THRESHOLD, FRAUD_LOSS_PER_CASE, HIGH_SCORE_CUTOFF, OUTPUT_DIR,
                    REPORT_TABLE_DIR, REVIEW_COST_PER_CASE)
from data import get_model, load_train_test
from queue_builder import build_review_queue
from review_tools import ReviewToolkit

# 페이지 설정은 다른 st 명령보다 먼저 한 번만 호출해야 함
st.set_page_config(page_title="FDS 검토 보조 에이전트", page_icon="💳", layout="wide")

# ── 1. 표시용 상수 ────────────────────────────────────────
# 색은 두 가지만 씀 (색약 검사 통과한 파랑/주황 조합)
FRAUD_COLOR = "#eb6834"   # 사기 쪽 근거, 비지도 모델, 놓친 사기 손실
LEGIT_COLOR = "#2a78d6"   # 정상 쪽 근거, 지도 모델, 검토 비용

# 코드 값 → 화면 표시 이름 (색 원만으로 구별하지 않도록 글자를 함께 붙임)
REC_LABEL = {"fraud_likely": "🔴 사기 가능성 높음", "uncertain": "🟡 추가 확인 필요", "legit_likely": "🟢 정상 가능성 높음"}
PRI_LABEL = {"high": "높음", "medium": "중간", "low": "낮음"}
PRI_RANK = {"high": 0, "medium": 1, "low": 2}


# ── 2. 데이터 준비 (처음 한 번만, 이후에는 캐시에서 꺼냄) ────────
@st.cache_resource(show_spinner="데이터와 모델을 준비하는 중... (처음 한 번만, 1분 정도)")
def load_toolkit():
    """리뷰 큐와 도구 상자 준비 (main.py ①~③, ⑤와 같은 순서)"""
    X_train, y_train, X_test, _ = load_train_test()       # _ : y_test는 앱에서 쓰지 않음 (라벨 숨김)
    model = get_model(X_train, y_train)
    queue = build_review_queue(model, X_test)
    toolkit = ReviewToolkit(model, X_train, y_train, X_test)
    return queue, toolkit


@st.cache_data(show_spinner="도구 4개를 실행하는 중...")
def run_tools(test_row):
    """거래 1건에 도구 4개를 실행 (같은 거래를 다시 고르면 캐시 결과를 씀)"""
    _, toolkit = load_toolkit()
    return {
        "transaction": toolkit.get_transaction(test_row),
        "explain": toolkit.explain_score(test_row, top_k=10),
        "similar": toolkit.find_similar_cases(test_row, k=10),
        "priority": toolkit.priority_score(test_row),
    }


def read_csv(path):
    """파일이 있으면 읽고, 없으면 None (노트북·에이전트를 아직 안 돌린 경우)"""
    return pd.read_csv(path) if path.exists() else None


def load_agent_results():
    """저장된 에이전트 권고 (agent.py가 저장한 형식)"""
    df = read_csv(OUTPUT_DIR / "agent_results.csv")
    if df is not None:
        # CSV 저장 때 " | "로 이어 붙인 근거 목록을 다시 리스트로 되돌림
        df["key_evidence"] = df["key_evidence"].fillna("").apply(lambda s: [x for x in s.split(" | ") if x])
    return df


def score_band(p):
    return "확신 구간" if p >= HIGH_SCORE_CUTOFF else "애매한 구간"


def agent_available():
    """에이전트 실행 조건: API 키 + anthropic 패키지"""
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and importlib.util.find_spec("anthropic") is not None


# ── 3. 공통 그리기 도우미 ─────────────────────────────────
def show_recommendation(result, actual=None):
    """에이전트 권고 카드 1장"""
    with st.container(border=True):
        st.markdown(f"#### {REC_LABEL.get(result['recommendation'], result['recommendation'])}")
        st.markdown(f"**검토 우선순위:** {PRI_LABEL.get(result['priority'], result['priority'])}")
        st.write(result["summary"])
        for ev in result["key_evidence"]:
            st.markdown(f"- {ev}")
        if actual is not None:
            st.markdown(f"**실제 라벨:** {'🔴 사기' if actual == '사기' else '🟢 정상'}")


def bar_chart(df, x, y, color=None, color_domain=None, color_range=None, x_title=None, y_title=None,
              sort="-x", height=None, tooltip=None, single_color=LEGIT_COLOR):
    """가로 막대그래프 (색 지정이 없으면 한 가지 색)"""
    enc_color = (alt.Color(f"{color}:N", scale=alt.Scale(domain=color_domain, range=color_range),
                           legend=alt.Legend(title=None, orient="top"))
                 if color else alt.value(single_color))
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X(f"{x}:Q", title=x_title),
        y=alt.Y(f"{y}:N", sort=sort, title=y_title),
        color=enc_color,
        tooltip=tooltip or [y, alt.Tooltip(f"{x}:Q", format=",.3f")],
    )
    return chart.properties(height=height or max(140, 34 * len(df)))


# ── 4. 사이드바 ───────────────────────────────────────────
queue, toolkit = load_toolkit()
results = load_agent_results()
evaluation = read_csv(OUTPUT_DIR / "agent_evaluation.csv")

with st.sidebar:
    st.header("리뷰 큐 필터")
    bands = st.multiselect("점수 구간", ["확신 구간", "애매한 구간"], default=["확신 구간", "애매한 구간"],
                           help=f"확신 구간: 사기 확률 {HIGH_SCORE_CUTOFF} 이상")
    recs = st.multiselect("에이전트 권고", list(REC_LABEL), default=list(REC_LABEL), format_func=REC_LABEL.get)
    pris = st.multiselect("우선순위", list(PRI_LABEL), default=list(PRI_LABEL), format_func=PRI_LABEL.get)
    st.divider()
    reveal = st.toggle("정답 공개", value=False,
                       help="에이전트는 정답을 모른 채 권고했습니다. 시연 마지막에 켜서 비교하세요.")
    if reveal and evaluation is None:
        st.warning("agent_evaluation.csv가 없습니다. evaluate.py로 평가를 먼저 실행하세요.")
    st.divider()
    st.caption(f"운영 임계값 {FINAL_THRESHOLD} · 비용 가정: 사기 1건 {FRAUD_LOSS_PER_CASE} EUR, 검토 1건 {REVIEW_COST_PER_CASE} EUR")

# 정답은 평가 결과 파일에서만 가져옴 (정답 공개를 켰을 때만)
actual_map = (dict(zip(evaluation["test_row"], evaluation["actual"]))
              if reveal and evaluation is not None else {})

# ── 5. 머리말과 핵심 숫자 ─────────────────────────────────
st.title("💳 신용카드 FDS 검토 보조 에이전트")
st.caption("판정은 RF 모델, 근거 수집·요약과 우선순위는 에이전트, 최종 결정은 사람이 합니다.")

n_high_band = int((queue["fraud_prob"] >= HIGH_SCORE_CUTOFF).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test 거래", f"{len(toolkit.X_test):,}건")
c2.metric(f"리뷰 큐 (사기 확률 ≥ {FINAL_THRESHOLD})", f"{len(queue)}건")
c3.metric("확신 구간 / 애매한 구간", f"{n_high_band} / {len(queue) - n_high_band}건")
if results is not None:
    n_pri_high = int((results["priority"] == "high").sum())
    if actual_map:
        hit = sum(actual_map.get(t) == "사기" for t in results.loc[results["priority"] == "high", "test_row"])
        c4.metric("우선순위 높음 중 실제 사기", f"{hit} / {n_pri_high}건", f"{hit / n_pri_high:.0%}", delta_color="off")
    else:
        c4.metric("우선순위 높음", f"{n_pri_high}건")

tab_queue, tab_models, tab_eval, tab_system = st.tabs(["📋 리뷰 큐", "📊 모델 비교", "✅ 에이전트 평가", "🧭 시스템 구조"])

# ── 6. 탭 1: 리뷰 큐 ──────────────────────────────────────
with tab_queue:
    # 6-1. 큐 + 권고를 합쳐 사람이 볼 순서(우선순위 → 점수)로 정렬
    view = queue.copy()
    view["구간"] = view["fraud_prob"].apply(score_band)
    if results is not None:
        view = view.merge(results[["test_row", "recommendation", "priority"]], on="test_row", how="left")
    else:
        view["recommendation"], view["priority"] = None, None
    view["rank"] = view["priority"].map(PRI_RANK).fillna(3)
    view = view.sort_values(["rank", "fraud_prob"], ascending=[True, False])

    # 6-2. 사이드바 필터 적용 (권고가 없는 행은 권고·우선순위 필터와 상관없이 남김)
    mask = view["구간"].isin(bands)
    if results is not None:
        mask &= view["recommendation"].isin(recs) & view["priority"].isin(pris)
    view = view[mask].reset_index(drop=True)

    table = pd.DataFrame({
        "거래 번호": view["test_row"],
        "사기 확률": view["fraud_prob"],
        "구간": view["구간"],
        "에이전트 권고": view["recommendation"].map(REC_LABEL).fillna("미검토"),
        "우선순위": view["priority"].map(PRI_LABEL).fillna("-"),
    })
    if actual_map:
        table["실제"] = view["test_row"].map(actual_map)

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader(f"리뷰 큐 {len(table)}건")
        st.caption("행을 클릭하면 오른쪽에 상세 근거가 나옵니다.")
        event = st.dataframe(
            table, hide_index=True, width="stretch", height=560,
            on_select="rerun", selection_mode="single-row", key="queue_table",
            column_config={"사기 확률": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)},
        )

    with right:
        if table.empty:
            st.info("필터 조건에 맞는 거래가 없습니다.")
        else:
            # 6-3. 선택한 거래 (선택이 없으면 맨 위 거래)
            picked = event.selection.rows[0] if event.selection.rows else 0
            t = int(view.loc[picked, "test_row"])
            tools = run_tools(t)
            p = tools["priority"]["fraud_prob"]

            st.subheader(f"거래 {t} · 사기 확률 {p:.3f} ({score_band(p)})")

            # 6-4. 에이전트 권고 (저장된 결과)
            saved = None if results is None else results[results["test_row"] == t]
            if saved is not None and len(saved):
                st.caption("에이전트 권고 (outputs/agent_results.csv에 저장된 결과)")
                show_recommendation(saved.iloc[0].to_dict(), actual_map.get(t))
            else:
                st.info("저장된 에이전트 권고가 없습니다.")

            # 6-5. 에이전트 실시간 실행 (API 키가 있을 때만)
            can_run = agent_available()
            if st.button("▶ 이 거래로 에이전트 다시 실행 (Claude API, 비용 발생)", disabled=not can_run):
                import anthropic                                   # 키가 있을 때만 불러옴
                from agent import review_transaction
                with st.status("에이전트가 도구를 호출하며 조사하는 중...", expanded=True) as status:
                    live = review_transaction(anthropic.Anthropic(), toolkit, t, p)
                    status.update(label=f"조사 완료 (도구 {live['n_tool_calls']}회 호출)", state="complete")
                show_recommendation(live, actual_map.get(t))
            if not can_run:
                st.caption("API 키(ANTHROPIC_API_KEY)와 anthropic 패키지가 있어야 실행할 수 있습니다.")

            # 6-6. 도구 2: SHAP 기여도
            st.markdown("##### 도구 2 · 점수에 영향을 준 변수 (SHAP)")
            contrib = pd.DataFrame(tools["explain"]["top_contributions"])
            st.altair_chart(bar_chart(
                contrib, "shap", "feature", color="direction",
                color_domain=["사기 쪽", "정상 쪽"], color_range=[FRAUD_COLOR, LEGIT_COLOR],
                x_title="SHAP 값 (+ 사기 쪽 / − 정상 쪽)", y_title=None, sort=list(contrib["feature"]),
                tooltip=["feature", "direction", alt.Tooltip("value:Q", title="변환값", format=".3f"),
                         alt.Tooltip("shap:Q", format="+.3f")],
            ), width="stretch")
            st.caption(f"평균 예측값 {tools['explain']['base_value']:.3f}에서 시작해 변수들의 SHAP 값을 더하면 {p:.3f}가 됩니다.")

            # 6-7. 도구 3: 비슷한 과거 사례
            sim = tools["similar"]
            st.markdown(f"##### 도구 3 · 비슷한 과거 거래 10건 중 사기 {sim['similar_fraud_count']}건")
            cases = pd.DataFrame(sim["cases"])
            cases["순위"] = range(1, len(cases) + 1)
            dots = alt.Chart(cases).mark_circle(size=160, opacity=1, stroke="white", strokeWidth=2).encode(
                x=alt.X("distance:Q", title="거리 (작을수록 비슷함)", scale=alt.Scale(zero=True)),
                y=alt.Y("label:N", title=None, sort=["사기", "정상"]),
                color=alt.Color("label:N", scale=alt.Scale(domain=["사기", "정상"], range=[FRAUD_COLOR, LEGIT_COLOR]),
                                legend=None),
                tooltip=["순위", "label", alt.Tooltip("distance:Q", format=".2f"), "train_row"],
            ).properties(height=120)
            st.altair_chart(dots, width="stretch")
            st.caption("Train 데이터에서 30개 특성이 가장 비슷한 거래들의 실제 라벨입니다. Train 전체의 사기 비율은 약 0.17%입니다.")

            # 6-8. 도구 1, 4: 이례적인 특성과 기대 손실
            col_a, col_b = st.columns(2)
            with col_a:
                tx = tools["transaction"]
                st.markdown(f"##### 도구 1 · 이례적인 특성 {len(tx['unusual_features'])}개")
                unusual = pd.DataFrame([{"특성": c, "정상 거래 기준 백분위": tx["features"][c]["normal_percentile"]}
                                        for c in tx["unusual_features"]])
                if unusual.empty:
                    st.write("정상 거래의 상·하위 1% 밖에 있는 특성이 없습니다.")
                else:
                    st.dataframe(unusual, hide_index=True, width="stretch", height=min(250, 36 * (len(unusual) + 1)))
                st.caption(f"금액 백분위 {tx['features']['Amount']['normal_percentile']}% · "
                           f"{int(tx['features']['Hour']['value'])}시 거래")
            with col_b:
                pr = tools["priority"]
                st.markdown("##### 도구 4 · 검토 가치")
                st.metric("기대 손실 (사기 확률 × 사기 1건 손실)", f"{pr['expected_loss_eur']:.1f} EUR")
                st.metric("검토로 얻는 순이익 (기대 손실 − 검토 비용)", f"{pr['net_benefit_eur']:.1f} EUR")

# ── 7. 탭 2: 모델 비교 ────────────────────────────────────
TABLES = ["baseline_val_metrics", "eda_val_split_comparison", "supervised_cv_summary", "final_test_metrics",
          "final_test_model_comparison", "final_test_thresholds", "final_test_cost", "final_test_shap_importance"]

with tab_models:
    tb = {name: read_csv(REPORT_TABLE_DIR / f"{name}.csv") for name in TABLES}
    missing = [n for n, df in tb.items() if df is None]
    if missing:
        st.warning("일부 결과표가 없습니다. 과제 노트북(experiments/)을 순서대로 실행하면 생성됩니다: "
                   + ", ".join(f"`{m}.csv`" for m in missing))

    # 7-1. 정확도의 함정
    if tb["baseline_val_metrics"] is not None:
        st.subheader("① 정확도의 함정")
        base = tb["baseline_val_metrics"]
        dummy = base[base["Model"].str.startswith("Dummy")].iloc[0]
        a, b, c = st.columns(3)
        a.metric("전부 정상으로 예측했을 때 정확도", f"{dummy['Accuracy']:.2%}")
        b.metric("그때의 PR-AUC", f"{dummy['PR_AUC']:.4f}")
        c.metric("그때 잡은 사기", f"{int(dummy['TP'])}건 / {int(dummy['TP'] + dummy['FN'])}건")
        st.caption("사기가 0.17%뿐이라 정확도는 의미가 없습니다. 그래서 PR-AUC와 검토 건수로 평가합니다. (Validation 기준)")
        st.dataframe(base[["Model", "Accuracy", "PR_AUC", "Recall", "Precision", "TP", "FP", "FN"]].round(4),
                     hide_index=True, width="stretch")

    # 7-2. 최종 Test 성능
    if tb["final_test_metrics"] is not None:
        st.subheader("② 최종 Test 성능 (Test에 한 번만 적용)")
        fm = tb["final_test_metrics"].copy()
        fm["모델 (임계값)"] = fm["Model"] + " @" + fm["Threshold"].astype(str)
        fm["임계값 종류"] = fm["Threshold_Type"].map({"default": "기본값 0.5", "selected_on_validation": "Validation에서 선택"})
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.dataframe(fm[["모델 (임계값)", "임계값 종류", "PR_AUC", "Recall", "Precision", "TP", "FN", "Total_Flagged"]]
                         .rename(columns={"Total_Flagged": "검토 건수", "FN": "놓친 사기"}).round(3),
                         hide_index=True, width="stretch")
        with col2:
            pr_auc = fm.drop_duplicates("Model")[["Model", "PR_AUC"]]
            st.altair_chart(bar_chart(pr_auc, "PR_AUC", "Model", x_title="PR-AUC (Test)"), width="stretch")
        st.caption("운영 모델은 RandomForest @0.15입니다. Validation에서 'Recall 0.8 이상이면서 검토 건수 최소' 규칙으로 골랐습니다.")

    # 7-3. 지도 vs 비지도
    if tb["final_test_model_comparison"] is not None:
        st.subheader("③ 지도학습 vs 비지도 이상 탐지 (Recall 약 0.8 기준)")
        mc = tb["final_test_model_comparison"]
        type_scale = dict(color="Type", color_domain=["지도", "비지도"], color_range=[LEGIT_COLOR, FRAUD_COLOR])
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.altair_chart(bar_chart(mc, "PR_AUC", "Model", x_title="PR-AUC", sort="-x", **type_scale,
                                      tooltip=["Type", "Model", alt.Tooltip("PR_AUC:Q", format=".3f")]),
                            width="stretch")
        with col2:
            st.altair_chart(bar_chart(mc, "Total_Flagged", "Model", x_title="검토해야 하는 건수", sort="x", **type_scale,
                                      tooltip=["Type", "Model", "Total_Flagged", alt.Tooltip("Recall:Q", format=".3f")]),
                            width="stretch")
        sup_min = mc.loc[mc["Type"] == "지도", "Total_Flagged"].min()
        uns = mc.loc[mc["Type"] == "비지도", "Total_Flagged"]
        st.caption(f"같은 Recall을 내려면 비지도 모델은 검토 건수가 약 {uns.min() / sup_min:.0f}~{uns.max() / sup_min:.0f}배 필요합니다. "
                   "비지도는 라벨 없는 신종 사기 감시용 보조 탐지기로 의미가 있습니다.")

    # 7-4. 반복 교차검증
    if tb["supervised_cv_summary"] is not None:
        st.subheader("④ 모델·불균형 처리 비교 (Train 내부 5-fold × 2회 반복 교차검증)")
        cv = tb["supervised_cv_summary"].copy()
        cv["low"], cv["high"] = cv["PR_AUC_mean"] - cv["PR_AUC_std"], cv["PR_AUC_mean"] + cv["PR_AUC_std"]
        order = list(cv.sort_values("PR_AUC_mean", ascending=False)["Model"])
        base_enc = alt.Chart(cv).encode(y=alt.Y("Model:N", sort=order, title=None))
        err = base_enc.mark_rule(strokeWidth=2, color=LEGIT_COLOR).encode(
            x=alt.X("low:Q", title="PR-AUC (평균 ± 표준편차)", scale=alt.Scale(zero=False)), x2="high:Q")
        pts = base_enc.mark_circle(size=120, color=LEGIT_COLOR, opacity=1).encode(
            x="PR_AUC_mean:Q",
            tooltip=["Model", alt.Tooltip("PR_AUC_mean:Q", title="평균", format=".3f"),
                     alt.Tooltip("PR_AUC_std:Q", title="표준편차", format=".3f"),
                     alt.Tooltip("Flagged_mean:Q", title="Recall 0.8 검토 건수", format=".1f"),
                     alt.Tooltip("Fit_Seconds_mean:Q", title="학습 시간(초)", format=".1f")])
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.altair_chart((err + pts).properties(height=260), width="stretch")
        with col2:
            st.dataframe(cv[["Model", "PR_AUC_mean", "PR_AUC_std", "Flagged_mean", "Fit_Seconds_mean"]]
                         .rename(columns={"PR_AUC_mean": "PR-AUC 평균", "PR_AUC_std": "표준편차",
                                          "Flagged_mean": "검토 건수", "Fit_Seconds_mean": "학습(초)"}).round(3),
                         hide_index=True, width="stretch")
        st.caption("트리 모델끼리의 차이는 표준편차보다 작아 사실상 비슷합니다. 로지스틱 회귀와의 차이는 표준편차보다 훨씬 큽니다.")

    # 7-5. 임계값 선택
    if tb["final_test_thresholds"] is not None:
        st.subheader("⑤ 임계값에 따른 Recall과 검토 건수 (RandomForest, Test)")
        th = tb["final_test_thresholds"].rename(columns={"Total Flagged (총 경고)": "검토 건수"})
        th = th[th["Model"] == "RandomForest"]
        rule = alt.Chart(pd.DataFrame({"Threshold": [FINAL_THRESHOLD]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="Threshold:Q")

        def threshold_line(col, title, fmt):
            line = alt.Chart(th).mark_line(strokeWidth=2, color=LEGIT_COLOR, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
                x=alt.X("Threshold:Q", title="임계값"), y=alt.Y(f"{col}:Q", title=title),
                tooltip=[alt.Tooltip("Threshold:Q", title="임계값"), alt.Tooltip(f"{col}:Q", format=fmt)])
            return (line + rule).properties(height=240)

        col1, col2 = st.columns(2, gap="large")
        col1.altair_chart(threshold_line("Recall", "Recall (사기 탐지율)", ".3f"), width="stretch")
        col2.altair_chart(threshold_line("검토 건수", "검토 건수", ",d"), width="stretch")
        st.caption(f"점선은 운영 임계값 {FINAL_THRESHOLD}입니다. 임계값을 낮출수록 사기를 더 잡지만 검토 건수가 빠르게 늘어납니다.")

    # 7-6. 비용
    if tb["final_test_cost"] is not None:
        st.subheader("⑥ 총 손실 비용 (Test)")
        cost = tb["final_test_cost"].copy()
        cost["모델 (임계값)"] = cost["Model"] + " @" + cost["Threshold"].astype(str)
        long = cost.melt(id_vars=["모델 (임계값)", "Total_Cost"], value_vars=["Fraud_Loss", "Review_Cost"],
                         var_name="항목", value_name="비용")
        long["항목"] = long["항목"].map({"Fraud_Loss": "놓친 사기 손실", "Review_Cost": "검토 비용"})
        stacked = alt.Chart(long).mark_bar(height={"band": 0.7}, stroke="white", strokeWidth=2).encode(
            x=alt.X("비용:Q", title="비용 (EUR)", stack="zero"),
            y=alt.Y("모델 (임계값):N", sort=alt.EncodingSortField("Total_Cost", order="ascending"), title=None),
            color=alt.Color("항목:N", scale=alt.Scale(domain=["놓친 사기 손실", "검토 비용"], range=[FRAUD_COLOR, LEGIT_COLOR]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["모델 (임계값)", "항목", alt.Tooltip("비용:Q", format=",.0f"), alt.Tooltip("Total_Cost:Q", title="합계", format=",.0f")],
        ).properties(height=34 * len(cost) + 40)
        st.altair_chart(stacked, width="stretch")
        st.caption(f"총 손실 = 놓친 사기 × {FRAUD_LOSS_PER_CASE} EUR + 검토 건수 × {REVIEW_COST_PER_CASE} EUR (가정값). "
                   "놓친 사기 손실이 대부분이라 검토를 조금 늘려서라도 사기를 잡는 쪽이 유리합니다.")

    # 7-7. SHAP 변수 중요도 + 분할 방식 비교
    col1, col2 = st.columns(2, gap="large")
    if tb["final_test_shap_importance"] is not None:
        with col1:
            st.subheader("⑦ 전체 변수 중요도 (SHAP)")
            imp = tb["final_test_shap_importance"].head(10)
            st.altair_chart(bar_chart(imp, "Mean_Abs_SHAP", "Feature", x_title="평균 |SHAP| (Test)"), width="stretch")
            st.caption("V1~V28은 익명화된 PCA 변수라 실제 의미(가맹점, 위치 등)는 알 수 없습니다.")
    if tb["eda_val_split_comparison"] is not None:
        with col2:
            st.subheader("⑧ 분할 방식 비교 (Validation)")
            sp = tb["eda_val_split_comparison"]
            st.dataframe(sp[["Split_Method", "Val_Fraud_Count", "Val_PR_AUC", "Flagged_at_Recall_0.8"]]
                         .rename(columns={"Split_Method": "분할 방식", "Val_Fraud_Count": "Validation 사기",
                                          "Val_PR_AUC": "PR-AUC", "Flagged_at_Recall_0.8": "Recall 0.8 검토 건수"}).round(3),
                         hide_index=True, width="stretch")
            st.caption("중복 사기 거래가 Train과 Test에 나뉘어 들어가지 않도록 계층 + 그룹 분할을 썼습니다.")

# ── 8. 탭 3: 에이전트 평가 ────────────────────────────────
with tab_eval:
    if not reveal:
        st.info("사이드바에서 **정답 공개**를 켜면 에이전트 권고와 실제 라벨을 비교합니다.")
    elif evaluation is None:
        st.warning("agent_evaluation.csv가 없습니다. 에이전트 결과를 evaluate.py로 평가하면 생성됩니다.")
    else:
        ev = evaluation.copy()
        ev["권고"] = ev["recommendation"].map(REC_LABEL)
        legit_rec = ev["recommendation"] == "legit_likely"
        dangerous = ev[legit_rec & (ev["actual"] == "사기")]
        saved_reviews = ev[legit_rec & (ev["actual"] == "정상")]
        wrong_fraud = ev[(ev["recommendation"] == "fraud_likely") & (ev["actual"] == "정상")]

        a, b, c = st.columns(3)
        a.metric("위험한 실수 (사기를 정상으로 권고)", f"{len(dangerous)}건")
        b.metric("아낀 검토 (정상을 정상으로 권고)", f"{len(saved_reviews)} / {int((ev['actual'] == '정상').sum())}건")
        c.metric("사기로 권고했지만 정상", f"{len(wrong_fraud)}건")

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("##### 우선순위별 실제 라벨")
            pri = ev.groupby(["priority", "actual"]).size().reset_index(name="건수")
            pri["우선순위"] = pri["priority"].map(PRI_LABEL)
            st.altair_chart(alt.Chart(pri).mark_bar(height={"band": 0.7}, stroke="white", strokeWidth=2).encode(
                x=alt.X("건수:Q", stack="zero"),
                y=alt.Y("우선순위:N", sort=["높음", "중간", "낮음"], title=None),
                color=alt.Color("actual:N", scale=alt.Scale(domain=["사기", "정상"], range=[FRAUD_COLOR, LEGIT_COLOR]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["우선순위", "actual", "건수"],
            ).properties(height=180), width="stretch")
            st.caption("우선순위가 높은 거래부터 보면 사기를 먼저 만나는지 확인합니다.")
        with col2:
            st.markdown("##### 점수 구간 × 권고별 실제 라벨")
            cross = pd.crosstab([ev["score_band"], ev["권고"]], ev["actual"], margins=True, margins_name="합계")
            st.dataframe(cross, width="stretch")

        st.markdown("##### 틀린 권고")
        wrong = pd.concat([dangerous, wrong_fraud])
        if wrong.empty:
            st.write("없음")
        else:
            st.dataframe(wrong[["test_row", "fraud_prob", "권고", "priority", "actual", "summary"]]
                         .rename(columns={"test_row": "거래 번호", "fraud_prob": "사기 확률", "priority": "우선순위",
                                          "actual": "실제", "summary": "에이전트 요약"}).round(3),
                         hide_index=True, width="stretch")
            st.caption("리뷰 큐 탭에서 거래 번호를 찾아 도구 결과를 함께 보면 왜 틀렸는지 확인할 수 있습니다.")

# ── 9. 탭 4: 시스템 구조 ──────────────────────────────────
with tab_system:
    st.subheader("역할 분담")
    st.markdown(f"""
| 단계 | 담당 | 하는 일 |
|---|---|---|
| ① 판정 | RF 모델 | Test 거래 전체에 사기 확률을 매기고, {FINAL_THRESHOLD} 이상만 리뷰 큐로 보냄 |
| ② 근거 수집 | 도구 4개 (Python) | 거래 정보·SHAP·비슷한 과거 사례·기대 손실 계산 |
| ③ 해석·요약 | 에이전트 (Claude) | 도구를 골라 호출하고, 근거가 같은 쪽을 가리키는지 해석해 권고와 우선순위를 정함 |
| ④ 최종 결정 | 사람 | 우선순위 순으로 검토하고 라벨을 확정함 |
""")
    st.subheader("설계 원칙")
    st.markdown("""
- **에이전트는 정답을 볼 수 없습니다.** 도구에는 Test 정답 라벨을 넘기지 않고, 평가 단계에서만 비교합니다.
- **계산은 코드가, 해석은 LLM이 합니다.** 숫자는 모두 도구가 계산하고, LLM은 그 숫자를 근거로 요약합니다.
- **에이전트는 점수를 다시 매기지 않습니다.** 판정은 RF가 하고, 에이전트는 검토 순서와 근거 정리를 돕습니다.
""")
    st.subheader("한계")
    st.markdown("""
- 에이전트는 RF와 **같은 익명화 특성 30개**를 봅니다. 그래서 RF가 0점을 준 미탐 사기는 에이전트도 찾을 수 없습니다.
- 비슷한 과거 사례가 모두 사기인데 실제로는 정상인 거래처럼, 도구 결과로는 구별할 근거가 없는 경우가 있습니다.
- 애매한 구간이 15건뿐이라 1건 차이로 비율이 크게 흔들립니다.
- 위치, 고객 이용 패턴 같은 정보가 있는 데이터라면 도구를 더 늘려 한계를 줄일 수 있습니다.
""")
