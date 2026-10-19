import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))               # experiments 폴더 경로
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '../data/processed')   # data/processed 폴더 절대 경로 변환

# 기존 경로 변수(BASE_DIR)를 활용해 상위 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.join(BASE_DIR, '..'))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import sys

from src.utils import evaluate_fds_model, evaluate_thresholds, plot_and_save_simulation, load_and_preprocess_data

def run_baseline_evaluation(X_train, y_train, X_val, y_val):
    """
    Dummy Model 및 Logistic Regression 베이스라인 모델을 평가합니다.
    """
    results = []

    # 5. Baseline 1: Dummy Model (무조건 최빈값 0으로만 예측하여 '정확도의 함정' 증명)
    dummy_model = DummyClassifier(strategy='most_frequent')        # 가장 빈도수 높은 클래스(0: 정상)로만 예측하는 Dummy 객체 생성
    dummy_model.fit(X_train, y_train)                              # Train 데이터셋의 최빈값 클래스 파악

    dummy_pred = dummy_model.predict(X_val)                        # Validation 데이터의 모든 행에 대해 0(정상)으로 일괄 예측
    dummy_prob = np.full_like(y_val, fill_value=np.mean(y_train), dtype=float) # y_val 크기만큼 Train 데이터의 사기 비율 고정값으로 예측 확률 배열 생성

    dummy_res = evaluate_fds_model("Dummy Model (Most Frequent)", y_val, dummy_pred, dummy_prob) # Dummy 모델 결과 계산 및 평가 출력
    results.append(dummy_res)

    # 6. Baseline 2: Basic Logistic Regression (기본 로지스틱 회귀 모델)
    lr_base = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs') # L-BFGS 최적화 알고리즘 기반 로지스틱 회귀 객체 생성
    lr_base.fit(X_train, y_train)                                  # X_train과 y_train 간의 선형 관계 학습 (가중치 및 편향 최적화)
    
    lr_pred = lr_base.predict(X_val)                               # X_val에 대한 예측 클래스(0 또는 1) 산출 (기본 임계값 0.5 기준)
    lr_prob = lr_base.predict_proba(X_val)[:, 1]                   # X_val 각 행이 클래스 1(사기)일 계산된 확률값만 추출

    lr_res = evaluate_fds_model("Basic Logistic Regression", y_val, lr_pred, lr_prob) # 로지스틱 회귀 모델 결과 계산 및 평가 출력
    results.append(lr_res)

    # utils.evaluate_thresholds를 활용해 로지스틱 회귀의 임계값별 지표 산출
    lr_threshold_df = evaluate_thresholds(y_val, lr_prob)

    return pd.DataFrame(results), lr_threshold_df

def main():
    # 데이터 로드 및 전처리 수행
    X_train, y_train, X_val, y_val = load_and_preprocess_data(PROCESSED_DATA_DIR)

    # 베이스라인 모델 평가 수행 및 시각화용 예측값 받아오기
    results_df, lr_threshold_df = run_baseline_evaluation(X_train, y_train, X_val, y_val)

    # 시각화 그래프 저장 경로 지정 및 파일 생성
    FIGURE_DIR = os.path.join(BASE_DIR, '../reports/figures')

    # 이중 Y축 시뮬레이션 그래프 생성 및 reports/figures 폴더 저장 실행
    plot_and_save_simulation(
        df_results=lr_threshold_df, 
        model_name="Basic Logistic Regression", 
        filename="logistic_regression_simulation.png",
        figures_dir=FIGURE_DIR
    )
    
    # 평가 결과 데이터프레임 생성 및 CSV 파일 최종 저장
    save_path = os.path.join(PROCESSED_DATA_DIR, 'baseline_results.csv') # 저장 디렉토리 경로 지정
    results_df.to_csv(save_path, index=False)                            # Index 번호 없이 깔끔한 CSV 파일로 저장
    print(f"[Results Saved] {save_path}")

if __name__ == "__main__":
    main()