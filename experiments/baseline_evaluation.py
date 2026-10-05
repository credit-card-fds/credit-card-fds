import os
import sys
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)

required_files = {
    'X_train': '../data/processed/X_train.csv',
    'X_val': '../data/processed/X_val.csv',
    'y_train': '../data/processed/y_train.csv',
    'y_val': '../data/processed/y_val.csv'
}

# 4개 파일 중 하나라도 없는지 사전 점검
missing_files = [path for name, path in required_files.items() if not os.path.exists(path)]

if missing_files:
    print("필수 분할 데이터 파일이 존재하지 않습니다:")
    for path in missing_files:
        print(f" - 누락된 파일: {path}")
    print("eda_and_split.ipynb를 먼저 실행하여 Split 데이터를 생성해 주세요!")
    sys.exit(1) # 파일 누락 시 스크립트 실행 즉시 안전 종료

# 1.데이터 로드 및 1차원 배열 변환
X_train = pd.read_csv('../data/processed/X_train.csv')                          # Train 학습용 피처 데이터셋 로드
X_val = pd.read_csv('../data/processed/X_val.csv')                              # Validation 검증용 피처 데이터셋 로드
y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()           # DataFrame을 1차원 NumPy 배열로 평탄화
y_val = pd.read_csv('../data/processed/y_val.csv').values.ravel()               # DataFrame을 1차원 NumPy 배열로 평탄화

print(f"[Data Loaded] Train: {X_train.shape}, Validation: {X_val.shape}")

# 2.Time 변수 제거 및 Amount 변수 RobustScaler 적용
drop_cols = ['Time'] if 'Time' in X_train.columns else []                       # X_train에 'Time' 컬럼이 존재하면 삭제 대상 리스트에 추가

X_train_scaled = X_train.drop(columns=drop_cols).copy()                         # X_train에서 Time 컬럼을 제거한 복사본 생성
X_val_scaled = X_val.drop(columns=drop_cols).copy()                             # X_val에서 Time 컬럼을 제거한 복사본 생성

scaler = RobustScaler() # 중앙값(Median)과 IQR(사분위수 범위) 기반 스케일러 객체 생성
X_train_scaled['Amount'] = scaler.fit_transform(X_train_scaled[['Amount']])     # X_train의 Amount 중앙값/IQR 계산 후 스케일링 적용
X_val_scaled['Amount'] = scaler.transform(X_val_scaled[['Amount']])             # X_train에서 계산된 통계량으로 X_val의 Amount 스케일링

# 3. 평가 지표 계산 및 출력을 담당하는 함수
def evaluate_fds_model(model_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)                                        # 전체 데이터 중 정답을 맞춘 비율 (TP+TN)/All 계산
    prec = precision_score(y_true, y_pred, zero_division=0)                     # 사기로 예측한 건 중 진짜 사기 비율 TP/(TP+FP) 계산
    rec = recall_score(y_true, y_pred, zero_division=0)                         # 실제 사기 건 중 탐지해낸 비율 TP/(TP+FN) 계산
    f1 = f1_score(y_true, y_pred, zero_division=0)                              # Precision과 Recall의 조화평균 계산
    pr_auc = average_precision_score(y_true, y_prob)                            # Precision-Recall 곡선 하단 면적(AP) 계산
    
    cm = confusion_matrix(y_true, y_pred)                                       # 실제값과 예측값을 비교하여 2x2 혼동행렬 생성
    tn, fp, fn, tp = cm.ravel()                                                 # 2x2 행렬을 TN(정상->정상), FP(정상->사기), FN(사기->정상), TP(사기->사기)로 분해
    
    print(f"\n================ [{model_name}] 평가 결과 ================")
    print(f"1. Accuracy  (정확도) : {acc:.5f}")
    print(f"2. PR-AUC    (평균정밀도): {pr_auc:.5f} ★")
    print(f"3. Recall    (재현율)  : {rec:.5f}")
    print(f"4. Precision (정밀도)  : {prec:.5f}")
    print(f"5. F1-Score  (조화평균): {f1:.5f}")
    print(f"----------------------------------------------------------")
    print(f"Confusion Matrix (혼동행렬):\n{cm}")
    print(f" - TN: {tn:,}건 | FP: {fp:,}건 | FN: {fn:,}건 | TP: {tp:,}건")
    print(f"==========================================================\n")
    
    return {
        'Model': model_name, 'Accuracy': acc, 'PR_AUC': pr_auc,
        'Recall': rec, 'Precision': prec, 'F1_Score': f1,
        'FP': fp, 'FN': fn, 'TP': tp, 'TN': tn
    }

# 4. Baseline 1: Dummy Model (최빈값 0으로만 예측)
dummy_model = DummyClassifier(strategy='most_frequent')                         # 무조건 데이터 수가 가장 많은 클래스(0: 정상)만 출력하는 모델 생성
dummy_model.fit(X_train_scaled, y_train)                                        # Train 데이터셋의 최빈값 클래스 파악

dummy_pred = dummy_model.predict(X_val_scaled)                                  # X_val의 모든 행에 대해 0(정상) 값으로 예측
dummy_prob = np.full_like(y_val, fill_value=np.mean(y_train), dtype=float)      # y_val 배열 크기만큼 Train 데이터의 사기 비율(약 0.0017) 고정값으로 배열 생성

dummy_res = evaluate_fds_model("Dummy Model (Most Frequent)", y_val, dummy_pred, dummy_prob) # Dummy 모델 결과 계산 및 출력

# 5. Baseline 2: Basic Logistic Regression
lr_base = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')    # L-BFGS 최적화 알고리즘 기반 로지스틱 회귀 객체 생성
lr_base.fit(X_train_scaled, y_train)                                            # X_train과 y_train 간의 선형 관계 학습 (가중치 및 편향 최적화)

lr_pred = lr_base.predict(X_val_scaled)          # X_val에 대한 예측 클래스(0 또는 1) 산출 (임계값 0.5 기준)
lr_prob = lr_base.predict_proba(X_val_scaled)[:, 1] # X_val 각 행이 클래스 1(사기)에 속할 확률값([:, 1])만 추출

lr_res = evaluate_fds_model("Basic Logistic Regression", y_val, lr_pred, lr_prob) # Logistic Regression 결과 계산 및 출력

# [Step 6] 결과 데이터프레임 생성 및 파일 저장
results_df = pd.DataFrame([dummy_res, lr_res])                                  # 딕셔너리 형태의 결과를 하나의 pandas DataFrame 테이블로 통합
os.makedirs('../data/processed', exist_ok=True)                                 # 저장 경로 폴더가 없을 경우 생성
results_df.to_csv('../data/processed/day1_baseline_results.csv', index=False)   # Index 번호 없이 CSV 파일로 저장