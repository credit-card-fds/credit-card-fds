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

# 현재 스크립트 파일(baseline_evaluation.py)의 절대 경로 기준으로 프로젝트 루트 및 data 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))               # experiments 폴더 경로
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '../data/processed')   # data/processed 폴더 절대 경로 변환

# 1. 팀원이 분할하여 생성한 3개 핵심 파일 존재 여부 정의 및 사전 점검
required_files = {
    'train': os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv'),       # Train 데이터 절대 경로
    'val': os.path.join(PROCESSED_DATA_DIR, 'validation_processed.csv'),    # Validation 데이터 절대 경로
    'test': os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv')          # Test 데이터 절대 경로
}

# 3개 파일 중 하나라도 실제 경로에 존재하지 않는지 검사
missing_files = [path for name, path in required_files.items() if not os.path.exists(path)]

if missing_files:                                             # 누락된 파일이 하나라도 존재할 경우
    print("필수 분할 데이터 파일이 존재하지 않습니다:")
    for path in missing_files:                                # 누락된 파일 경로들을 반복 출력
        print(f" - 누락된 파일: {path}")
    print("eda_and_split.ipynb를 먼저 실행하여 Split 데이터를 생성해 주세요!")
    sys.exit(1)                                               # 파일 누락 시 프로그램이 에러로 튕기지 않도록 안전하게 종료

# 2. 정제된 CSV 데이터 로드
train_df = pd.read_csv(required_files['train'])               # Train 분할 데이터셋 로드
val_df = pd.read_csv(required_files['val'])                   # Validation 분할 데이터셋 로드

# 3. 데이터프레임에서 X(피처 데이터)와 y(정답/Class 라벨) 분리
X_train = train_df.drop(columns=['Class'])                    # Train 데이터에서 Class 컬럼을 제외하여 입력 피처 생성
y_train = train_df['Class'].values                            # Train 데이터의 Class 컬럼만 추출하여 1차원 정답 배열 생성

X_val = val_df.drop(columns=['Class'])                        # Validation 데이터에서 Class 컬럼을 제외하여 입력 피처 생성
y_val = val_df['Class'].values                                # Validation 데이터의 Class 컬럼만 추출하여 1차원 정답 배열 생성

print(f"[Data Loaded] Train: {X_train.shape}, Validation: {X_val.shape}") # 로드된 데이터셋의 행/열 형태 출력

# 4. Time 변수 제거 및 Amount 변수 RobustScaler 적용 (Data Leakage 방지 방식)
drop_cols = ['Time'] if 'Time' in X_train.columns else []     # X_train에 'Time' 컬럼이 존재하면 제거 대상에 추가

X_train_scaled = X_train.drop(columns=drop_cols).copy()       # X_train에서 Time 컬럼을 제거한 피처 복사본 생성
X_val_scaled = X_val.drop(columns=drop_cols).copy()           # X_val에서 Time 컬럼을 제거한 피처 복사본 생성

scaler = RobustScaler()                                       # 이상치에 강한 중앙값/IQR 기반 스케일러 객체 생성
X_train_scaled['Amount'] = scaler.fit_transform(X_train_scaled[['Amount']]) # Train 데이터의 Amount로 기준을 학습(fit) 후 변환(transform)
X_val_scaled['Amount'] = scaler.transform(X_val_scaled[['Amount']])         # Train 기준을 평가 데이터에 유출 없이 그대로 적용(transform만)

# 5. FDS(이상거래탐지) 모델 평가 지표 계산 및 출력을 담당하는 공통 함수
def evaluate_fds_model(model_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)                      # 전체 데이터 중 정답을 맞춘 비율 (TP+TN)/All 계산
    prec = precision_score(y_true, y_pred, zero_division=0)   # 사기로 예측한 건 중 진짜 사기 비율 TP/(TP+FP) 계산
    rec = recall_score(y_true, y_pred, zero_division=0)       # 실제 사기 건 중 탐지해낸 비율 TP/(TP+FN) 계산
    f1 = f1_score(y_true, y_pred, zero_division=0)            # Precision과 Recall의 조화평균 계산
    pr_auc = average_precision_score(y_true, y_prob)          # 불균형 데이터 핵심 지표인 PR-AUC(평균 정밀도) 계산
    
    cm = confusion_matrix(y_true, y_pred)                     # 실제값과 예측값을 비교하여 2x2 혼동행렬 생성
    tn, fp, fn, tp = cm.ravel()                               # 2x2 행렬을 TN(정상->정상), FP(정상->사기), FN(사기->정상), TP(사기->사기)로 분해
    
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
    }                                                         # 측정된 평가지표들을 딕셔너리 형태로 반환

# 6. Baseline 1: Dummy Model (무조건 최빈값 0으로만 예측하여 '정확도의 함정' 증명)
dummy_model = DummyClassifier(strategy='most_frequent')       # 가장 빈도수 높은 클래스(0: 정상)로만 예측하는 Dummy 객체 생성
dummy_model.fit(X_train_scaled, y_train)                      # Train 데이터셋의 최빈값 클래스 파악

dummy_pred = dummy_model.predict(X_val_scaled)                # Validation 데이터의 모든 행에 대해 0(정상)으로 일괄 예측
dummy_prob = np.full_like(y_val, fill_value=np.mean(y_train), dtype=float) # y_val 크기만큼 Train 데이터의 사기 비율 고정값으로 예측 확률 배열 생성

dummy_res = evaluate_fds_model("Dummy Model (Most Frequent)", y_val, dummy_pred, dummy_prob) # Dummy 모델 결과 계산 및 평가 출력

# 7. Baseline 2: Basic Logistic Regression (기본 로지스틱 회귀 모델)
lr_base = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs') # L-BFGS 최적화 알고리즘 기반 로지스틱 회귀 객체 생성
lr_base.fit(X_train_scaled, y_train)                          # X_train과 y_train 간의 선형 관계 학습 (가중치 및 편향 최적화)

lr_pred = lr_base.predict(X_val_scaled)                       # X_val에 대한 예측 클래스(0 또는 1) 산출 (기본 임계값 0.5 기준)
lr_prob = lr_base.predict_proba(X_val_scaled)[:, 1]           # X_val 각 행이 클래스 1(사기)일 계산된 확률값만 추출

lr_res = evaluate_fds_model("Basic Logistic Regression", y_val, lr_pred, lr_prob) # 로지스틱 회귀 모델 결과 계산 및 평가 출력

# 8. 평가 결과 데이터프레임 생성 및 CSV 파일 최종 저장
results_df = pd.DataFrame([dummy_res, lr_res])                # 두 모델의 결과 딕셔너리를 하나의 pandas DataFrame 테이블로 통합
save_path = os.path.join(PROCESSED_DATA_DIR, 'day1_baseline_results.csv') # 저장 디렉토리 경로 지정
results_df.to_csv(save_path, index=False)                     # Index 번호 없이 깔끔한 CSV 파일로 저장