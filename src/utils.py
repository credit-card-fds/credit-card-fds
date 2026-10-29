"""
================================================================================
[FDS 프로젝트] 공통 유틸리티 모듈 (src/utils.py)
================================================================================
- 작성 목적 : FDS(이상거래탐지) 모델링 및 평가 과정에서 반복 사용되는 
              핵심 공통 평가 함수 및 시각화 함수들을 통합 관리하여 
              코드 중복을 제거하고 모듈화된 유지보수 환경을 제공함.
================================================================================
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 데이터프레임 조작 및 표 생성을 위한 라이브러리
import torch  # PyTorch 딥러닝 프레임워크 불러오기
import torch.nn as nn  # 신경망 계층(Linear, ReLU 등) 생성을 위한 모듈 불러오기
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
import koreanize_matplotlib # 한글 폰트 설정이 필요한 경우


def load_and_preprocess_data(data_dir):
    """
    Train, Validation 분할 데이터를 로드하고 X(피처)와 y(Class)로 분리합니다.
    - 스케일링(Amount)과 Time → Hour 변환은 eda_and_split.ipynb에서 Train 기준으로 이미 수행됨
    - Time 컬럼이 남아 있는 경우에만 제거 (현재 분할 데이터에는 Hour만 있음)
    """
    # 1. 팀원이 분할하여 생성한 3개 핵심 파일 존재 여부 정의 및 사전 점검
    required_files = {
        'train': os.path.join(data_dir, 'train_processed.csv'),       # Train 데이터 절대 경로
        'val': os.path.join(data_dir, 'validation_processed.csv'),    # Validation 데이터 절대 경로
        'test': os.path.join(data_dir, 'test_processed.csv')          # Test 데이터 절대 경로
    }

    # 3개 파일 중 하나라도 실제 경로에 존재하지 않는지 검사
    missing_files = [path for name, path in required_files.items() if not os.path.exists(path)]

    if missing_files:                                             # 누락된 파일이 하나라도 존재할 경우
        missing = "\n".join(f" - 누락된 파일: {path}" for path in missing_files)
        # 노트북에서도 원인이 바로 보이도록 sys.exit 대신 예외 발생
        raise FileNotFoundError(
            f"필수 분할 데이터 파일이 존재하지 않습니다:\n{missing}\n"
            "eda_and_split.ipynb를 먼저 실행하여 Split 데이터를 생성해 주세요!"
        )

    # 2. 정제된 CSV 데이터 로드
    train_df = pd.read_csv(required_files['train'])               # Train 분할 데이터셋 로드
    val_df = pd.read_csv(required_files['val'])                   # Validation 분할 데이터셋 로드

    # 3. 데이터프레임에서 X(피처 데이터)와 y(정답/Class 라벨) 분리
    X_train = train_df.drop(columns=['Class'])                    # Train 데이터에서 Class 컬럼을 제외하여 입력 피처 생성
    y_train = train_df['Class'].values                            # Train 데이터의 Class 컬럼만 추출하여 1차원 정답 배열 생성

    X_val = val_df.drop(columns=['Class'])                        # Validation 데이터에서 Class 컬럼을 제외하여 입력 피처 생성
    y_val = val_df['Class'].values                                # Validation 데이터의 Class 컬럼만 추출하여 1차원 정답 배열 생성

    print(f"[Data Loaded] Train: {X_train.shape}, Validation: {X_val.shape}") # 로드된 데이터셋의 행/열 형태 출력

    # 4. Time 변수가 남아 있으면 제거
    drop_cols = ['Time'] if 'Time' in X_train.columns else []     # X_train에 'Time' 컬럼이 존재하면 제거 대상에 추가

    X_train = X_train.drop(columns=drop_cols)                     # X_train에서 Time 컬럼 제거
    X_val = X_val.drop(columns=drop_cols)                         # X_val에서 Time 컬럼 제거

    return X_train, y_train, X_val, y_val

"""
evaluate_fds_model() :
    - 모델 예측 결과(y_true, y_pred, y_prob)를 바탕으로 FDS 핵심 성능 지표 산출
    - 지표 : Accuracy, PR-AUC(평균 정밀도), Recall, Precision, F1-Score, Confusion Matrix
    - 불균형 금융 데이터 특성에 맞춰 PR-AUC 지표 계산 및 상세 혼동행렬 출력
"""
def evaluate_fds_model(model_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n================ [{model_name}] 평가 결과 ================")
    print(f"1. Accuracy  (정확도) : {acc:.5f}")
    print(f"2. PR-AUC    (평균정밀도): {pr_auc:.5f} ★")
    print(f"3. Recall    (재현율)  : {rec:.5f}")
    print(f"4. Precision (정밀도)  : {prec:.5f}")
    print(f"5. F1-Score  (조화평균): {f1:.5f}")
    print(f"----------------------------------------------------------")
    print(f"Confusion Matrix (혼동행렬):\n{cm}")
    print(f" - TN(정상 예측 성공): {tn:,}건 | FP(오탐 / 오인 신고): {fp:,}건 | FN(미탐 / 놓친 사기): {fn:,}건 | TP(사기 탐지 성공): {tp:,}건")
    print(f"==========================================================\n")
    
    return {
        'Model': model_name, 'Accuracy': acc, 'PR_AUC': pr_auc,
        'Recall': rec, 'Precision': prec, 'F1_Score': f1,
        'FP': fp, 'FN': fn, 'TP': tp, 'TN': tn
    }


"""
임계값 Threshold 별 검토 건수 시뮬레이션 함수
사기일 가능성이 몇 점 이상일 때 실제 사기 거래로 최종 판단하고 검토/대응할 것인가?"를 결정하기 위한 함수
사기점수 계산 score , 다양한 기준선 Threshold
임계값이란 60점 이상이면 합격 이라고 정할 때 60점이 바로 임계값(threshold)
"""
def evaluate_thresholds(y_true, scores, thresholds=None):
    """
    y_true : 실제 사기 여부 (0: 정상, 1: 사기)
    scores : 모델이 출력한 확률값(y_prob) 또는 비지도 이상치 점수(score)
    thresholds : 테스트할 임계값 목록 (기본값 : None)
    """
    if thresholds is None :
        # scores 의 값의 상위 0.1% ~ 5% 영역을 커버하는 10개의 임계값을 자동으로 설정
        thresholds = np.percentile(scores, np.linspace(95, 99.9, 10))

    results = [] # 임계값별 평가 결과를 담을 빈 리스트 생성

    for th in thresholds: #임계값을 하나씩 순회
        # 이상치 점수가 임계값(th) 이상이면 1(사기), 미만이면 0(정상) 으로 예측
        y_pred = (scores >= th).astype(int)

        # 실제 값과 예측값을 비교해 True Negative, False Positive, False Negative, True Positive 건수 추출
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Recall 재현율: 실제 사기 중 잡은 비율 계산
        rec = recall_score(y_true, y_pred, zero_division=0)
        # Precision 정밀도 : 사기라고 한 것 중 진짜 사기 비율 계산
        prec = precision_score(y_true, y_pred, zero_division=0)
        # F1-Score 재현율과 정밀도의 조화 평균 계산
        f1 = f1_score(y_true, y_pred, zero_division=0)

        #계산된 지표들을 딕셔너리 형태로 저장
        results.append({
            "Threshold" : th, #적용된 임계값
            "Recall" : rec, #재현율 (사기 탐지율)
            "Precision" : prec, #정밀도
            "F1-Score" : f1, # f1 스코어
            "TP (사기 탐지)" : tp, # 실제 사기를 사기로 맞춘 건수
            "FP (오탐/검토 대상)" : fp, # 정상 거래를 사기로 오탐해 상담원이 검토해야 하는 건수
            "FN (놓친 사기)" : fn, #사기를 정상으로 놓쳐 피해가 발생한 건수
            "Total Flagged (총 경고)" : fp+tp # FDS 시스템에서 경고를 올린 총 건수
        })
    return pd.DataFrame(results) #결과를 데이터 프레임 표로 반환

def plot_and_save_simulation(df_results, model_name, filename, figures_dir=None):
    """
    임계값(Threshold) 변화에 따른 Recall, Precision, FP(오탐/검토 건수) 변화를 이중 Y축으로 시각화하고 저장합니다.
    
    Parameters:
    - df_results: evaluate_thresholds 결과 데이터프레임
    - model_name: 그래프 제목에 표시할 모델 이름 (예: 'Isolation Forest')
    - filename: reports/figures 폴더에 저장할 파일명 (예: 'isolation_forest_simulation.png')
    - figures_dir: 저장할 디렉토리 경로 (기본값 설정 시 자동으로 상위 reports/figures 경로 설정)
    """
    if figures_dir is None:
        # src/utils.py 기준 상위 디렉토리(프로젝트 루트)의 reports/figures 경로 지정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        figures_dir = os.path.join(base_dir, '../reports/figures')

    os.makedirs(figures_dir, exist_ok=True) # reports/figures 폴더가 없으면 새로 생성
    save_path = os.path.join(figures_dir, filename) # 최종 저장할 파일의 경로

    fig, ax1 = plt.subplots(figsize=(10, 6)) # 가로 10인치, 세로 6인치 크기의 그래프 도화지 생성

    # 왼쪽 Y축: Recall(재현율) 및 Precision(정밀도)
    ax1.set_xlabel('Threshold 이상치 점수 임계값', fontsize=12)
    ax1.set_ylabel('Score (Recall / Precision)', color='tab:blue', fontsize=12)
    
    line1 = ax1.plot(df_results['Threshold'], df_results['Recall'], marker='o', color='tab:blue', label='Recall (사기 탐지율)')
    line2 = ax1.plot(df_results['Threshold'], df_results['Precision'], marker='s', color='tab:green', label='Precision (정밀도)')

    ax1.grid(True, linestyle='--', alpha=0.5)  # 점선 형태의 격자 배경 표시

    # 오른쪽 Y축: FP (상담원 검토 건수)
    ax2 = ax1.twinx()  # X축을 공유하는 오른쪽 Y축 생성
    ax2.set_ylabel('FP (오탐/검토 건수)', color='tab:red', fontsize=12)  
    line3 = ax2.plot(df_results['Threshold'], df_results['FP (오탐/검토 대상)'], marker='^', color='tab:red', linestyle='--', label='FP (검토 건수)')  
    ax2.tick_params(axis='y', labelcolor='tab:red')  # 오른쪽 Y축 눈금 색상 변경

    # 범례(Legend) 및 타이틀 설정
    lines = line1 + line2 + line3  # 3개 그래프 선 정보 하나로 통합
    labels = [l.get_label() for l in lines]  # 범례 라벨 텍스트 추출
    ax1.legend(lines, labels, loc='center left')  # 중앙 왼쪽에 범례 상자 표시

    plt.title(f'[{model_name}] Threshold Simulation (Recall vs FP)', fontsize=14, pad=15)  
    plt.tight_layout()  # 여백을 자동으로 깔끔하게 조정
    
    plt.savefig(save_path, dpi=300)  # 지정한 경로에 300 DPI 고해상도 이미지 파일로 저장
    plt.show()  # 화면에 그래프 출력
    print(f"[저장 완료] 그래프 파일: {save_path}")  # 저장 완료 안내 출력

"""
# PyTorch Autoencoder 모델
# PyTorch의 기본 신경망 클래스(nn.Module) 상속
정상 거래 패턴만 완벽하게 요약해서 외운 뒤, 
새로 들어온 거래가 정상이 맞는지 '모방'해보는 작업입니다.
정상 거래가 들어오면: 이미 외운 패턴이라 아주 똑같이 복원
사기(이상) 거래가 들어오면 본 적 없는 기괴한 패턴이라 
원래 모양대로 복원을 못 하고 엉뚱하게 그려냅니다
즉 내가 복원하기 힘들 만큼 이상하게 생긴 거래 = 사기 거래 비지도학습 모델
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim): #초기화 메서드(input_dim : 입력 피처 개수, 예 30개 = V1~V28, Amount, Hour)
        super(Autoencoder, self).__init__() #부모 클래스의 초기화 함수 실행

        #인코더 : 입력 데이터를 작은 차원(30차원 -> 16차원 -> 8차원)으로 압축하는 신경망
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),   #입력 차원(30)을 16차원으로 줄임
            nn.ReLU(),                  #비선형 활성화 함수 적용
            nn.Linear(16, 8),           #16 차원을 8차원 잠재공간으로 축소
            nn.ReLU()                   #비선형 활성화 함수 적용
        )

        # 디코더 : 압축된 8차원 데이터를 다시 원본 차원(8차원 -> 16차원 -> 30차원)으로 복구
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),           # 8차원을 16차원으로 확충
            nn.ReLU(),                  # 비선형 활성화 함수 적용
            nn.Linear(16, input_dim)    # 16차원을 다시 원본 피처 차원(30)으로 최종 복원
        )

    def forward(self, x): # 순전파(Forward Propagation) 연산 정의
        latent = self.encoder(x) #입력 데이터 x를 인코더에 넣어 8차원으로 압축
        reconstructed = self.decoder(latent) #압축된 latent를 디코더에 넣어 원본 모양으로 복원
        return reconstructed #복원된 데이터 반환
