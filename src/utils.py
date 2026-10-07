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
import seaborn as sns
import torch  # PyTorch 딥러닝 프레임워크 불러오기
import torch.nn as nn  # 신경망 계층(Linear, ReLU 등) 생성을 위한 모듈 불러오기
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, precision_recall_curve
)


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
    print(f" - TN: {tn:,}건 | FP: {fp:,}건 | FN: {fn:,}건 | TP: {tp:,}건")
    print(f"==========================================================\n")
    
    return {
        'Model': model_name, 'Accuracy': acc, 'PR_AUC': pr_auc,
        'Recall': rec, 'Precision': prec, 'F1_Score': f1,
        'FP': fp, 'FN': fn, 'TP': tp, 'TN': tn
    }


"""
plot_pr_curve() :
    - 불균형 FDS 데이터의 핵심 평가 곡선인 Precision-Recall Curve 시각화 및 저장
    - 무작위/일괄 예측 성능 기준선(No Skill Baseline) 대비 모델 성능 직관적 비교 제공
"""
def plot_pr_curve(y_true, y_prob, model_name="Model", save_path=None):
    """
    Precision-Recall Curve(정밀도-재현율 곡선)를 시각화하여 생성하고 지정된 경로에 저장합니다.
    """
    # 1. PR 곡선 좌표(Precision, Recall) 계산
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    # 2. 불균형 데이터 기준 무작위 예측 성능 기준선(No Skill Line = 실제 사기 비율) 산출
    no_skill = np.mean(y_true)
    pr_auc = average_precision_score(y_true, y_prob)           # 차트 범례 표시용 PR-AUC 값 계산
    
    # 3. 그래프 그리기 설정
    plt.figure(figsize=(7, 5))                                  # 그래프 가로/세로 크기 설정
    plt.plot(recall, precision, marker='.', label=f'{model_name} (PR-AUC = {pr_auc:.4f})') # 모델 PR 곡선 플롯
    plt.axhline(y=no_skill, color='r', linestyle='--', label=f'No Skill Baseline ({no_skill:.5f})') # 기준선 플롯
    
    plt.xlabel('Recall (Recall / True Positive Rate)')           # X축 라벨 (재현율)
    plt.ylabel('Precision (Positive Predictive Value)')          # Y축 라벨 (정밀도)
    plt.title(f'Precision-Recall Curve - {model_name}')         # 차트 제목 설정
    plt.legend(loc='upper right')                               # 범례 위치 우상단 고정
    plt.grid(True, linestyle=':', alpha=0.6)                    # 격자선 투명도 조절
    
    # 4. 이미지 저장 로직
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # 저장 경로 상위 디렉터리 자동 생성
        plt.savefig(save_path, dpi=300, bbox_inches='tight')   # 여백 없이 선명하게 고해상도(300DPI) 저장
        print(f"[Graph Saved] PR Curve -> {save_path}")
    
    plt.close()                                                 # 메모리 누수 방지를 위한 플롯 닫기


"""
plot_confusion_matrix_heatmap() :
    - 혼동행렬(TN, FP, FN, TP) 수치를 Seaborn Heatmap 형태로 시각화 및 저장
    - FDS 운영의 핵심인 FN(미탐)과 FP(오탐) 비중을 시각적으로 명확히 파악 가능
"""
def plot_confusion_matrix_heatmap(y_true, y_pred, model_name="Model", save_path=None):
    """
    Confusion Matrix(혼동행렬)를 직관적인 Seaborn Heatmap으로 시각화하여 생성하고 저장합니다.
    """
    # 1. 혼동행렬 데이터 산출
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. 히트맵 그리기 설정
    plt.figure(figsize=(6, 5))                                  # 그래프 크기 지정
    sns.heatmap(
        cm, 
        annot=True,                                             # 셀 내부 수치 표시 활성화
        fmt='d',                                                # 정수(Integer) 포맷 설정
        cmap='Blues',                                           # 푸른색 계열 칼라맵 적용
        cbar=False,                                             # 범례 칼라바 숨김 처리
        xticklabels=['Normal (0)', 'Fraud (1)'],                # X축 예측 라벨 명칭
        yticklabels=['Normal (0)', 'Fraud (1)']                 # Y축 실제 정답 라벨 명칭
    )
    
    plt.xlabel('Predicted Label')                               # X축 타이틀
    plt.ylabel('True Label')                                    # Y축 타이틀
    plt.title(f'Confusion Matrix - {model_name}')               # 차트 제목
    
    # 3. 이미지 저장 로직
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # 저장 경로 상위 디렉터리 자동 생성
        plt.savefig(save_path, dpi=300, bbox_inches='tight')   # 여백 제거 및 고해상도 저장
        print(f"[Graph Saved] Confusion Matrix -> {save_path}")
        
    plt.close()                                                 # 메모리 누수 방지를 위한 플롯 닫기


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

    for th in thresholds: #생선된 10개의 임계값을 하나씩 순회
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
            "Threshold" : tn, #적용된 임계값
            "Recall" : rec, #재현율 (사기 탐지율)
            "Precision" : prec, #정밀도
            "F1-Score" : f1, # f1 스코어
            "TP (사기 탐지)" : tp, # 실제 사기를 사기로 맞춘 건수
            "FP (오탐/검토 대상)" : fp, # 사기를 정상으로 오탐해 상담원이 검토해야 하는 건
            "FN (놓친 사기)" : fn, #사기를 정상으로 놓쳐 피해가 발생한 건수
            "Total Flagged (총 경고)" : fp+tp # FDS 시스템에서 경고를 올린 총 건수
        })
    return pd.DataFrame(results) #결과를 데이터 프레임 표로 반환


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
    def __init__(self, input_dim): #초기화 메서드(input_dim : 입력 피처 개수, 예 29개)
        super(Autoencoder, self).__init__() #부모 클래스의 초기화 함수 실행

        #인코더 : 입력 데이터를 작은 차원(29차원 -> 16차원 -> 8차원)으로 압축하는 신경망
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),   #입력 차원29 를 16차원으로 줄임
            nn.ReLU(),                  #비선형 활성화 함수 적용
            nn.Linear(16, 8),           #16 차원을 8차원 잠재공간으로 축소
            nn.ReLU()                   #비선형 활성화 함수 적용
        )

        # 디코더 : 압축된 8차원 데이터를 다시 원본 차원(8차원 -> 16차원 ->29차원)으로 복구
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),           # 8차원을 16차원으로 확충
            nn.ReLU(),                  # 비선형 활성화 함수 적용
            nn.Linear(16, input_dim)    # 16차원을 다시 원본 피처 차원(29)으로 최종 복원
        )

    def forward(self, x): # 순전파(Foward Propagation) 연산 정의
        latent = self.encoder(x) #입력 데이터 x를 인코더에 넣어 8차원으로 압축
        reconstructed = self.decoder(latent) #압축된 latent를 디코더에 넣어 원본 모양으로 복원
        return reconstructed #복원된 데이터 반환
