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
import seaborn as sns
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