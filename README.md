# 💳 신용카드 이상거래 탐지 시스템 (FDS)

> **0.17% 극불균형 거래 환경에서의 운영 비용 최적화 기반 사기 탐지 모델 구축**

---

## 1. 프로젝트 개요 및 비즈니스 문제 정의

- **배경**: 유럽 카드사의 이틀간 거래 데이터(총 284,807건) 중 실제 사기 거래는 492건(**0.17%**)에 불과한 극도로 불균형한 데이터 환경입니다.
- **핵심 문제**: 
  - 단순 분류 모델을 사용하면 모든 거래를 "정상"으로만 예측해도 **정확도가 99.83%**에 달하는 **"정확도의 함정"**이 발생합니다.
  - 사기 거래를 놓쳤을 때 발생하는 **직접적인 금전 피해**와, 정상 거래를 사기로 잘못 판단했을 때 발생하는 **검토 인건비(운영 비용)** 사이의 상충 관계를 해결해야 합니다.

---

## 2. 프로젝트 목표 및 평가 기준

### ① 기술적 평가 지표 (모델 성능)
- **주요 지표**: **정밀도-재현율 곡선 아래 면적 (PR-AUC / AP)**
  - 희소한 사기 데이터 분류 성능을 부풀림 없이 객관적으로 평가하기 위해 사용합니다.

### ② 비즈니스/운영 지표 (실무 운영 기준)
- **목표**: **사기 탐지율(재현율) 80% 이상을 유지하면서, 하루 동안 담당 요원이 직접 검토해야 하는 건수를 최소화합니다.**
- **총 손실 비용 계산식**:
  $$\text{총 손실 비용} = (\text{놓친 사기 건수} \times \text{건당 사기 피해액}) + (\text{잘못 짚은 정상 건수} \times \text{건당 검토 인건비})$$

---

## 3. 프로젝트 주요 차별화 포인트

1. **데이터 누수 차단**:
   - **그룹 단위 분할**: 같은 거래의 복사본(중복 사기 거래)이 Train과 Validation/Test에 나뉘어 들어가지 않도록 그룹 단위로 계층 분할합니다. 교차검증 fold도 같은 원칙으로 나눕니다.
   - **분할 결과**: Train / Validation / Test = 70 / 15 / 15%, 사기 거래 347 / 73 / 72건. 세 데이터셋 사이에 겹치는 행이 0건인지 `assert`로 확인합니다.
   - **학습 구간 내 전처리**: `Amount`는 `log1p` 변환 후 Train으로만 fit한 `RobustScaler`를 적용하고, `Time`은 시간대(`Hour`)로 바꿉니다. 교차검증에서는 오버샘플링(SMOTE)을 Pipeline 안에 넣어 학습 fold에서만 수행합니다.
2. **지도학습 vs 비지도 이상 탐지 비교**:
   - 라벨이 없는 신종 사기 패턴에 대응하기 위해 고립 숲(Isolation Forest), PCA 복원오차, 오토인코더(Autoencoder) 등 이상 탐지 기법을 정답 기반 모델(RandomForest/XGBoost/LightGBM)과 같은 지표·같은 임계값 규칙으로 Test에서 비교합니다.
   - 비지도 모델은 **정상 거래로만** 학습하며, PCA·오토인코더 입력은 정상 Train 기준 `StandardScaler`로 맞춥니다.
3. **설명 가능한 인공지능 (XAI) 도입**:
   - SHAP 기법을 활용하여 차단된 거래가 왜 사기로 의심되었는지 구체적인 사유(변수별 영향도)를 상담원 관점에서 시각화하여 제공합니다.

---

## 4. 최종 성과 및 결론

> 모든 수치는 Validation에서 모델·임계값을 확정한 뒤 **Test Set에 한 번만** 적용한 결과입니다. (`experiments/final_report.ipynb`)

### ① 최종 모델 (Test)
| 항목 | 결과 |
|---|---|
| 최종 모델 | **RandomForest** (`class_weight='balanced'`), 운영 임계값 **0.15** |
| PR-AUC | **0.800** (베이스라인 로지스틱 회귀 0.732, 무작위 0.0017) |
| Recall / Precision | **0.79** / 0.84 (사기 72건 중 57건 탐지, 목표 0.80에 1건 차이) |
| 검토 건수 | Test 68건 → **하루 약 227건** |
| 총 손실 비용 (가정: 사기 122 EUR, 검토 5 EUR) | Test 1,885 EUR → 하루 약 6,284 EUR (기본 임계값 0.5 대비 약 19% 절감) |

### ② 지도학습 vs 비지도 이상 탐지 (Test, 같은 임계값 규칙)
| 구분 | 모델 | PR-AUC | Recall | 검토 건수 |
|---|---|---|---|---|
| 지도 | RandomForest | 0.800 | 0.79 | 68 |
| 지도 | XGBoost | 0.808 | 0.81 | 76 |
| 비지도 | PCA 복원오차 | 0.684 | 0.82 | 1,581 |
| 비지도 | Autoencoder | 0.314 | 0.81 | 1,845 |
| 비지도 | Isolation Forest | 0.147 | 0.81 | 1,517 |

- 같은 Recall을 내려면 비지도 모델은 검토 건수가 **약 20~27배** 필요합니다.
- 라벨이 있으면 지도학습이 압도적이고, 비지도(특히 PCA)는 **라벨 없는 신종 사기 감시용 보조 탐지기**로 의미가 있습니다.

### ③ 모델·불균형 처리 비교 (반복 교차검증, Train 내부 5-fold × 2회)
| 순위 | 모델 | PR-AUC (평균 ± 편차) | Recall 0.8 검토 건수 (평균) | 학습 시간 |
|---|---|---|---|---|
| 1 | LightGBM (scale_pos_weight) | **0.860 ± 0.034** | 58.6 | 4.0초 |
| 2 | RandomForest + SMOTE | 0.855 ± 0.030 | 59.9 | 28.8초 |
| 3 | XGBoost (scale_pos_weight) | 0.853 ± 0.038 | 60.9 | 1.5초 |
| 4 | RandomForest (class_weight) | 0.847 ± 0.037 | 60.3 | 16.6초 |
| 5 | LogReg + SMOTE | 0.771 ± 0.037 | 74.0 | 1.0초 |
| 6 | LogReg (class_weight) | 0.747 ± 0.045 | 79.7 | 0.8초 |

- **트리 모델 4개(0.847~0.860)는 로지스틱 회귀(0.747~0.771)보다 확실히 낫습니다.** 차이(약 0.08~0.11)가 표준편차(약 0.03~0.04)보다 훨씬 큽니다.
- **트리 모델끼리의 차이(최대 0.013)는 표준편차보다 작아서 "사실상 비슷하다"** 고 봐야 합니다. 같은 fold끼리 짝지어 비교하면 LightGBM이 RF(class_weight)를 10번 중 8번 이겨 약간의 우위는 있지만, 운영 모델로 RF를 고른 판단을 뒤집을 만큼은 아닙니다.
- **SMOTE vs class_weight**: 로지스틱 회귀에서는 SMOTE가 +0.024(10번 중 8번 승), RF에서는 +0.008(7번 승)입니다. 선형 모델에서 효과가 더 크고, 트리 모델에서는 작으면서 학습 시간은 약 1.7배 늘어납니다.
- **LightGBM**은 기본 설정으로는 무너졌고, `boost_from_average=False`와 `reg_lambda=1`을 넣은 뒤에야 가장 좋은 성능이 나왔습니다. 극불균형에서는 라이브러리 기본값을 꼭 점검해야 합니다.

### ④ 핵심 인사이트
1. **정확도의 함정**: 전부 정상으로 예측해도 Accuracy 0.998 → PR-AUC와 검토 건수로 평가해야 합니다.
2. **임계값이 성능만큼 중요**: 임계값을 0.5 → 0.15로 낮추자 사기를 4건 더 잡고 비용이 약 19% 줄었습니다.
3. **class_weight는 순위가 아니라 기준선을 바꿈**: 로지스틱 회귀에 가중치를 넣자 Recall 0.51 → 0.89, PR-AUC는 0.717 → 0.715로 그대로였습니다.
4. **누수와 표본 오차**: 중복 사기 거래가 Train/Test에 걸쳐 있던 누수를 그룹 분할로 제거했습니다. 사기가 70여 건뿐이라 분할 한 번의 수치는 흔들림이 크므로 반복 CV로 함께 판단했습니다.
5. **분할 방식 비교**: 시간 순서 분할(앞 70% 학습)은 Validation PR-AUC 0.878(사기 56건), 계층 + 그룹 분할은 0.857(사기 73건)로 큰 차이가 없었습니다. 데이터가 이틀치뿐이라 시간에 따른 패턴 변화보다 사기 건수 확보가 중요하다고 보고 계층 + 그룹 분할을 택했습니다.
6. **설명 가능성**: SHAP 상위 변수는 V14, V12, V4, V10, V3이며, EDA에서 정상·사기 평균 차이가 컸던 변수와 일치합니다.

---

## 5. 실행 방법

1. **환경**: Python 3.10, `pip install -r requirements.txt`
2. **데이터**: Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)의 `creditcard.csv`를 `data/raw/`에 넣습니다.
3. **노트북을 순서대로 실행**합니다. 뒤 노트북이 앞 노트북의 결과 CSV를 읽으므로 순서를 바꾸면 안 됩니다.

| 순서 | 노트북 | 하는 일 |
|---|---|---|
| 1 | `eda_and_split` | EDA, 그룹 계층 분할, 전처리 → `data/processed/` 생성 |
| 2 | `baseline_evaluation` | Dummy, 로지스틱 회귀 기준선 |
| 3 | `supervised_modeling` | RF·XGBoost 임계값 선택, 반복 교차검증 (약 9분 소요) |
| 4 | `unsupervised_anomaly` | Isolation Forest, PCA, Autoencoder |
| 5 | `final_report` | Test 1회 평가, 비용, 지도 vs 비지도, SHAP |

> **CSV와 그래프는 Git에 올리지 않습니다.** `data/`, `reports/tables/`, `reports/figures/`는 저장소를 받은 뒤 노트북을 실행해야 생성됩니다.

---

## 6. 프로젝트 구조
```text
credit-card-fds/
├── .gitignore
├── README.md
├── requirements.txt
├── data/                         # Git 제외
│   ├── raw/                      # 원본 데이터 (creditcard.csv)
│   └── processed/                # 전처리된 Split 데이터 (train/validation/test_processed.csv)
├── experiments/                  # 실험 노트북 (위 실행 순서대로)
│   ├── eda_and_split.ipynb       # [예목 담당] EDA, 데이터 정제 및 Train/Val/Test 분할
│   ├── baseline_evaluation.ipynb # [동경 담당] Baseline(Dummy, 로지스틱 회귀) 및 정확도의 함정 확인
│   ├── supervised_modeling.ipynb # [예목&동경 담당] RF·XGBoost 임계값 선택 & 반복 교차검증(SMOTE, LightGBM)
│   ├── unsupervised_anomaly.ipynb# [동경 담당] 비지도 이상 탐지 (Isolation Forest, PCA, Autoencoder)
│   └── final_report.ipynb        # [동경 담당] Test Set 최종 평가, 지도 vs 비지도 비교 & SHAP(XAI)
├── src/                          # 공통 모듈화 파이썬 코드
│   ├── __init__.py
│   └── utils.py                  # 데이터 로드, 평가·임계값 시뮬레이션, 시각화, Autoencoder 공통 함수
└── reports/                      # 산출물 (Git 제외, 노트북 실행 시 생성)
    ├── figures/                  # EDA 및 모델 평가 차트 이미지
    └── tables/                   # 결과표 CSV ("단계_데이터셋_내용.csv")
```

### 결과표 (`reports/tables/`)
파일 이름은 `단계_데이터셋_내용.csv` 규칙을 따릅니다. (`val` = Validation, `test` = Test)

| 생성 노트북 | 파일 | 내용 |
|---|---|---|
| `eda_and_split` | `eda_val_split_comparison.csv` | 시간 순서 분할 vs 계층 + 그룹 분할 비교 |
| `baseline_evaluation` | `baseline_val_metrics.csv`, `baseline_val_thresholds.csv` | 기준선 성능, 로지스틱 회귀 임계값별 검토 건수 |
| `supervised_modeling` | `supervised_val_metrics.csv`, `supervised_val_thresholds.csv`, `supervised_val_candidates.csv` | RF·XGBoost 성능, 임계값별 검토 건수, 운영 임계값 후보 |
| `supervised_modeling` | `supervised_cv_folds.csv`, `supervised_cv_summary.csv` | 반복 교차검증 fold별 결과와 평균 ± 표준편차 |
| `unsupervised_anomaly` | `unsupervised_val_metrics.csv`, `unsupervised_val_thresholds.csv` | 비지도 모델 성능(Recall 0.8 임계값 포함), 임계값별 검토 건수 |
| `final_report` | `final_test_metrics.csv`, `final_test_thresholds.csv`, `final_test_cost.csv` | Test 성능, 임계값별 검토 건수, 비용 시뮬레이션 |
| `final_report` | `final_test_model_comparison.csv`, `final_test_shap_importance.csv` | 지도 vs 비지도 Test 비교, SHAP 변수 중요도 |

---

## 7. 한계
- **익명화된 변수**: V1~V28은 PCA로 변환된 값이라 SHAP 상위 변수가 실제로 무엇을 뜻하는지(가맹점, 위치 등) 해석할 수 없습니다.
- **짧은 기간**: 이틀치 데이터라 계절·요일에 따른 사기 패턴 변화를 검증할 수 없습니다.
- **작은 사기 표본**: Test 사기가 72건이라 1건 차이로 Recall이 약 0.014 움직입니다. (최종 Recall 0.79는 목표 0.80에 1건 부족)
- **비용 가정**: 사기 1건 122 EUR, 검토 1건 5 EUR은 가정값이며, 실제 비용에 따라 최적 임계값이 달라집니다.
