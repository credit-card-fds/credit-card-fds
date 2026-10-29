"""
② 데이터와 모델 준비
- 과제 저장소의 분할 데이터(train/test)를 읽음
- RF 모델은 처음 한 번만 학습해 저장하고, 이후에는 불러오기만 함
"""
import joblib                                             # 파이썬 객체(학습된 모델)를 파일로 저장·불러오기
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import DATA_DIR, MODEL_DIR, MODEL_PATH, RF_PARAMS


def _split_xy(df):
    """데이터프레임을 X(특성)와 y(정답)로 나눔 (과제 utils.load_and_preprocess_data와 같은 규칙)"""
    X = df.drop(columns=["Class"])                        # 정답 열을 뺀 나머지가 특성
    X = X.drop(columns=[c for c in ["Time"] if c in X.columns])   # Time이 남아 있으면 제거 (현재는 Hour만 있음)
    y = df["Class"].values                                # 0 = 정상, 1 = 사기 (numpy 배열)
    return X, y


def load_train_test():
    """Train/Test 데이터를 읽어 (X_train, y_train, X_test, y_test) 반환"""
    # 1. 파일 경로 (Validation은 임계값을 고를 때 이미 썼으므로 에이전트에서는 쓰지 않음)
    train_path = DATA_DIR / "train_processed.csv"         # Path 객체끼리 / 로 경로를 이어 붙임
    test_path = DATA_DIR / "test_processed.csv"

    # 2. 파일이 없으면 원인을 알려 주는 에러를 냄 (과제 utils와 같은 방식)
    missing = [p for p in [train_path, test_path] if not p.exists()]
    if missing:                                           # 리스트가 비어 있지 않으면 True
        names = "\n".join(f" - {p}" for p in missing)
        raise FileNotFoundError(
            f"분할 데이터 파일이 없습니다:\n{names}\n"
            "과제 쪽 experiments/eda_and_split.ipynb를 먼저 실행해 주세요."
        )

    # 3. CSV 읽기 → X, y 나누기
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train = _split_xy(train_df)
    X_test, y_test = _split_xy(test_df)

    return X_train, y_train, X_test, y_test


def get_model(X_train, y_train):
    """저장된 RF 모델이 있으면 불러오고, 없으면 학습 후 저장해서 반환"""
    # 1. 이미 저장된 모델이 있으면 바로 불러옴 (학습 30초~1분 → 불러오기 몇 초)
    if MODEL_PATH.exists():
        print(f"저장된 모델 불러오기: {MODEL_PATH.name}")
        return joblib.load(MODEL_PATH)

    # 2. 없으면 과제 final_report와 같은 설정으로 학습
    print("저장된 모델이 없어 RF를 학습합니다 (30초~1분 걸림)...")
    model = RandomForestClassifier(**RF_PARAMS)           # ** : 딕셔너리를 키워드 인자로 풀어서 전달
    model.fit(X_train, y_train)                           # Train으로만 학습 (과제와 동일)

    # 3. 다음 실행을 위해 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)          # models/ 폴더가 없으면 만듦 (있으면 그냥 넘어감)
    joblib.dump(model, MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")

    return model


# ── 단독 테스트: python data.py 로 실행할 때만 동작 ───────────
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_train_test()
    print("Train:", X_train.shape, "사기", int(y_train.sum()))
    print("Test :", X_test.shape, "사기", int(y_test.sum()))

    model = get_model(X_train, y_train)
    print("모델 트리 수:", len(model.estimators_))
