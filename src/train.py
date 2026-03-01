import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

TRAIN_PATH = "data/processed/train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
PARAMS_PATH = "params.json"


def load_params():
    # default params if params.json not found
    params = {"model_type": "logreg"}
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            params.update(json.load(f))
    return params


def build_model(model_type: str):
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=2000)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        return pipe

    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        return clf

    if model_type == "svc":
        clf = SVC(kernel="rbf", probability=True)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        return pipe

    raise ValueError("model_type must be one of: logreg, rf, svc")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(TRAIN_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    params = load_params()
    model_type = params.get("model_type", "logreg")

    model = build_model(model_type)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH} (model_type={model_type})")


if __name__ == "__main__":
    main()