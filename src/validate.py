# src/validate.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/model.joblib"

METRICS_PATH = "metrics.json"
PLOT_PNG_PATH = "confusion_matrix.png"

PLOTS_DIR = "plots"
CM_CSV_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.csv")


def main():
    # Load test data
    df = pd.read_csv(TEST_PATH)
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {METRICS_PATH}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix as CSV (for DVC plots diff)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.savetxt(CM_CSV_PATH, cm, delimiter=",", fmt="%d")
    print(f"Saved plot data: {CM_CSV_PATH}")

    # Save confusion matrix as PNG (nice to look at)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(PLOT_PNG_PATH, dpi=200)
    plt.close()
    print(f"Saved plot: {PLOT_PNG_PATH}")


if __name__ == "__main__":
    main()