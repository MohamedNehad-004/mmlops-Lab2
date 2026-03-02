import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "metrics.json"
PLOT_PATH = "confusion_matrix.png"

def main():
    df = pd.read_csv(TEST_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved plot: {PLOT_PATH}")

if __name__ == "__main__":
    main()