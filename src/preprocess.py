import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/data.csv"
OUT_DIR = "data/processed"
TRAIN_PATH = os.path.join(OUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUT_DIR, "test.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Dataset has no header. Add column names.
    cols = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "target"
    ]

    df = pd.read_csv(RAW_PATH, header=None, names=cols)

    # Basic cleaning: remove duplicates, fill missing with median
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Saved: {TRAIN_PATH} and {TEST_PATH}")


if __name__ == "__main__":
    main()