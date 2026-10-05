import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "Top - 100 - Schz .csv"
TRAIN_FILE = "schz_train.csv"
TEST_FILE = "schz_test.csv"


def main():
    folder = Path(__file__).resolve().parent

    input_path = folder / INPUT_FILE
    train_path = folder / TRAIN_FILE
    test_path = folder / TEST_FILE

    if not input_path.exists():
        print("Input file not found:")
        print(input_path)
        return 1

    try:
        data = pd.read_csv(
            input_path,
            sep=",",
            encoding="utf-8"
        )
    except Exception as error:
        print("Could not read input file:")
        print(error)
        return 1

    if "ID_REF" not in data.columns:
        print("Column ID_REF was not found.")
        return 1

    if "Group" not in data.columns:
        print("Column Group was not found.")
        return 1

    if data.empty:
        print("The input file is empty.")
        return 1

    labels = data["Group"].astype(str).str.strip().str.lower()
    valid_labels = {"control", "schizophrenia"}

    if set(labels.unique()) != valid_labels:
        print("Invalid labels found:")
        print(labels.value_counts())
        return 1

    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        stratify=data["Group"],
        random_state=42
    )

    train_data.to_csv(train_path, index=False, encoding="utf-8")
    test_data.to_csv(test_path, index=False, encoding="utf-8")

    print("Split completed successfully.")
    print("Train rows:", len(train_data))
    print("Test rows:", len(test_data))

    print("Train Group counts:")
    print(train_data["Group"].value_counts())

    print("Test Group counts:")
    print(test_data["Group"].value_counts())

    print("Train file:", train_path)
    print("Test file:", test_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
