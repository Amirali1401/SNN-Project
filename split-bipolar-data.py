import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_FILE = "Top - 100 - Bipolar.csv"

TRAIN_FILE = "bipolar_train.csv"
TEST_FILE = "bipolar_test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


df = pd.read_csv(INPUT_FILE)

if "ID_REF" in df.columns:
    df = df.drop(columns=["ID_REF"])


target_col = df.columns[-1]

print("Target column:", target_col)
print("Total samples:", len(df))

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[target_col]
)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

print("\nFiles created successfully:")
print(TRAIN_FILE, len(train_df))
print(TEST_FILE, len(test_df))
