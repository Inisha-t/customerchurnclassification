import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import DATA_PATH, TARGET_COL, RANDOM_SEED, TEST_SIZE

def load_telco_csv() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Put the churn CSV into data/raw/ and ensure the filename matches config.DATA_PATH."
        )
    df = pd.read_csv(DATA_PATH)

    # Common cleanup for this dataset
    # TotalCharges sometimes loads as object with blanks; coerce to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df

def split_xy(df: pd.DataFrame):
    y = df[TARGET_COL].astype(str).map({"Yes": 1, "No": 0})
    X = df.drop(columns=[TARGET_COL])

    # Drop ID column if present
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    return X, y

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor

def train_test_data():
    df = load_telco_csv()
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = make_preprocessor(X_train)
    return X_train, X_test, y_train, y_test, preprocessor
