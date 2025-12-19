import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from config import OUTPUT_MODELS_DIR, RANDOM_SEED
from data_utils import train_test_data

def main():
    X_train, X_test, y_train, y_test, preprocessor = train_test_data()

    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "glm_logistic": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "svm_rbf": SVC(probability=True, random_state=RANDOM_SEED),
    }

    for name, clf in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, OUTPUT_MODELS_DIR / f"{name}.joblib")
        print(f"Saved: {name}.joblib")

if __name__ == "__main__":
    main()
