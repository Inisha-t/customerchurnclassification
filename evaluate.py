import joblib
import tensorflow as tf
from pathlib import Path

from config import OUTPUT_MODELS_DIR, OUTPUT_REPORTS_DIR, FIGURES_DIR
from data_utils import train_test_data
from metrics_utils import compute_metrics, save_json, plot_roc

def eval_joblib_model(model_path: Path, X_test, y_test):
    pipe = joblib.load(model_path)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    return y_pred, y_proba

def eval_deep_model(X_test, y_test):
    pre = joblib.load(OUTPUT_MODELS_DIR / "deep_preprocessor.joblib")
    model = tf.keras.models.load_model(OUTPUT_MODELS_DIR / "deep_mlp.keras")

    X_t = pre.transform(X_test)
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else X_t

    y_proba = model.predict(X_t, verbose=0).reshape(-1)
    y_pred = (y_proba >= 0.5).astype(int)
    return y_pred, y_proba

def main():
    X_train, X_test, y_train, y_test, _ = train_test_data()

    OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Classical models
    for name in ["glm_logistic", "random_forest", "svm_rbf"]:
        path = OUTPUT_MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"Missing {path}. Train first.")
            continue
        y_pred, y_proba = eval_joblib_model(path, X_test, y_test)
        results[name] = compute_metrics(y_test, y_pred, y_proba)
        if y_proba is not None:
            plot_roc(y_test, y_proba, f"ROC - {name}", FIGURES_DIR / f"roc_{name}.png")

    # Deep model
    deep_model_path = OUTPUT_MODELS_DIR / "deep_mlp.keras"
    deep_pre_path = OUTPUT_MODELS_DIR / "deep_preprocessor.joblib"
    if deep_model_path.exists() and deep_pre_path.exists():
        y_pred, y_proba = eval_deep_model(X_test, y_test)
        results["deep_mlp"] = compute_metrics(y_test, y_pred, y_proba)
        plot_roc(y_test, y_proba, "ROC - deep_mlp", FIGURES_DIR / "roc_deep_mlp.png")
    else:
        print("Deep model not found. Train deep model first.")

    save_json(results, OUTPUT_REPORTS_DIR / "metrics.json")
    print("Saved reports/metrics.json and ROC figures.")

if __name__ == "__main__":
    main()
