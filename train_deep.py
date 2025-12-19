import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.pipeline import Pipeline

from config import OUTPUT_MODELS_DIR, RANDOM_SEED
from data_utils import train_test_data

def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_model(input_dim: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model

def main():
    set_seeds(RANDOM_SEED)

    X_train, X_test, y_train, y_test, preprocessor = train_test_data()

    # Fit preprocessor on training data and transform
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Dense model needs dense arrays
    X_train_t = X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t
    X_test_t = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t

    model = build_model(X_train_t.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)
    ]

    model.fit(
        X_train_t, y_train.values,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save keras model
    keras_path = OUTPUT_MODELS_DIR / "deep_mlp.keras"
    model.save(keras_path)

    # Save preprocessor separately for deep model
    joblib.dump(preprocessor, OUTPUT_MODELS_DIR / "deep_preprocessor.joblib")

    print(f"Saved: {keras_path}")
    print("Saved: deep_preprocessor.joblib")

if __name__ == "__main__":
    main()
