import os
import joblib
import numpy as np
import pandas as pd

from utils import ensure_dirs, DATA_DIR, MODELS_DIR
from preprocess import TitleExtractor, FamilySizeAdder, CabinKnown, load_kaggle_or_seaborn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def build_pipeline():
    # Feature engineering steps that add new columns
    fe_steps = [
        ("title", TitleExtractor()),
        ("family", FamilySizeAdder()),
        ("hascabin", CabinKnown())
    ]

    # Base features to use
    categorical = ["Sex", "Embarked", "Title"]
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]

    # Preprocess
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        ],
        remainder="drop"
    )

    # Classifier
    clf = LogisticRegression(max_iter=1000)

    # Full pipeline (FE → Preprocess → Model)
    pipe = Pipeline(steps=[
        *fe_steps,
        ("preprocess", preprocessor),
        ("model", clf)
    ])
    return pipe

def evaluate(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = None
    return metrics

def main():
    ensure_dirs()

    df, source = load_kaggle_or_seaborn(str(DATA_DIR))
    if "Survived" not in df.columns:
        raise RuntimeError("Could not find 'Survived' column in dataset.")

    X = df.drop(columns=["Survived"])
    y = df["Survived"].astype(int)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build & train
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_val)
    y_proba = None
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_val)[:, 1]

    metrics = evaluate(y_val, y_pred, y_proba)
    print(f"Data source: {source}")
    for k, v in metrics.items():
        if v is not None:
            print(f"{k.capitalize()}: {v:.4f}")

    # Save full pipeline (preprocessing + model)
    out_path = MODELS_DIR / "titanic_model.pkl"
    joblib.dump(pipe, out_path)
    print(f"Saved model to: {out_path}")

if __name__ == "__main__":
    main()
