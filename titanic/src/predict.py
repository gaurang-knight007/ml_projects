import sys
import joblib
import pandas as pd
from utils import MODELS_DIR, DATA_DIR

MODEL_PATH = MODELS_DIR / "titanic_model.pkl"

def sample_passengers():
    # Minimal fields required by the pipeline. Extra fields are ignored/dropped.
    return pd.DataFrame([
        {
            "PassengerId": 1001,
            "Pclass": 1,
            "Name": "Smith, Mr. John",
            "Sex": "male",
            "Age": 35,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "PC 17599",
            "Fare": 85.0,
            "Cabin": "C85",
            "Embarked": "C"
        },
        {
            "PassengerId": 1002,
            "Pclass": 3,
            "Name": "Doe, Miss. Jane",
            "Sex": "female",
            "Age": 22,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": None,
            "Embarked": "S"
        }
    ])

def main():
    if not MODEL_PATH.exists():
        print("Model not found. Train it first: python src/train.py")
        sys.exit(1)

    pipe = joblib.load(MODEL_PATH)

    # If a CSV path is provided, predict on that; else use samples
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    else:
        df = sample_passengers()
        print("No CSV provided. Using built-in sample passengers.")

    probs = pipe.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = df.copy()
    out["Survival_Prob"] = probs
    out["Predicted_Survived"] = preds

    # Show a compact view
    cols_to_show = ["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survival_Prob","Predicted_Survived"]
    print(out[cols_to_show].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
