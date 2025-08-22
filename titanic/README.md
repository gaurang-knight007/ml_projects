# ▶️ How to Run

Create the folders & install deps

## clone the repo

## install requirements
pip install -r requirements.txt


## Train
```bash
python src/train.py
```

You’ll see metrics like Accuracy, Precision, Recall, F1, ROC AUC and a message:
Saved model to: .../models/titanic_model.pkl

## Predict (built-in samples)
```bash
python src/predict.py
```

## Predict on your own CSV (Kaggle-like columns)
```bash
python src/predict.py data/test.csv
```

This prints survival probabilities and class (0/1).
(If your CSV is Kaggle test.csv, note it lacks the Survived column—that’s fine for prediction.)

# ✅ Notes & Tips

The model is Logistic Regression with simple feature engineering (Title, FamilySize, HasCabin) and standard preprocessing (median/mode imputation, OHE, scaling).

Because the entire pipeline is saved, you don’t need to reapply preprocessing manually during prediction.

You can upgrade to RandomForestClassifier or XGBoost by swapping the estimator in build_pipeline().