import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------- Custom Transformers ----------

class TitleExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 'Title' from the 'Name' column (Mr, Mrs, Miss, Master, etc.)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        titles = X["Name"].fillna("").apply(self._extract_title)
        X["Title"] = titles.replace({
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Lady": "Royalty",
            "Countess": "Royalty",
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Dr": "Officer",
            "Rev": "Officer",
            "Sir": "Royalty",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Dona": "Royalty"
        })
        return X

    @staticmethod
    def _extract_title(name):
        m = re.search(r",\s*([^\.]+)\.", name)
        return m.group(1).strip() if m else "Unknown"


class FamilySizeAdder(BaseEstimator, TransformerMixin):
    """
    Creates FamilySize = SibSp + Parch + 1
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X["SibSp"].fillna(0).astype(int) + X["Parch"].fillna(0).astype(int) + 1
        return X


class CabinKnown(BaseEstimator, TransformerMixin):
    """
    Adds boolean 'HasCabin' (1 if Cabin non-null else 0)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HasCabin"] = X["Cabin"].notna().astype(int)
        return X


def load_kaggle_or_seaborn(data_dir):
    """
    Load Kaggle train.csv if available; otherwise load seaborn titanic dataset
    and adapt it to a Kaggle-like schema.
    """
    import os
    import seaborn as sns

    kaggle_path = os.path.join(data_dir, "train.csv")
    if os.path.exists(kaggle_path):
        df = pd.read_csv(kaggle_path)
        source = "kaggle"
        return df, source

    # Fallback: seaborn titanic
    s = sns.load_dataset("titanic")  # columns differ
    # Map to Kaggle-like names and values (best-effort)
    # seaborn columns: survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male...
    df = s.copy()
    df.rename(columns={
        "survived": "Survived",
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked"
    }, inplace=True)

    # Fill essential missing Kaggle columns for pipeline compatibility
    for col in ["Name", "Ticket", "Cabin"]:
        if col not in df.columns:
            df[col] = None

    # Reorder approximate Kaggle columns
    desired_cols = [
        "PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"
    ]
    if "PassengerId" not in df.columns:
        df["PassengerId"] = range(1, len(df) + 1)
    df = df[[c for c in desired_cols if c in df.columns]]
    source = "seaborn"
    return df, source
