"""Model loading, scoring, reasons, suggested actions and what-if checks."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load():
    model = joblib.load(ROOT / "models" / "churn_model.joblib")
    info = json.loads((ROOT / "models" / "model_info.json").read_text())
    return model, info


MONEY = ["Balance", "EstimatedSalary"]


def in_euros(X, currency="EUR"):
    """The model learned money in euros, so amounts in other currencies are converted before scoring."""
    from loader import TO_EURO
    rate = TO_EURO.get(currency, 1.0)
    if rate == 1.0:
        return X
    X = X.copy()
    for col in MONEY:
        X[col] = X[col].astype(float) * rate
    return X


def score(model, X, currency="EUR"):
    X = in_euros(X, currency)
    numbers = X.select_dtypes("number").columns
    if len(numbers) and not np.isfinite(X[numbers].to_numpy(dtype=float)).all():   # never let one bad cell crash scoring
        X = X.copy()
        X[numbers] = X[numbers].replace([np.inf, -np.inf], np.nan).clip(-1e15, 1e15)
    return model.predict_proba(X)[:, 1]


def level(prob, threshold, base_rate):
    if prob >= threshold:
        return "High"
    return "Medium" if prob >= base_rate else "Low"


def reasons(row):
    """Plain-English reasons, based on the patterns found in the analysis notebook."""
    out = []
    if 45 <= row["Age"] <= 60:
        out.append("Age 45-60: the highest-churn age group (about 50% leave)")
    if row["NumOfProducts"] >= 3:
        out.append("3+ products: 83-100% of these customers leave")
    elif row["NumOfProducts"] == 1:
        out.append("Only 1 product: 28% churn vs 8% with 2 products")
    if row["IsActiveMember"] == 0:
        out.append("Inactive member: churns about twice as often")
    if row["Geography"] == "Germany":
        out.append("Germany: churn is double France and Spain")
    return out


def actions(row, currency="EUR"):
    out = []
    if row["IsActiveMember"] == 0:
        out.append("Start a re-engagement campaign")
    if row["NumOfProducts"] == 1:
        out.append("Offer a second product")
    if row["NumOfProducts"] >= 3:
        out.append("Review their product bundle")
    if 45 <= row["Age"] <= 60:
        out.append("Assign a relationship manager")
    from loader import TO_EURO
    if row["Balance"] * TO_EURO.get(currency, 1.0) >= 100_000:       # a high balance, measured in euros
        out.append("Priority call from a senior advisor")
    return out or ["Keep in the regular loyalty programme"]


def what_if(model, row, currency="EUR"):
    """How the risk would change if the bank fixed what it can influence."""
    base = float(score(model, pd.DataFrame([row]), currency)[0])
    changes = []
    if row["IsActiveMember"] == 0:
        changes.append(("If they become an active member", {**row, "IsActiveMember": 1}))
    if row["NumOfProducts"] != 2:
        changes.append(("If they hold 2 products", {**row, "NumOfProducts": 2}))
    if row["IsActiveMember"] == 0 and row["NumOfProducts"] != 2:
        changes.append(("If both happen", {**row, "IsActiveMember": 1, "NumOfProducts": 2}))
    return base, [(label, float(score(model, pd.DataFrame([r]), currency)[0])) for label, r in changes]


def main_reason(row):
    r = reasons(row)
    return r[0].split(":")[0] if r else "No major risk sign"


def levels(probs, threshold, base_rate):
    """Risk level for a whole column at once (fast for millions of rows)."""
    return np.where(probs >= threshold, "High", np.where(probs >= base_rate, "Medium", "Low"))


def main_reasons(X):
    """Main reason for a whole table at once, in the same order as reasons()."""
    conditions = [X["Age"].between(45, 60), X["NumOfProducts"] >= 3, X["NumOfProducts"] == 1,
                  X["IsActiveMember"] == 0, X["Geography"] == "Germany"]
    labels = ["Age 45-60", "3+ products", "Only 1 product", "Inactive member", "Germany"]
    return np.select(conditions, labels, default="No major risk sign")


# How much the tuned model relies on each input (permutation importance from the notebook).
# Tiny values get a small floor so every column counts for something.
IMPORTANCE = {"Age": 0.126, "NumOfProducts": 0.113, "IsActiveMember": 0.034, "Balance": 0.026,
              "Geography": 0.019, "Gender": 0.008, "CreditScore": 0.002, "HasCrCard": 0.002,
              "Tenure": 0.002, "EstimatedSalary": 0.002}


def coverage(real_share):
    """Share of the model's information the data really provides, weighted by importance.

    real_share maps an input to the share of customers with a real (not guessed) value;
    inputs left out count as fully present. 1.0 means nothing had to be guessed.
    """
    total = sum(IMPORTANCE.values())
    return sum(IMPORTANCE[f] * real_share.get(f, 1.0) for f in IMPORTANCE) / total


def quality(cov):
    if cov >= 0.9:
        return "Excellent"
    if cov >= 0.6:
        return "Good"
    return "Limited" if cov >= 0.3 else "Very limited"


def evaluate(outcome, probs, threshold):
    """How well the model did on a file that includes what really happened (1 = left, 0 = stayed)."""
    from sklearn.metrics import roc_auc_score
    known = ~np.isnan(outcome)
    y, p = outcome[known].astype(int), probs[known]
    out = {"n": int(len(y)), "unknown": int((~known).sum()), "left": int(y.sum()) if len(y) else 0}
    out["rate"] = out["left"] / out["n"] if out["n"] else 0.0
    out["auc"] = float(roc_auc_score(y, p)) if out["n"] and 0 < out["left"] < out["n"] else None
    flagged = p >= threshold
    caught = int((flagged & (y == 1)).sum())
    out.update(flagged=int(flagged.sum()), caught=caught,
               recall=caught / out["left"] if out["left"] else None,
               precision=caught / flagged.sum() if flagged.sum() else None,
               accuracy=float((flagged == (y == 1)).mean()) if out["n"] else None)
    if out["left"]:
        top = np.argsort(-p)[: max(1, int(np.ceil(0.2 * len(p))))]
        out["top20"] = float(y[top].sum() / out["left"])
    else:
        out["top20"] = None
    return out


def auc_label(auc):
    if auc is None:
        return "Can't measure"
    if auc >= 0.8:
        return "Strong"
    if auc >= 0.7:
        return "Good"
    if auc >= 0.6:
        return "Fair"
    return "Weak (close to guessing)"
