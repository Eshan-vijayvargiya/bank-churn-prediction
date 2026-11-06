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


def score(model, X):
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


def actions(row):
    out = []
    if row["IsActiveMember"] == 0:
        out.append("Start a re-engagement campaign")
    if row["NumOfProducts"] == 1:
        out.append("Offer a second product")
    if row["NumOfProducts"] >= 3:
        out.append("Review their product bundle")
    if 45 <= row["Age"] <= 60:
        out.append("Assign a relationship manager")
    if row["Balance"] >= 100_000:
        out.append("Priority call from a senior advisor")
    return out or ["Keep in the regular loyalty programme"]


def what_if(model, row):
    """How the risk would change if the bank fixed what it can influence."""
    base = float(score(model, pd.DataFrame([row]))[0])
    changes = []
    if row["IsActiveMember"] == 0:
        changes.append(("If they become an active member", {**row, "IsActiveMember": 1}))
    if row["NumOfProducts"] != 2:
        changes.append(("If they hold 2 products", {**row, "NumOfProducts": 2}))
    if row["IsActiveMember"] == 0 and row["NumOfProducts"] != 2:
        changes.append(("If both happen", {**row, "IsActiveMember": 1, "NumOfProducts": 2}))
    return base, [(label, float(score(model, pd.DataFrame([r]))[0])) for label, r in changes]


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
