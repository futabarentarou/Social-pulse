"""
model.py
--------
Predictive stage of Social Pulse.

Question: using only signals available BEFORE a post proves popular
(sentiment, text characteristics, timing), can we predict which posts will
land in the top 25% of engagement (high_engagement = 1)?

This mirrors the way a social analyst would want an early-warning signal:
"which content is likely to resonate?" We deliberately EXCLUDE the raw
engagement counts (score, num_comments, upvote_ratio) and the ERS itself from
the features, because those are what define the target -- using them would be
data leakage and give a dishonest, inflated result.

Models compared: a majority-class baseline, Logistic Regression, and Random
Forest. We report accuracy, precision, recall, F1 and ROC-AUC, and we read the
feature importances to produce interpretable, actionable insights.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features known at (or near) posting time -- NO leakage from engagement counts.
FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "has_question",
    "has_exclamation",
    "caps_ratio",
    "sentiment_compound",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "hour_utc",
    "dayofweek",
    "is_weekend",
]


def prepare_xy(df: pd.DataFrame):
    data = df.dropna(subset=FEATURE_COLUMNS + ["high_engagement"]).copy()
    X = data[FEATURE_COLUMNS]
    y = data["high_engagement"].astype(int)
    return X, y


def run_models(df: pd.DataFrame, random_state: int = 42):
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Baseline: always predict the majority class.
    majority = int(y_train.mode()[0])
    baseline_pred = np.full(shape=len(y_test), fill_value=majority)
    results["Baseline"] = {
        "accuracy": accuracy_score(y_test, baseline_pred),
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
    }

    # Logistic Regression.
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_s, y_train)
    lr_pred = logreg.predict(X_test_s)
    lr_proba = logreg.predict_proba(X_test_s)[:, 1]
    results["Logistic Regression"] = {
        "accuracy": accuracy_score(y_test, lr_pred),
        "precision": precision_score(y_test, lr_pred, zero_division=0),
        "recall": recall_score(y_test, lr_pred, zero_division=0),
        "f1": f1_score(y_test, lr_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, lr_proba),
    }

    # Random Forest.
    rf = RandomForestClassifier(
        n_estimators=300, random_state=random_state, class_weight="balanced"
    )
    rf.fit(X_train, y_train)  # trees don't need scaling
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred, zero_division=0),
        "recall": recall_score(y_test, rf_pred, zero_division=0),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, rf_proba),
    }

    comparison = pd.DataFrame(results).T
    importances = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    report = classification_report(y_test, rf_pred, zero_division=0)
    return comparison, importances, report, rf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train engagement models.")
    parser.add_argument("--in", dest="inp", default="data/processed/posts_features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    comparison, importances, report, _ = run_models(df)
    print("\n=== Model comparison ===")
    print(comparison.round(3))
    print("\n=== Random Forest feature importances ===")
    print(importances.to_string(index=False))
    print("\n=== Random Forest classification report ===")
    print(report)
