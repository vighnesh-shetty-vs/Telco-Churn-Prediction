"""Evaluate the churn model against a business-rule baseline and alternatives.

Same data preparation and split as app.py. Writes reports/metrics.md.
Usage: python evaluate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT = Path("reports")


def load():
    df = pd.read_csv("data/Telco_Customer_Dataset.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df, pd.get_dummies(df, drop_first=True)


def scores(name, y, pred, prob=None):
    return {"model": name, "recall": f"{recall_score(y, pred):.1%}", "precision": f"{precision_score(y, pred):.1%}",
            "F1": f"{f1_score(y, pred):.3f}", "ROC-AUC": "" if prob is None else f"{roc_auc_score(y, prob):.3f}",
            "PR-AUC": "" if prob is None else f"{average_precision_score(y, prob):.3f}"}


def main():
    raw, df = load()
    X, y = df.drop("Churn", axis=1), df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    raw_te = raw.loc[X_te.index]
    sc = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)

    rows = [scores("Rule: month-to-month contract", y_te, (raw_te["Contract"] == "Month-to-month").astype(int))]
    plain = LogisticRegression(max_iter=1000).fit(Xs_tr, y_tr)
    rows.append(scores("Logistic regression (no rebalancing)", y_te, plain.predict(Xs_te), plain.predict_proba(Xs_te)[:, 1]))
    Xr, yr = SMOTE(random_state=42).fit_resample(Xs_tr, y_tr)
    smote = LogisticRegression(max_iter=1000).fit(Xr, yr)
    p_smote = smote.predict_proba(Xs_te)[:, 1]
    rows.append(scores("Logistic regression + SMOTE (app)", y_te, smote.predict(Xs_te), p_smote))
    bal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs_tr, y_tr)
    rows.append(scores("Logistic regression, class_weight='balanced'", y_te, bal.predict(Xs_te), bal.predict_proba(Xs_te)[:, 1]))
    table = pd.DataFrame(rows)

    # Targeting: how many churners are in the riskiest 20% / 30% of customers?
    order = np.argsort(-p_smote)
    capture = {k: y_te.to_numpy()[order[: int(k * len(order))]].sum() / y_te.sum() for k in (0.2, 0.3)}

    # Churn rate by the drivers the model leans on most (on the full dataset)
    seg = (raw.assign(tenure_band=pd.cut(raw["tenure"], [0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]))
           .groupby(["Contract", "InternetService"], observed=True)["Churn"].agg(customers="size", churn_rate="mean")
           .sort_values("churn_rate", ascending=False).head(5))
    coefs = pd.Series(smote.coef_[0], index=X.columns).sort_values()

    OUT.mkdir(exist_ok=True)
    md = ["# Model evaluation (20% test set)", "",
          f"Test set: {len(y_te):,} customers, churn rate {y_te.mean():.1%}.", "",
          "| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns),
          *["| " + " | ".join(r) + " |" for r in table.astype(str).itertuples(index=False)], "",
          f"Targeting with the app's model: the riskiest 20% of customers contain {capture[0.2]:.0%} of all churners, "
          f"the riskiest 30% contain {capture[0.3]:.0%}.", "",
          "## Highest-churn segments (full dataset)", "",
          "| Contract | Internet | Customers | Churn rate |", "|---|---|---|---|",
          *[f"| {c} | {i} | {int(r.customers):,} | {r.churn_rate:.1%} |" for (c, i), r in seg.iterrows()], "",
          "## Strongest model drivers (standardised coefficients)", "",
          "Raise churn: " + ", ".join(f"{k} ({v:+.2f})" for k, v in coefs.tail(4)[::-1].items()), "",
          "Lower churn: " + ", ".join(f"{k} ({v:+.2f})" for k, v in coefs.head(4).items())]
    (OUT / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
