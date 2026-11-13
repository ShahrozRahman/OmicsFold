"""One engine shared by Python, the CLI, and R."""
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Result:
    predictions: pd.DataFrame
    folds: list
    metrics: dict
    provenance: dict
    exploratory_rankings: pd.DataFrame

    def write(self, directory):
        """Write to a new directory; never overwrite an existing analysis."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.predictions.to_csv(path / "predictions.csv", index=False)
        self.exploratory_rankings.to_csv(path / "exploratory_rankings.csv", index=False)
        for name in ("folds", "metrics", "provenance"):
            (path / f"{name}.json").write_text(
                json.dumps(getattr(self, name), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _ids(frame, label):
    if "sample_id" not in frame:
        raise ValueError(f"{label} requires sample_id")
    ids = frame.sample_id
    if ids.isna().any() or not ids.map(lambda x: isinstance(x, str) and bool(x.strip())).all() or ids.duplicated().any():
        raise ValueError(f"{label} sample IDs must be unique, nonempty strings")


def _fingerprint(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def analyze(features, metadata, *, positive_label, group_columns=(), n_splits=5, seed=42, top_k=None):
    """Evaluate fixed logistic regression with out-of-fold predictions.

    Both frames require sample_id; metadata requires outcome. Feature rows are
    samples. Multiple group columns are joined by connected components, so any
    shared patient/family/gene/cohort value prevents a cross-fold split.
    """
    features, metadata = features.copy(), metadata.copy()
    _ids(features, "features")
    _ids(metadata, "metadata")
    if features.columns.duplicated().any() or metadata.columns.duplicated().any():
        raise ValueError("Column names must be unique")
    if set(features.sample_id) != set(metadata.sample_id):
        raise ValueError(f"Unmatched IDs: matrix-only={sorted(set(features.sample_id) - set(metadata.sample_id))}; "
                         f"metadata-only={sorted(set(metadata.sample_id) - set(features.sample_id))}")
    features = features.sort_values("sample_id").reset_index(drop=True)
    metadata = metadata.set_index("sample_id").loc[features.sample_id].reset_index()
    if "outcome" not in metadata or metadata.outcome.isna().any():
        raise ValueError("Explicit nonmissing outcomes are required")
    if not metadata.outcome.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Outcomes must be nonempty strings")
    labels = sorted(metadata.outcome.unique())
    if len(labels) != 2 or positive_label not in labels:
        raise ValueError("Exactly two outcomes and an explicit positive_label are required")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if isinstance(group_columns, str):
        raise ValueError("group_columns must be a sequence of column names")
    group_columns = tuple(group_columns)
    names = [c for c in features if c != "sample_id"]
    if not names or not all(isinstance(c, str) and bool(c.strip()) for c in names):
        raise ValueError("At least one named numeric feature is required")
    if set(names) & ({"outcome"} | set(group_columns)):
        raise ValueError("Outcome and splitting columns cannot be features")
    try:
        X = features[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Features must be numeric or missing") from exc
    if np.isinf(X).any():
        raise ValueError("Infinite feature values are invalid")
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= len(names)):
        raise ValueError("top_k must be between 1 and the feature count")
    y = (metadata.outcome == positive_label).to_numpy(dtype=int)
    if np.bincount(y).min() < n_splits:
        raise ValueError("Each outcome needs at least n_splits samples")
    parent = list(range(len(y)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in group_columns:
        if col in ("sample_id", "outcome") or col not in metadata or metadata[col].isna().any():
            raise ValueError("Group columns must exist, be nonmissing, and differ from ID/outcome")
        seen = {}
        for i, value in enumerate(metadata[col]):
            if not str(value).strip():
                raise ValueError("Group values must be nonempty")
            if value in seen:
                parent[root(i)] = root(seen[value])
            seen[value] = i
    groups = np.array([root(i) for i in range(len(y))])
    if group_columns and len(set(groups)) < n_splits:
        raise ValueError("Too few independent group components for n_splits")
    splitter = (StratifiedGroupKFold if group_columns else StratifiedKFold)(
        n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups if group_columns else None))
    for train, test in splits:
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("Every training and test fold must contain both outcomes; change splits or grouping")
    probabilities = np.zeros(len(y))
    fold_ids = np.zeros(len(y), dtype=int)
    folds, rankings = [], []
    for fold, (train, test) in enumerate(splits):
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=top_k or "all")),
            ("classifier", LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=seed)),
        ])
        model.fit(X[train], y[train])
        if np.any(model.named_steps["classifier"].n_iter_ >= 2000):
            raise ValueError("Logistic regression did not converge")
        probabilities[test] = model.predict_proba(X[test])[:, 1]
        fold_ids[test] = fold
        folds.append({"fold": fold, "train_ids": features.sample_id.iloc[train].tolist(),
                      "test_ids": features.sample_id.iloc[test].tolist()})
        selected = np.array(names)[model.named_steps["select"].get_support()]
        for name, coefficient in zip(selected, model.named_steps["classifier"].coef_[0]):
            rankings.append({"fold": fold, "feature": name, "coefficient": float(coefficient),
                             "interpretation": "exploratory training-fold coefficient; not biological evidence"})
    predicted = (probabilities >= 0.5).astype(int)
    metrics = {"roc_auc": float(roc_auc_score(y, probabilities)),
               "average_precision": float(average_precision_score(y, probabilities)),
               "accuracy": float(accuracy_score(y, predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(y, predicted))}
    provenance = {"schema_version": 1, "omicsfold": "0.1.0", "python": platform.python_version(),
                  "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                  "seed": seed, "n_splits": n_splits, "top_k": top_k, "positive_label": positive_label,
                  "negative_label": next(x for x in labels if x != positive_label),
                  "group_columns": list(group_columns), "features_sha256": _fingerprint(features),
                  "metadata_sha256": _fingerprint(metadata), "threshold": 0.5,
                  "pipeline": "training-fold median imputation, standard scaling, ANOVA selection, logistic regression C=1 liblinear",
                  "evaluation": "pooled out-of-fold metrics; no tuning or independent external validation"}
    predictions = pd.DataFrame({"sample_id": features.sample_id, "outcome": metadata.outcome,
                                "fold": fold_ids, "positive_probability": probabilities,
                                "predicted_outcome": np.where(predicted, positive_label, provenance["negative_label"])})
    return Result(predictions, folds, metrics, provenance, pd.DataFrame(rankings))
