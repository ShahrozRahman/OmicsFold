"""Predeclared model search and chronological splitting inside training partitions."""
import numpy as np
import pandas as pd
from datetime import datetime, date
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from .evaluation import group_weights
from .preprocessing import AssayPreprocessor


def parse_times(values):
    if any(not isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)) and
           (not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$", value)) for value in values):
        raise ValueError("Time values must be explicit ISO dates/timestamps, not guessed numeric epochs or ambiguous dates")
    try:
        times = pd.to_datetime(values, errors="raise", utc=True, format="mixed")
    except (ValueError, TypeError) as exc:
        raise ValueError("Time values must be explicit parseable dates/timestamps") from exc
    if times.isna().any():
        raise ValueError("Time values must not be missing")
    return times.to_numpy()


def temporal_splits(times, n_splits, cutoff=None, gap_days=0):
    if not isinstance(gap_days, (int, float)) or isinstance(gap_days, bool) or not np.isfinite(gap_days) or gap_days < 0:
        raise ValueError("temporal_gap_days must be finite and nonnegative")
    try:
        gap = pd.Timedelta(days=gap_days)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Temporal gap is outside the supported timestamp range") from exc
    if cutoff is not None:
        boundary = parse_times(pd.Series([cutoff]))[0]
        return [(np.flatnonzero(times < boundary - gap), np.flatnonzero(times >= boundary))]
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("Temporal CV requires n_splits >=2")
    unique = np.unique(times)
    if len(unique) < n_splits + 1:
        raise ValueError("Too few distinct timestamps for temporal CV")
    blocks = np.array_split(unique, n_splits + 1)
    return [(np.flatnonzero(times < block[0] - gap), np.flatnonzero(np.isin(times, block))) for block in blocks[1:]]


def candidates(models, tune, search_space):
    if isinstance(models, str):
        models = [models]
    if not models or any(m not in ("logistic_l2", "random_forest", "gaussian_nb") for m in models):
        raise ValueError("models must contain logistic_l2, random_forest or gaussian_nb")
    if len(set(models)) != len(models):
        raise ValueError("Duplicate model names")
    if not isinstance(tune, bool):
        raise ValueError("tune must be a boolean")
    if not tune and (len(models) > 1 or search_space is not None):
        raise ValueError("Multiple candidates or search_space require tune=True and inner validation")
    if search_space is None:
        grid = []
        for model in models:
            if model == "logistic_l2":
                grid.extend({"model": model, "C": c} for c in ([.01, .1, 1., 10.] if tune else [1.]))
            elif model == "random_forest":
                grid.extend({"model": model, "n_estimators": 200, "max_depth": depth, "min_samples_leaf": 2}
                            for depth in ([3, None] if tune else [None]))
            else:
                grid.extend({"model": model, "var_smoothing": v} for v in ([1e-9, 1e-7] if tune else [1e-9]))
    else:
        if not isinstance(search_space, list) or not search_space:
            raise ValueError("search_space must be a nonempty list of candidate dictionaries")
        if any(not isinstance(candidate, dict) for candidate in search_space):
            raise ValueError("Every search candidate must be a parameter dictionary")
        grid = [dict(candidate) for candidate in search_space]
        if any(candidate.get("model") not in models for candidate in grid):
            raise ValueError("Search candidates must belong to the declared models")
    for candidate in grid:
        make_model(candidate, 42)
    return grid


def make_model(candidate, seed):
    name = candidate.get("model")
    allowed = {"logistic_l2": {"model", "C"}, "random_forest": {"model", "n_estimators", "max_depth", "min_samples_leaf"},
               "gaussian_nb": {"model", "var_smoothing"}}
    if name not in allowed or set(candidate) - allowed[name]:
        raise ValueError("Unsupported model parameter in search_space")
    if name == "logistic_l2":
        c = candidate.get("C", 1.)
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not np.isfinite(c) or c <= 0:
            raise ValueError("Logistic C must be finite and positive")
        return LogisticRegression(C=c, solver="liblinear", tol=1e-4, max_iter=2000, random_state=seed)
    if name == "gaussian_nb":
        smoothing = candidate.get("var_smoothing", 1e-9)
        if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("var_smoothing must be finite and positive")
        return GaussianNB(var_smoothing=smoothing)
    for key, default in (("n_estimators", 200), ("min_samples_leaf", 2), ("max_depth", None)):
        value = candidate.get(key, default)
        if key == "max_depth" and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Random forest {key} must be a positive integer")
    return RandomForestClassifier(n_estimators=candidate.get("n_estimators", 200), max_depth=candidate.get("max_depth"),
                                  min_samples_leaf=candidate.get("min_samples_leaf", 2), random_state=seed, n_jobs=1)


def fit_model(candidate, X, y, weights, seed):
    model = make_model(candidate, seed)
    model.fit(X, y, sample_weight=weights)
    if hasattr(model, "n_iter_") and np.any(model.n_iter_ >= 2000):
        raise ValueError("Logistic regression did not converge")
    return model


def select_candidate(grid, X, y, ids, groups, batches, times, n_splits, seed,
                     input_type, normalization, missing_counts, weighting, gap_days, validate_split):
    """This function receives outer TRAINING data only; outer tests are inaccessible."""
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("inner_splits must be an integer >=2")
    if times is not None:
        splits = temporal_splits(times, n_splits, gap_days=gap_days)
    else:
        if np.bincount(y).min() < n_splits:
            raise ValueError("Each outer training outcome needs at least inner_splits samples")
        splitter = (StratifiedGroupKFold if groups is not None else StratifiedKFold)(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups) if groups is not None else splitter.split(X, y))
    for train, valid in splits:
        validate_split(train, valid, ids, y, groups)
    prepared, records = [], []
    for train, valid in splits:
        pre = AssayPreprocessor(input_type, normalization, missing_counts).fit(
            X[train], batches[train] if batches is not None else None, y[train])
        training = pre.transform(X[train], batches[train] if batches is not None else None)
        validation = pre.transform(X[valid], batches[valid] if batches is not None else None)
        weights = group_weights(groups[train]) if weighting == "group_balanced" else np.ones(len(train))
        valid_weights = group_weights(groups[valid]) if weighting == "group_balanced" else np.ones(len(valid))
        prepared.append((training, validation, y[train], y[valid], weights, valid_weights))
        records.append({"train_ids": ids[train].tolist(), "validation_ids": ids[valid].tolist(), "preprocessing": pre.state()})
    results = []
    for candidate in grid:
        scores = []
        for training, validation, train_y, valid_y, weights, valid_weights in prepared:
            model = fit_model(candidate, training, train_y, weights, seed)
            probability = model.predict_proba(validation)[:, 1]
            scores.append(float(average_precision_score(valid_y, probability, sample_weight=valid_weights)))
        results.append({"candidate": candidate, "inner_average_precision": scores, "mean_inner_average_precision": float(np.mean(scores))})
    best = max(range(len(results)), key=lambda i: results[i]["mean_inner_average_precision"])
    return grid[best], {"objective": "unweighted mean inner-fold weighted average precision", "tie_break": "first declared candidate",
                        "inner_splits": records, "candidates": results, "selected_candidate": grid[best]}
