"""Sample-level held-out benchmarks; shared by the Python API and CLI."""
import platform
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .engine import _fingerprint, _ids
from .io import LoadedData
from .report import REPORT_SCHEMA_VERSION
from .preprocessing import AssayPreprocessor
from .evaluation import group_weights, cluster_intervals
from .reproducibility import environment, verify_environment, single_thread, source_revision
from .selection import candidates, fit_model, parse_times, temporal_splits, select_candidate


@dataclass
class BenchmarkResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    folds: list
    provenance: dict
    features: pd.DataFrame
    splits: pd.DataFrame

    def write(self, directory):
        """Write the stable report schema to a new directory only."""
        from .report import write_results
        write_results(self, directory)


def _normalize_counts(X):
    # Scaling by row maxima first avoids overflow in the library-size sum.
    relative = X / X.max(axis=1, keepdims=True)
    return relative / relative.sum(axis=1, keepdims=True) * 1_000_000.0


class _FoldPreprocessor(AssayPreprocessor):
    """Fit feature masks, imputation and scaling on training samples only."""
    def __init__(self, input_type):
        super().__init__(input_type)


def _validate_split(train, test, ids, y, groups=None):
    """Check every proposed split, including cohort holdouts, before fitting."""
    if not len(train) or not len(test):
        raise ValueError("Training and test partitions must both be nonempty")
    overlap = sorted(set(ids[train]) & set(ids[test]))
    if overlap:
        raise ValueError(f"Train/test sample IDs overlap: {overlap}")
    if groups is not None:
        shared = sorted(set(groups[train]) & set(groups[test]), key=str)
        if shared:
            raise ValueError(f"Train/test groups overlap: {shared}; shared patients/groups cannot cross a split")
    for name, indices in (("training", train), ("test", test)):
        if len(np.unique(y[indices])) != 2:
            raise ValueError(f"The {name} partition must contain both outcome classes")


@single_thread
def benchmark(data, *, split=None, n_splits=5, holdout_cohort=None, seed=42, dataset_label="Dataset",
              normalization="auto", missing_counts="error", batch_column=None,
              weighting="auto", bootstrap_repeats=None, environment_lock=None,
              models=("logistic_l2",), tune=False, inner_splits=3, search_space=None,
              time_column=None, time_cutoff=None, temporal_gap_days=0.0):
    """Benchmark a LoadedData object from load_data using held-out samples.

    split: stratified, grouped, or cohort. By default, choose cohort when a
    holdout is named, grouped when a group column is supplied, else stratified.
    Positive probabilities and metrics always refer to data.positive_label.
    """
    if not isinstance(data, LoadedData):
        raise ValueError("benchmark requires a LoadedData object from load_data")
    runtime = environment()
    verify_environment(environment_lock, runtime)
    if not isinstance(models, (str, list, tuple)):
        raise ValueError("models must be a model name or a list/tuple of names")
    models = [models] if isinstance(models, str) else list(models)
    grid = candidates(models, tune, search_space)
    if not isinstance(temporal_gap_days, (int, float)) or isinstance(temporal_gap_days, bool) or not np.isfinite(temporal_gap_days) or temporal_gap_days < 0:
        raise ValueError("temporal_gap_days must be finite and nonnegative")
    temporal_gap_days = float(temporal_gap_days)
    if time_cutoff is not None and not isinstance(time_cutoff, str):
        raise ValueError("time_cutoff must be an explicit ISO date/timestamp string")
    if weighting not in ("auto", "sample", "group_balanced"):
        raise ValueError("weighting must be auto, sample, or group_balanced")
    if not isinstance(dataset_label, str) or not dataset_label.strip():
        raise ValueError("dataset_label must be a nonempty string")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if data.input_type not in ("counts", "expression", "numeric"):
        raise ValueError("input_type must be counts, expression, or numeric")
    features, metadata = data.features.copy(), data.metadata.copy()
    for frame, name in ((features, "features"), (metadata, "metadata")):
        if frame.columns.duplicated().any():
            raise ValueError(f"Duplicate {name} column names")
        _ids(frame, name)
    if set(features.sample_id) != set(metadata.sample_id):
        raise ValueError(f"Unmatched IDs: matrix-only={sorted(set(features.sample_id) - set(metadata.sample_id))}; "
                         f"metadata-only={sorted(set(metadata.sample_id) - set(features.sample_id))}")
    original_ids = features.sample_id.tolist()
    original_metadata_ids = metadata.sample_id.tolist()
    features = features.sort_values("sample_id").reset_index(drop=True)
    metadata = metadata.set_index("sample_id").loc[features.sample_id].reset_index()
    if "outcome" not in metadata or metadata.outcome.isna().any() or not metadata.outcome.map(
            lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Explicit nonmissing string outcomes are required")
    if metadata.outcome.nunique() != 2 or data.positive_label not in set(metadata.outcome):
        raise ValueError("Exactly two outcomes and a matching user-specified positive label are required")
    names = [c for c in features if c != "sample_id"]
    if not names or any(not isinstance(c, str) or not c.strip() for c in names):
        raise ValueError("Preprocessing leaves no usable features: matrix has no named features")
    overlap = sorted(set(names) & set(metadata.columns))
    if overlap:
        raise ValueError(f"Metadata cannot be matrix features: {overlap}")
    try:
        X = features[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature values must be numeric or missing") from exc
    if data.input_type == "counts":
        observed = X[np.isfinite(X)]
        if np.isinf(X).any() or (missing_counts != "median" and np.isnan(X).any()) or (observed < 0).any() or (observed != np.floor(observed)).any():
            raise ValueError("Raw counts must be finite, nonnegative integers")
        zero = features.sample_id[(np.nan_to_num(X, nan=0) == 0).all(axis=1)].tolist()
        if zero:
            raise ValueError(f"Zero library size for samples: {zero}")
    elif np.isinf(X).any():
        raise ValueError("Numerical features must be finite or missing")
    y = (metadata.outcome == data.positive_label).to_numpy(dtype=int)
    ids = features.sample_id.to_numpy()
    for column in (data.group_column, data.cohort_column, batch_column, time_column):
        if column is not None and (column in ("sample_id", "outcome") or column not in metadata or
                                   metadata[column].isna().any() or
                                   metadata[column].map(lambda x: not str(x).strip()).any()):
            raise ValueError(f"Invalid or missing splitting column: {column}")
    groups = metadata[data.group_column].to_numpy() if data.group_column else None
    weighting = ("group_balanced" if groups is not None else "sample") if weighting == "auto" else weighting
    if weighting == "group_balanced" and groups is None:
        raise ValueError("Group-balanced weighting requires a group column")
    bootstrap_repeats = (200 if groups is not None else 0) if bootstrap_repeats is None else bootstrap_repeats
    if not isinstance(bootstrap_repeats, int) or isinstance(bootstrap_repeats, bool) or (bootstrap_repeats != 0 and bootstrap_repeats < 40):
        raise ValueError("bootstrap_repeats must be 0 or an integer >=40")
    batches = metadata[batch_column].astype(str).to_numpy() if batch_column else None
    split = split or ("cohort" if holdout_cohort is not None else "grouped" if groups is not None else "stratified")
    if split not in ("stratified", "grouped", "cohort", "temporal"):
        raise ValueError("split must be stratified, grouped, cohort or temporal")
    if split != "cohort" and holdout_cohort is not None:
        raise ValueError("holdout_cohort requires a cohort split")
    if split != "temporal" and (time_cutoff is not None or temporal_gap_days != 0):
        raise ValueError("Time cutoff and gap require a temporal split")
    times = parse_times(metadata[time_column]) if time_column else None
    if split == "temporal":
        if times is None:
            raise ValueError("Temporal validation requires time_column")
        splits = temporal_splits(times, n_splits, time_cutoff, temporal_gap_days)
    elif split == "cohort":
        if data.cohort_column is None or holdout_cohort is None:
            raise ValueError("Cohort holdout requires a cohort column and a named holdout_cohort")
        test_mask = (metadata[data.cohort_column] == holdout_cohort).to_numpy()
        if not test_mask.any():
            raise ValueError(f"Holdout cohort not found: {holdout_cohort!r}")
        splits = [(np.flatnonzero(~test_mask), np.flatnonzero(test_mask))]
    else:
        if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")
        if np.bincount(y).min() < n_splits:
            raise ValueError("Each outcome needs at least n_splits samples")
        if split == "grouped":
            if groups is None:
                raise ValueError("Grouped cross-validation requires a group column")
            if len(set(groups)) < n_splits:
                raise ValueError("Too few independent groups for n_splits")
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y, groups))
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y))
    for fold, (train, test) in enumerate(splits):
        try:
            _validate_split(train, test, ids, y, groups)
        except ValueError as exc:
            raise ValueError(f"Fold {fold}: {exc}") from exc
    predictions, metrics, folds, feature_rows, split_rows = [], [], [], [], []
    for fold, (train, test) in enumerate(splits):
        selected, selection = grid[0], None
        if tune:
            try:
                selected, selection = select_candidate(grid, X[train], y[train], ids[train],
                    groups[train] if groups is not None else None, batches[train] if batches is not None else None,
                    times[train] if split == "temporal" else None, inner_splits, seed, data.input_type,
                    normalization, missing_counts, weighting, temporal_gap_days, _validate_split)
            except ValueError as exc:
                raise ValueError(f"Outer fold {fold} inner selection: {exc}") from exc
        try:
            if normalization == "auto" and missing_counts == "error" and batches is None:
                preprocessing = _FoldPreprocessor(data.input_type).fit(X[train])
                train_x = preprocessing.transform(X[train])
                test_x = preprocessing.transform(X[test])
            else:
                preprocessing = AssayPreprocessor(data.input_type, normalization, missing_counts).fit(
                    X[train], batches[train] if batches is not None else None, y[train])
                train_x = preprocessing.transform(X[train], batches[train] if batches is not None else None)
                test_x = preprocessing.transform(X[test], batches[test] if batches is not None else None)
        except ValueError as exc:
            raise ValueError(f"Fold {fold}: {exc}") from exc
        # liblinear's default penalty is L2. Leaving the deprecated penalty
        # parameter at its default also works across sklearn >=1.6.
        train_weights = group_weights(groups[train]) if weighting == "group_balanced" else np.ones(len(train))
        test_weights = group_weights(groups[test]) if weighting == "group_balanced" else np.ones(len(test))
        model = fit_model(selected, train_x, y[train], train_weights, seed)
        model_label = "selected_model" if tune else selected["model"]
        coefficients = ({int(i): float(c) for i, c in zip(preprocessing.selected_indices, model.coef_[0])}
                        if hasattr(model, "coef_") else {})
        importances = ({int(i): float(c) for i, c in zip(preprocessing.selected_indices, model.feature_importances_)}
                       if hasattr(model, "feature_importances_") else {})
        ranking = {i: rank for rank, i in enumerate(sorted(coefficients, key=lambda i: (-abs(coefficients[i]), names[i])), 1)}
        for i, name in enumerate(names):
            retained = i in preprocessing.selected_indices
            reason = ("retained" if retained else "all_missing_in_training" if not preprocessing.observed[i] else "constant_in_training" if preprocessing.mask[i]
                      else "expression_filter" if data.input_type == "counts" else "all_missing_in_training")
            feature_rows.append({"fold": fold, "model": model_label, "feature": name,
                                 "retained": retained, "reason": reason, "coefficient": coefficients.get(i),
                                 "absolute_coefficient_rank": ranking.get(i), "interpretation": "exploratory model output"})
            if importances:
                feature_rows[-1]["impurity_importance"] = importances.get(i)
        for partition, indices in (("train", train), ("test", test)):
            for i in indices:
                split_rows.append({"fold": fold, "partition": partition, "sample_id": ids[i],
                                   "outcome": metadata.outcome.iloc[i],
                                   "group": str(metadata[data.group_column].iloc[i]) if data.group_column else "",
                                   "cohort": str(metadata[data.cohort_column].iloc[i]) if data.cohort_column else ""})
        training_prevalence = float(np.average(y[train], weights=train_weights))
        scores = {"prevalence": np.full(len(test), training_prevalence),
                  model_label: model.predict_proba(test_x)[:, 1]}
        intervals = {}
        for model_name, probability in scores.items():
            metrics.append({"fold": fold, "model": model_name,
                            "average_precision": float(average_precision_score(y[test], probability, sample_weight=test_weights)),
                            "roc_auc": float(roc_auc_score(y[test], probability, sample_weight=test_weights)),
                            "test_positive_prevalence": float(np.average(y[test], weights=test_weights)), "test_sample_count": len(test)})
            intervals[model_name] = cluster_intervals(y[test], probability, groups[test] if groups is not None else ids[test],
                                                       test_weights, bootstrap_repeats, seed + fold)
            for i, score, evaluation_weight in zip(test, probability, test_weights):
                predictions.append({"sample_id": ids[i], "outcome": metadata.outcome.iloc[i],
                                    "positive_label": data.positive_label, "fold": fold, "model": model_name,
                                    "positive_probability": float(score),
                                    "evaluation_weight": float(evaluation_weight)})
        folds.append({"fold": fold, "train_ids": ids[train].tolist(), "test_ids": ids[test].tolist(),
                      "train_positive_prevalence": training_prevalence,
                      "training_sample_prevalence": float(y[train].mean()),
                      "test_group_count": len(set(groups[test])) if groups is not None else len(test),
                      "weighting": weighting, "intervals": intervals,
                      "preprocessing": preprocessing.state(),
                      "intercept": float(model.intercept_[0]) if hasattr(model, "intercept_") else None,
                      "selected_candidate": selected, "selection": selection, "model_parameters": model.get_params(),
                      "temporal": {"latest_train": str(max(times[train])), "earliest_test": str(min(times[test])),
                                   "excluded_ids": ids[np.setdiff1d(np.arange(len(ids)), np.concatenate([train, test]))].tolist()}
                                  if split == "temporal" else None,
                      "selected_features": [names[i] for i in preprocessing.selected_indices]})
    provenance = {"omicsfold": "0.1.0", "schema_version": 1, "python": platform.python_version(),
                  "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                  "scikit_learn": sklearn.__version__, "seed": seed, "split": split,
                  "n_splits": len(splits), "holdout_cohort": holdout_cohort,
                  "positive_label": data.positive_label, "input_type": data.input_type,
                  "group_column": data.group_column, "cohort_column": data.cohort_column,
                  "features_sha256": _fingerprint(features), "metadata_sha256": _fingerprint(metadata),
                  "count_preprocessing": "sample-wise CPM (1e6), log1p; training CPM>=1 in >=ceil(10% of training samples); remove training constants; training standard scaling",
                  "numerical_preprocessing": "drop training-all-missing features; training median imputation; remove training constants; training standard scaling",
                  "logistic_regression": {"penalty": "l2", "C": 1.0, "solver": "liblinear", "tol": 1e-4,
                                          "max_iter": 2000, "class_weight": None, "fit_intercept": True},
                  "baseline": "constant weighted training-partition positive prevalence",
                  "evaluation": "sample-level held-out scores and per-fold metrics; user-specified positive label; no tuning"}
    current_positions = {value: i for i, value in enumerate(original_metadata_ids)}
    current_audit = {"matched_samples": len(ids), "matrix_only_ids": [], "metadata_only_ids": [],
                     "duplicate_matrix_ids": 0, "duplicate_metadata_ids": 0,
                     "metadata_rows_reordered": sum(current_positions[value] != i for i, value in enumerate(original_ids)),
                     "position_basis": "zero-based in-memory input positions",
                     "samples": [{"sample_id": value, "matrix_position": i,
                                  "metadata_position": current_positions[value]} for i, value in enumerate(original_ids)]}
    provenance.update(report_schema_version=REPORT_SCHEMA_VERSION, dataset_label=dataset_label,
                      input_checksums={"source_files": data.input_provenance,
                                       "canonical_features_sha256": provenance["features_sha256"],
                                       "canonical_metadata_sha256": provenance["metadata_sha256"]},
                      sample_alignment_audit={"source_alignment": data.alignment_audit or current_audit,
                                              "benchmark_alignment": current_audit,
                                              "analysis_order": "sample_id sorted ascending"},
                      class_balance={str(label): int(count) for label, count in metadata.outcome.value_counts().items()},
                      configuration={"loading": data.loading_configuration,
                                     "split": split, "n_splits_requested": n_splits, "n_splits_actual": len(splits),
                                     "holdout_cohort": holdout_cohort, "seed": seed, "dataset_label": dataset_label,
                                     "positive_label": data.positive_label, "input_type": data.input_type,
                                     "group_column": data.group_column, "cohort_column": data.cohort_column,
                                     "logistic_regression": provenance["logistic_regression"],
                                     "count_preprocessing": provenance["count_preprocessing"],
                                     "numerical_preprocessing": provenance["numerical_preprocessing"],
                                     "normalization": normalization, "missing_counts": missing_counts,
                                     "batch_column": batch_column, "weighting": weighting,
                                     "bootstrap_repeats": bootstrap_repeats, "models": models, "tune": tune,
                                     "inner_splits": inner_splits, "search_space": search_space,
                                     "time_column": time_column, "time_cutoff": time_cutoff, "temporal_gap_days": temporal_gap_days},
                      package_versions={"omicsfold": "0.1.0", "python": platform.python_version(),
                                        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                                        "scikit-learn": sklearn.__version__})
    result = BenchmarkResult(pd.DataFrame(predictions), pd.DataFrame(metrics), folds, provenance,
                             pd.DataFrame(feature_rows), pd.DataFrame(split_rows))
    provenance["reproducibility"] = {"environment": runtime, "environment_lock_verified": environment_lock is not None,
                                    "source": source_revision(),
                                    "guarantee": "Exact replay is accepted only after environment, input and result checksum verification"}
    provenance["result_checksums"] = {key: _fingerprint(getattr(result, key)) for key in ("predictions", "metrics", "splits", "features")}
    provenance["result_checksums"]["fold_states"] = hashlib.sha256(json.dumps(folds, sort_keys=True, allow_nan=False).encode()).hexdigest()
    provenance["evaluation"] = f"{weighting} held-out evaluation; cluster bootstrap resamples whole supplied groups; no independence claim"
    provenance["validation_design"] = {"outer_strategy": split, "nested_selection": tune, "declared_candidates": grid,
        "selection_data": "outer training partitions only", "selection_metric": "mean inner-fold average precision",
        "test_role": "performance estimation only; never used for tuning", "time_cutoff": time_cutoff,
        "scored_samples": sorted(result.predictions.sample_id.unique().tolist()),
        "unscored_samples": sorted(set(ids) - set(result.predictions.sample_id)),
        "external_validity": "A named-cohort or chronological holdout supplies empirical evidence for that population/time only; no universal proof"}
    return result
