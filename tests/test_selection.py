"""Synthetic nested-selection and chronological-validation checks."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from omicsfold import benchmark, load_data, replay
from omicsfold.io import LoadedData
from omicsfold.preprocessing import AssayPreprocessor
from omicsfold.selection import candidates, parse_times, temporal_splits


def fixture():
    rng = np.random.default_rng(19)
    x = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"synthetic_{i}" for i in range(4)])
    x.insert(0, "sample_id", [f"S{i:03}" for i in range(60)])
    x["synthetic_0"] = np.arange(60, dtype=float)
    m = pd.DataFrame({"sample_id": x.sample_id, "outcome": ["case", "control"] * 30,
                      "patient": [f"P{i//2:03}" for i in range(60)],
                      "cohort": ["development"] * 48 + ["external"] * 12,
                      "date": [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i//2)).date().isoformat() for i in range(60)]})
    return LoadedData(x, m, "case", "numeric", "patient", "cohort")


GRID = [{"model": "logistic_l2", "C": .1}, {"model": "gaussian_nb", "var_smoothing": 1e-9},
        {"model": "random_forest", "n_estimators": 12, "max_depth": 2, "min_samples_leaf": 2}]


def test_inner_selection_never_sees_outer_test_and_refits_preprocessing():
    data = fixture()
    seen = []
    original = AssayPreprocessor.fit
    def spy(self, X, *args, **kwargs):
        seen.append(set(X[:, 0].astype(int)))
        return original(self, X, *args, **kwargs)
    with patch.object(AssayPreprocessor, "fit", spy):
        result = benchmark(data, holdout_cohort="external", tune=True,
                           models=["logistic_l2", "gaussian_nb", "random_forest"], search_space=GRID, bootstrap_repeats=0)
    assert len(seen) == 4
    assert all(indices <= set(range(48)) for indices in seen)
    assert len(seen[-1]) == 48 and all(len(indices) < 48 for indices in seen[:-1])
    outer = result.folds[0]
    patient_by_id = data.metadata.set_index("sample_id").patient.to_dict()
    for inner in outer["selection"]["inner_splits"]:
        assert not set(inner["train_ids"]) & set(inner["validation_ids"])
        assert not set(inner["train_ids"] + inner["validation_ids"]) & set(outer["test_ids"])
        assert not {patient_by_id[i] for i in inner["train_ids"]} & {patient_by_id[i] for i in inner["validation_ids"]}
    data.metadata.loc[48:, "outcome"] = ["control", "case"] * 6
    data.features.loc[48:, "synthetic_1"] += 100
    changed = benchmark(data, holdout_cohort="external", tune=True,
                        models=["logistic_l2", "gaussian_nb", "random_forest"], search_space=GRID, bootstrap_repeats=0)
    assert outer["selection"] == changed.folds[0]["selection"]
    assert set(result.predictions.model) == {"prevalence", "selected_model"}


@pytest.mark.parametrize("model", ["random_forest", "gaussian_nb"])
def test_alternative_models_write_honest_feature_outputs(tmp_path, model):
    result = benchmark(fixture(), holdout_cohort="external", models=[model], bootstrap_repeats=0)
    assert result.features.retained.all()
    assert result.features.coefficient.isna().all()
    result.write(tmp_path / model)
    report = (tmp_path / model / "report.html").read_text(encoding="utf-8")
    assert model in report
    if model == "random_forest":
        assert result.features.impurity_importance.sum() == pytest.approx(1)
        assert "not logistic coefficients" in report
    assert result.predictions.positive_probability.between(0, 1).all()


def test_temporal_holdout_embargo_and_inner_forward_selection():
    data = fixture()
    result = benchmark(data, split="temporal", time_column="date", time_cutoff="2020-01-25",
                       temporal_gap_days=2, tune=True, models=["logistic_l2"],
                       search_space=[{"model": "logistic_l2", "C": .1}, {"model": "logistic_l2", "C": 1}],
                       bootstrap_repeats=0)
    fold = result.folds[0]
    dates = data.metadata.set_index("sample_id").date.to_dict()
    assert max(dates[i] for i in fold["train_ids"]) < "2020-01-23"
    assert min(dates[i] for i in fold["test_ids"]) == "2020-01-25"
    assert len(fold["temporal"]["excluded_ids"]) == 4
    for inner in fold["selection"]["inner_splits"]:
        assert max(dates[i] for i in inner["train_ids"]) < min(dates[i] for i in inner["validation_ids"])
        assert set(inner["train_ids"] + inner["validation_ids"]) <= set(fold["train_ids"])


def test_temporal_forward_windows_ties_and_coverage():
    data = fixture()
    result = benchmark(data, split="temporal", time_column="date", n_splits=3, bootstrap_repeats=0)
    assert len(result.folds) == 3
    assert len(result.provenance["validation_design"]["unscored_samples"]) == 16
    for fold in result.folds:
        assert fold["temporal"]["latest_train"] < fold["temporal"]["earliest_test"]
    predictions = result.predictions[result.predictions.model == "logistic_l2"]
    assert not predictions.sample_id.duplicated().any()
    data.metadata.loc[48, "patient"] = "P000"
    with pytest.raises(ValueError, match="groups overlap"):
        benchmark(data, split="temporal", time_column="date", time_cutoff="2020-01-25", bootstrap_repeats=0)


def test_invalid_selection_and_time_requests_fail():
    for value in ([1, 2], ["02/03/2020"], ["missing"]):
        with pytest.raises(ValueError): parse_times(pd.Series(value))
    for grid in ([{"model": "logistic_l2", "C": -1}], [{"model": "random_forest", "n_estimators": True}],
                 [{"model": "gaussian_nb", "var_smoothing": 0}]):
        with pytest.raises(ValueError): candidates([grid[0]["model"]], True, grid)
    with pytest.raises(ValueError, match="require tune"):
        benchmark(fixture(), models=["logistic_l2", "gaussian_nb"])
    with pytest.raises(ValueError, match="time_column"):
        benchmark(fixture(), split="temporal")
    with pytest.raises(ValueError, match="inner_splits samples"):
        benchmark(fixture(), holdout_cohort="external", tune=True, inner_splits=30)
    with pytest.raises(ValueError, match="nonempty"):
        benchmark(fixture(), split="temporal", time_column="date", time_cutoff="2020-01-01")
    data = fixture()
    data.metadata.loc[data.metadata.date < "2020-01-10", "outcome"] = "case"
    with pytest.raises(ValueError, match="training partition must contain both"):
        benchmark(data, split="temporal", time_column="date", time_cutoff="2020-01-10")


def test_temporal_nested_replay(tmp_path):
    data = fixture()
    data.features.to_csv(tmp_path / "x.csv", index=False)
    data.metadata.to_csv(tmp_path / "m.csv", index=False)
    loaded = load_data(tmp_path / "x.csv", tmp_path / "m.csv", orientation="samples-by-features",
                       sample_id_column="sample_id", outcome_column="outcome", positive_label="case",
                       input_type="numeric", group_column="patient", cohort_column="cohort")
    result = benchmark(loaded, split="temporal", time_column="date", time_cutoff="2020-01-25",
                        tune=True, models=["logistic_l2", "gaussian_nb"], inner_splits=3, bootstrap_repeats=0)
    result.write(tmp_path / "original")
    reproduced = replay(tmp_path / "original/run.json", tmp_path / "x.csv", tmp_path / "m.csv", tmp_path / "replay")
    assert reproduced.provenance["result_checksums"] == result.provenance["result_checksums"]
