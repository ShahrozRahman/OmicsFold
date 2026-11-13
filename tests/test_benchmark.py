"""Synthetic benchmark fixtures; no biological data or claims."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from omicsfold import benchmark, load_data
from omicsfold.benchmark import _FoldPreprocessor, _normalize_counts, _validate_split
from omicsfold.io import LoadedData


def synthetic(input_type="numeric", grouped=False):
    rng = np.random.default_rng(13)
    n = 24
    X = rng.poisson(8, (n, 4)) if input_type == "counts" else rng.normal(size=(n, 4))
    features = pd.DataFrame(X, columns=[f"synthetic_{i}" for i in range(4)])
    features.insert(0, "sample_id", [f"{i:03}" for i in range(n)])
    metadata = pd.DataFrame({"sample_id": features.sample_id,
                             "outcome": np.tile(["control", "case"], n // 2),
                             "patient": [f"P{i // 2:02}" for i in range(n)],
                             "cohort": ["external"] * 8 + ["development"] * 16})
    return LoadedData(features, metadata, "case", input_type,
                      "patient" if grouped else None, "cohort")


class BenchmarkTests(unittest.TestCase):
    def test_independent_scores_metrics_and_baseline(self):
        data = synthetic()
        result = benchmark(data, n_splits=3)
        self.assertEqual(result.provenance["split"], "stratified")
        self.assertEqual(len(result.metrics), 6)
        self.assertEqual(len(result.predictions), 48)
        for fold in result.folds:
            train, test = set(fold["train_ids"]), set(fold["test_ids"])
            self.assertFalse(train & test)
            for model in ["prevalence", "logistic_l2"]:
                predictions = result.predictions[
                    (result.predictions.fold == fold["fold"]) & (result.predictions.model == model)]
                metric = result.metrics[(result.metrics.fold == fold["fold"]) & (result.metrics.model == model)].iloc[0]
                y = (predictions.outcome == "case").astype(int)
                self.assertEqual(set(predictions.sample_id), test)
                self.assertEqual(metric.test_sample_count, len(test))
                self.assertEqual(metric.test_positive_prevalence, y.mean())
                self.assertAlmostEqual(metric.average_precision, average_precision_score(y, predictions.positive_probability))
                self.assertAlmostEqual(metric.roc_auc, roc_auc_score(y, predictions.positive_probability))
                if model == "prevalence":
                    self.assertTrue((predictions.positive_probability == fold["train_positive_prevalence"]).all())
                    self.assertAlmostEqual(metric.average_precision, y.mean())
                    self.assertEqual(metric.roc_auc, 0.5)

    def test_grouped_and_repeatable_with_shuffled_metadata(self):
        data = synthetic(grouped=True)
        first = benchmark(data, n_splits=3)
        data.metadata = data.metadata.sample(frac=1, random_state=18)
        data.features = data.features.sample(frac=1, random_state=20)
        second = benchmark(data, n_splits=3)
        pd.testing.assert_frame_equal(first.predictions, second.predictions)
        pd.testing.assert_frame_equal(first.metrics, second.metrics)
        for fold in first.folds:
            train = data.metadata[data.metadata.sample_id.isin(fold["train_ids"])]
            test = data.metadata[data.metadata.sample_id.isin(fold["test_ids"])]
            self.assertFalse(set(train.patient) & set(test.patient))

    def test_cohort_holdout_and_shared_patient_rejection(self):
        data = synthetic(grouped=True)
        result = benchmark(data, holdout_cohort="external")
        self.assertEqual(result.provenance["split"], "cohort")
        self.assertEqual(len(result.folds), 1)
        self.assertEqual(set(result.predictions.sample_id), set(data.metadata.sample_id[:8]))
        data.metadata.loc[8, "patient"] = data.metadata.loc[0, "patient"]
        with self.assertRaisesRegex(ValueError, "groups overlap.*P00"):
            benchmark(data, holdout_cohort="external")

    def test_single_class_partitions_are_rejected(self):
        for partition in ["training", "test"]:
            data = synthetic()
            if partition == "test": data.metadata.loc[:7, "outcome"] = "control"
            else: data.metadata.loc[8:, "outcome"] = "case"
            with self.subTest(partition=partition), self.assertRaisesRegex(ValueError, partition + " partition.*both outcome"):
                benchmark(data, holdout_cohort="external")
        data = synthetic(grouped=True)
        data.metadata["patient"] = data.metadata.outcome
        with self.assertRaisesRegex(ValueError, "partition.*both outcome"):
            benchmark(data, n_splits=2)

    def test_overlap_rejected_before_fitting(self):
        data = synthetic()
        with patch("omicsfold.benchmark.StratifiedKFold.split", return_value=[
                (np.array([0, 1, 2]), np.array([2, 3]))]), patch("omicsfold.benchmark.LogisticRegression.fit") as fit:
            with self.assertRaisesRegex(ValueError, "sample IDs overlap.*002"):
                benchmark(data, n_splits=2)
            fit.assert_not_called()
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            _validate_split(np.array([0, 1]), np.array([2, 3]), np.array(["a", "b", "c", "d"]),
                            np.array([0, 1, 0, 1]), np.array(["P1", "P2", "P1", "P3"]))

    def test_preprocessing_uses_only_training_samples(self):
        for input_type in ("numeric", "expression", "counts"):
            data = synthetic(input_type)
            seen = []
            original = _FoldPreprocessor.fit
            def spy(obj, X):
                seen.append(X.copy())
                return original(obj, X)
            with patch.object(_FoldPreprocessor, "fit", spy):
                result = benchmark(data, n_splits=3)
            self.assertEqual(len(seen), 3)
            for values, fold in zip(seen, result.folds):
                np.testing.assert_equal(values, data.features.set_index("sample_id").loc[fold["train_ids"]].to_numpy())

    def test_numeric_imputation_and_scaling_train_only(self):
        train = np.array([[1., np.nan, 5.], [3., np.nan, 5.], [np.nan, np.nan, 5.]])
        preprocessing = _FoldPreprocessor("numeric").fit(train)
        self.assertEqual(preprocessing.selected_indices.tolist(), [0])
        self.assertEqual(preprocessing.imputer.statistics_[0], 2.)
        self.assertEqual(preprocessing.scaler.mean_[0], 2.)
        np.testing.assert_equal(preprocessing.transform(np.array([[np.nan, 100., 100.]])), [[0.]])

    def test_counts_normalization_filtering_and_no_features(self):
        counts = np.array([[10., 20., 0.], [20., 10., 0.], [30., 30., 0.]])
        np.testing.assert_allclose(_normalize_counts(counts).sum(axis=1), 1e6)
        np.testing.assert_allclose(_normalize_counts(counts), _normalize_counts(counts * 5))
        preprocessing = _FoldPreprocessor("counts").fit(counts)
        self.assertEqual(preprocessing.selected_indices.tolist(), [0, 1])
        expected = np.log1p(_normalize_counts(counts)[:, :2]).mean(axis=0)
        np.testing.assert_allclose(preprocessing.scaler.mean_, expected)
        data = synthetic("counts")
        data.features["heldout_only"] = [10] * 8 + [0] * 16
        result = benchmark(data, holdout_cohort="external")
        self.assertNotIn("heldout_only", result.folds[0]["selected_features"])
        with self.assertRaisesRegex(ValueError, "no usable features.*expression filtering"):
            _FoldPreprocessor("counts").fit(np.eye(20))
        for matrix in (np.ones((4, 3)), np.full((4, 3), np.nan)):
            with self.assertRaisesRegex(ValueError, "no usable features"):
                _FoldPreprocessor("numeric").fit(matrix)
        data.features.loc[:, data.features.columns != "sample_id"] = 1
        with self.assertRaisesRegex(ValueError, "Fold 0.*no usable features"):
            benchmark(data, holdout_cohort="external")

    def test_invalid_requests_and_user_positive_label(self):
        data = synthetic()
        for kwargs in ({"split": "grouped"}, {"holdout_cohort": "absent"}, {"n_splits": 1},
                       {"split": "cohort"}, {"seed": -1}, {"split": "stratified", "holdout_cohort": "external"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): benchmark(data, **kwargs)
        data.metadata.loc[:7, "outcome"] = ["case"] * 6 + ["control"] * 2
        result = benchmark(data, holdout_cohort="external")
        self.assertTrue((result.metrics.test_positive_prevalence == 0.75).all())
        self.assertTrue((result.predictions.query("model == 'prevalence'").positive_probability == 0.5).all())
        data.positive_label = "control"
        reverse = benchmark(data, holdout_cohort="external")
        self.assertTrue((reverse.metrics.test_positive_prevalence == 0.25).all())

    def test_cli_api_parity_and_artifacts(self):
        data = synthetic()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data.features.to_csv(path / "features.csv", index=False)
            data.metadata.to_csv(path / "metadata.csv", index=False)
            cmd = [sys.executable, "-m", "omicsfold", "benchmark", "--features", str(path / "features.csv"),
                   "--metadata", str(path / "metadata.csv"), "--orientation", "samples-by-features",
                   "--sample-id-column", "sample_id", "--outcome-column", "outcome", "--positive-label", "case",
                   "--input-type", "numeric", "--folds", "3", "--output", str(path / "results")]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            loaded = load_data(path / "features.csv", path / "metadata.csv", orientation="samples-by-features",
                               sample_id_column="sample_id", outcome_column="outcome", positive_label="case", input_type="numeric")
            expected = benchmark(loaded, n_splits=3)
            from omicsfold.report import metric_rows
            pd.testing.assert_frame_equal(pd.read_csv(path / "results/metrics.tsv", sep="\t", keep_default_na=False,
                                                      dtype={"fold": str}),
                                          metric_rows(expected).assign(fold=lambda f: f.fold.astype(str)), check_dtype=False)
            pd.testing.assert_frame_equal(pd.read_csv(path / "results/predictions.tsv", sep="\t", dtype={"sample_id": str}), expected.predictions)
            self.assertEqual(json.loads((path / "results/run.json").read_text())["folds"], expected.folds)
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
