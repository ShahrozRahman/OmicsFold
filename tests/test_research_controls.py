"""Synthetic tests for normalization, group inference and verified replay."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from omicsfold import benchmark, load_data
from omicsfold.evaluation import group_weights, cluster_intervals
from omicsfold.preprocessing import AssayPreprocessor
from omicsfold.reproducibility import replay, verify_environment
from omicsfold.bundle import bundle_run
from test_benchmark import synthetic


def test_frozen_size_factors_use_training_reference_only():
    train = np.array([[10., 20., 4.], [20., 35., 10.], [30., 55., 7.]])
    preprocessing = AssayPreprocessor("counts", "median_ratio_log1p").fit(train)
    expected = np.exp(np.log(train).mean(axis=0))
    np.testing.assert_allclose(preprocessing.reference, expected)
    frozen = preprocessing.reference.copy()
    test = np.array([[100., 200., 40.]])
    factors = np.median(test / expected, axis=1) / preprocessing.factor_center
    manual = preprocessing.scaler.transform(np.log1p(test / factors[:, None]))
    np.testing.assert_allclose(preprocessing.transform(test), manual)
    preprocessing.transform(test * 100)
    np.testing.assert_equal(preprocessing.reference, frozen)
    with pytest.raises(ValueError, match="reference genes"):
        AssayPreprocessor("counts", "median_ratio_log1p").fit(np.eye(3))


@pytest.mark.parametrize("normalization", ["log1p", "clr"])
def test_numeric_assay_transforms(normalization):
    X = np.array([[0., 4., 3.], [2., 1., 8.], [3., 5., 2.]])
    preprocessing = AssayPreprocessor("numeric", normalization).fit(X)
    expected = np.log1p(X)
    if normalization == "clr":
        expected -= expected.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(expected.mean(axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(preprocessing.transform(X), preprocessing.scaler.transform(expected))
    with pytest.raises(ValueError, match="nonnegative"):
        AssayPreprocessor("numeric", normalization).fit(X - 10)


def test_batch_adjustment_is_frozen_and_rejects_unseen_or_confounded_batches():
    X = np.array([[1., 2.], [3., 4.], [11., 12.], [13., 14.]])
    batches = np.array(["A", "A", "B", "B"])
    y = np.array([0, 1, 0, 1])
    preprocessing = AssayPreprocessor("numeric").fit(X, batches, y)
    transformed = preprocessing.transform(X, batches)
    np.testing.assert_allclose(transformed[:2], transformed[2:], atol=1e-12)
    frozen = preprocessing.state()
    preprocessing.transform(X + 1000, batches)
    assert preprocessing.state() == frozen
    with pytest.raises(ValueError, match="Unseen test batches"):
        preprocessing.transform(X, np.array(["unseen"] * 4))
    with pytest.raises(ValueError, match="cannot separate batch"):
        AssayPreprocessor("numeric").fit(X, batches, np.array([0, 0, 1, 1]))


def test_partial_missing_counts_are_opt_in_and_medians_are_train_only(tmp_path):
    x, m = tmp_path / "x.tsv", tmp_path / "m.csv"
    x.write_text("sample_id\tA\tB\n001\t2\t\n002\t4\t8\n003\t6\t4\n004\t10\t2\n")
    m.write_text("sample_id,outcome\n001,case\n002,control\n003,case\n004,control\n")
    options = dict(orientation="samples-by-features", sample_id_column="sample_id", outcome_column="outcome",
                   positive_label="case", input_type="counts")
    with pytest.raises(ValueError, match="counts must be finite"):
        load_data(x, m, **options)
    data = load_data(x, m, **options, allow_missing_counts=True)
    assert np.isnan(data.features.B.iloc[0])
    p = AssayPreprocessor("counts", missing_counts="median").fit(np.array([[2., np.nan], [4., 8.], [6., 4.]]))
    assert p.imputer.statistics_[1] == 6
    p.transform(np.array([[100., np.nan]]))
    assert p.imputer.statistics_[1] == 6


def test_configured_metadata_harmonization_is_exact_and_audited(tmp_path):
    x, m = tmp_path / "x.csv", tmp_path / "m.csv"
    x.write_text("sample_id,A\n001,1\n002,2\n")
    m.write_text("Subject,Status\n002,healthy\n001,affected\n")
    mapping = {"columns": {"Subject": "sample_id", "Status": "outcome"},
               "values": {"outcome": {"healthy": "control", "affected": "case"}}}
    data = load_data(x, m, orientation="samples-by-features", sample_id_column="sample_id", outcome_column="outcome",
                     positive_label="case", input_type="numeric", metadata_mapping=mapping)
    assert data.metadata.outcome.tolist() == ["case", "control"]
    assert len(data.alignment_audit["metadata_harmonization"]) == 2
    x.write_text("sample_id,A,Status\n001,1,1\n002,2,0\n")
    with pytest.raises(ValueError, match="Metadata/reserved columns"):
        load_data(x, m, orientation="samples-by-features", sample_id_column="sample_id", outcome_column="outcome",
                  positive_label="case", input_type="numeric", metadata_mapping=mapping)
    x.write_text("sample_id,A\n001,1\n002,2\n")
    mapping["values"]["sample_id"] = {"001": "1", "002": "2"}
    with pytest.raises(ValueError, match="ID harmonization is prohibited"):
        load_data(x, m, orientation="samples-by-features", sample_id_column="sample_id", outcome_column="outcome",
                  positive_label="case", input_type="numeric", metadata_mapping=mapping)


def test_group_weights_and_cluster_bootstrap_treat_whole_groups_equally():
    groups = np.array(["large"] * 10 + ["small"] * 2)
    weights = group_weights(groups)
    assert weights[:10].sum() == pytest.approx(1)
    assert weights[10:].sum() == pytest.approx(1)
    y = np.array([0, 1] * 6)
    scores = np.linspace(.1, .9, 12)
    first = cluster_intervals(y, scores, groups, weights, 60, 42)
    assert first == cluster_intervals(y, scores, groups, weights, 60, 42)
    assert first["valid_resamples"] == 60
    assert "whole-test-group" in first["method"]


def test_group_balanced_scores_and_baseline_are_weighted():
    data = synthetic(grouped=True)
    data.metadata.loc[:5, "patient"] = "large"
    result = benchmark(data, holdout_cohort="external", bootstrap_repeats=40)
    assert result.provenance["configuration"]["weighting"] == "group_balanced"
    fold = result.folds[0]
    test = result.predictions[result.predictions.model == "logistic_l2"]
    y = (test.outcome == "case").to_numpy(dtype=int)
    expected = average_precision_score(y, test.positive_probability, sample_weight=test.evaluation_weight)
    assert result.metrics.query("model == 'logistic_l2'").average_precision.iloc[0] == expected
    assert fold["intervals"]["logistic_l2"]["valid_resamples"] == 40


def test_strict_replay_verifies_inputs_environment_and_exact_results(tmp_path):
    data = synthetic()
    data.features.to_csv(tmp_path / "x.csv", index=False)
    data.metadata.to_csv(tmp_path / "m.csv", index=False)
    loaded = load_data(tmp_path / "x.csv", tmp_path / "m.csv", orientation="samples-by-features",
                       sample_id_column="sample_id", outcome_column="outcome", positive_label="case", input_type="numeric")
    original = benchmark(loaded, n_splits=3)
    original.write(tmp_path / "original")
    reproduced = replay(tmp_path / "original/run.json", tmp_path / "x.csv", tmp_path / "m.csv", tmp_path / "replay")
    assert original.provenance["result_checksums"] == reproduced.provenance["result_checksums"]
    assert reproduced.provenance["reproducibility"]["environment_lock_verified"]
    changed = dict(original.provenance["reproducibility"]["environment"], machine="other-hardware")
    with pytest.raises(ValueError, match="Environment lock mismatch"):
        verify_environment(changed, original.provenance["reproducibility"]["environment"])
    record = json.loads((tmp_path / "original/run.json").read_text())
    record["result_checksums"]["predictions"] = "wrong"
    (tmp_path / "wrong.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="result checksum mismatch"):
        replay(tmp_path / "wrong.json", tmp_path / "x.csv", tmp_path / "m.csv", tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_bundle_preserves_inputs_code_dependency_pins_and_results(tmp_path):
    data = synthetic()
    data.features.to_csv(tmp_path / "x.csv", index=False)
    data.metadata.to_csv(tmp_path / "m.csv", index=False)
    loaded = load_data(tmp_path / "x.csv", tmp_path / "m.csv", orientation="samples-by-features",
                       sample_id_column="sample_id", outcome_column="outcome", positive_label="case", input_type="numeric")
    result = benchmark(loaded, n_splits=3)
    result.write(tmp_path / "result")
    bundle_run(tmp_path / "result/run.json", tmp_path / "x.csv", tmp_path / "m.csv", tmp_path / "archive.zip")
    with zipfile.ZipFile(tmp_path / "archive.zip") as zipped:
        assert zipped.read("inputs/features.csv") == (tmp_path / "x.csv").read_bytes()
        assert "engine/src/omicsfold/preprocessing.py" in zipped.namelist()
        assert "original/predictions.tsv" in zipped.namelist()
        assert b"scikit-learn==" in zipped.read("requirements.lock.txt")
        manifest = json.loads(zipped.read("bundle.json"))
        for name, expected in manifest["sha256"].items():
            assert hashlib.sha256(zipped.read(name)).hexdigest() == expected
    with pytest.raises(FileExistsError):
        bundle_run(tmp_path / "result/run.json", tmp_path / "x.csv", tmp_path / "m.csv", tmp_path / "archive.zip")
    with (tmp_path / "result/predictions.tsv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Saved artifact checksum mismatch"):
        bundle_run(tmp_path / "result/run.json", tmp_path / "x.csv", tmp_path / "m.csv", tmp_path / "corrupt.zip")
    assert not (tmp_path / "corrupt.zip").exists()
