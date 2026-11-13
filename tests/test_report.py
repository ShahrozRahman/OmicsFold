"""Synthetic report tests: membership, numerical summaries and HTML safety."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omicsfold import benchmark, load_data
from omicsfold.report import OUTPUT_FILES, REPORT_SCHEMA_VERSION, metric_rows, render_report
from test_benchmark import synthetic


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.external_resources = []
    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for key, value in attrs:
            if key in ("src", "href") or key.lower().startswith("on"):
                self.external_resources.append((tag, key, value))


class ReportTests(unittest.TestCase):
    def test_complete_schema_source_checksums_and_holdout_membership(self):
        root = Path(__file__).parents[1]
        data = load_data(root / "examples/benchmark/counts.tsv", root / "examples/benchmark/metadata.csv",
                         orientation="samples-by-features", sample_id_column="sample_id", outcome_column="outcome",
                         positive_label="case", input_type="counts", group_column="patient", cohort_column="cohort")
        result = benchmark(data, holdout_cohort="external", dataset_label="Synthetic counts")
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "new-run"
            result.write(out)
            self.assertEqual(set(p.name for p in out.iterdir()), set(OUTPUT_FILES))
            run = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["report_schema_version"], REPORT_SCHEMA_VERSION)
            self.assertEqual(run["configuration"]["seed"], 42)
            self.assertIn("scikit-learn", run["package_versions"])
            for role in ("features", "metadata"):
                source = run["input_checksums"]["source_files"][role]
                self.assertEqual(source["sha256"], hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest())
            self.assertEqual(run["sample_alignment_audit"]["source_alignment"]["metadata_rows_reordered"], 8)
            splits = pd.read_csv(out / "splits.tsv", sep="\t", dtype={"sample_id": str})
            self.assertEqual(len(splits), 8)
            self.assertEqual(set(splits.partition), {"train", "test"})
            self.assertTrue((splits.query("partition == 'test'").cohort == "external").all())
            self.assertTrue((splits.query("partition == 'train'").cohort == "development").all())
            self.assertFalse(set(splits.query("partition == 'train'").sample_id) & set(splits.query("partition == 'test'").sample_id))
            features = pd.read_csv(out / "features.tsv", sep="\t")
            self.assertEqual(features.retained.sum(), 3)
            self.assertTrue((features.interpretation == "exploratory model output").all())
            ranked = features.sort_values("absolute_coefficient_rank")
            self.assertTrue(np.all(np.diff(ranked.coefficient.abs()) <= 0))
            report = (out / "report.html").read_text(encoding="utf-8")
            for text in ("Sample alignment audit", "Class balance", "fold sizes", "Model comparison", "Named cohort holdout: external", "Exploratory", "Synthetic counts"):
                self.assertIn(text, report)
            original_files = {file.name: file.read_bytes() for file in out.iterdir()}
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                result.write(out)
            self.assertEqual(original_files, {file.name: file.read_bytes() for file in out.iterdir()})

    def test_fold_means_and_individual_fold_curve(self):
        result = benchmark(synthetic(), n_splits=3)
        rows = metric_rows(result)
        self.assertEqual(len(rows.query("scope == 'fold'")), 6)
        self.assertEqual(len(rows.query("scope == 'mean'")), 2)
        for model in ("prevalence", "logistic_l2"):
            actual = rows[(rows.scope == "mean") & (rows.model == model)].iloc[0]
            expected = result.metrics[result.metrics.model == model].average_precision.mean()
            self.assertAlmostEqual(actual.average_precision, expected)
        report = render_report(result)
        self.assertIn("Individual fold 0 only; not pooled across folds", report)
        self.assertIn("unweighted arithmetic means", report)
        self.assertIn("<svg", report)

    def test_user_text_is_escaped_and_report_is_self_contained(self):
        attack = '<script>alert("unsafe")</script><img src=x onerror="bad()">&'
        data = synthetic()
        data.features = data.features.rename(columns={"synthetic_0": attack})
        data.metadata["outcome"] = data.metadata.outcome.replace({"case": attack})
        data.positive_label = attack
        data.metadata["cohort"] = data.metadata.cohort.replace({"external": attack})
        data.features.loc[0, "sample_id"] = attack
        data.metadata.loc[0, "sample_id"] = attack
        result = benchmark(data, holdout_cohort=attack, dataset_label=attack)
        report = render_report(result)
        self.assertNotIn(attack, report)
        self.assertIn("&lt;script&gt;", report)
        parser = ReportParser()
        parser.feed(report)
        self.assertNotIn("script", parser.tags)
        self.assertNotIn("img", parser.tags)
        self.assertNotIn("link", parser.tags)
        self.assertEqual(parser.external_resources, [])

    def test_dropped_features_have_reasons_and_no_coefficients(self):
        data = synthetic()
        data.features["all_missing"] = np.nan
        data.features["constant"] = 5
        result = benchmark(data, holdout_cohort="external")
        dropped = result.features[~result.features.retained].set_index("feature")
        self.assertEqual(dropped.loc["all_missing", "reason"], "all_missing_in_training")
        self.assertEqual(dropped.loc["constant", "reason"], "constant_in_training")
        self.assertTrue(dropped.coefficient.isna().all())
        self.assertTrue(dropped.absolute_coefficient_rank.isna().all())

    def test_destination_exists_and_render_failure_preserve_files(self):
        result = benchmark(synthetic(), n_splits=3)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            out.mkdir()
            sentinel = out / "important.txt"
            sentinel.write_text("preserve")
            with self.assertRaisesRegex(FileExistsError, "Output destination already exists"):
                result.write(out)
            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertEqual([p.name for p in out.iterdir()], ["important.txt"])
            fresh = Path(temporary) / "fresh"
            with patch("omicsfold.report.render_report", side_effect=RuntimeError("render failed")):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    result.write(fresh)
            self.assertFalse(fresh.exists())
            self.assertEqual(set(p.name for p in Path(temporary).iterdir()), {"run"})
            original_rename = Path.rename
            def fail_second_file(source, destination):
                if source.name == "predictions.tsv":
                    raise OSError("publish failed")
                return original_rename(source, destination)
            with patch.object(Path, "rename", fail_second_file):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    result.write(fresh)
            self.assertFalse(fresh.exists())
            file = Path(temporary) / "existing-file"
            file.write_text("original")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                result.write(file)
            self.assertEqual(file.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
