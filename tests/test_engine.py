"""All fixtures are synthetic. No biological data are used."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from omicsfold import analyze

spec = importlib.util.spec_from_file_location("synthetic", Path(__file__).parents[1] / "examples/synthetic.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.x, self.m = example.synthetic_data()

    def run_analysis(self, **kwargs):
        return analyze(self.x, self.m, positive_label="case", **kwargs)

    def test_repeatability_and_id_alignment(self):
        a = self.run_analysis()
        b = analyze(self.x.sample(frac=1, random_state=1), self.m.sample(frac=1, random_state=2), positive_label="case")
        pd.testing.assert_frame_equal(a.predictions, b.predictions)
        self.assertEqual(a.provenance, b.provenance)
        self.assertEqual(len(set(a.predictions.sample_id)), len(self.x))
        self.assertTrue(a.predictions.positive_probability.between(0, 1).all())

    def test_multiple_group_components(self):
        self.m["family"] = [f"F{i // 4}" for i in range(len(self.m))]
        result = self.run_analysis(group_columns=["patient", "family"])
        for fold in result.folds:
            train = self.m[self.m.sample_id.isin(fold["train_ids"])]
            test = self.m[self.m.sample_id.isin(fold["test_ids"])]
            for col in ["patient", "family"]:
                self.assertFalse(set(train[col]) & set(test[col]))

    def test_preprocessing_sees_training_rows_only(self):
        seen = []
        original = SimpleImputer.fit
        def spy(obj, X, y=None):
            seen.append(X.copy())
            return original(obj, X, y)
        with patch.object(SimpleImputer, "fit", spy):
            result = self.run_analysis(top_k=3)
        for values, fold in zip(seen, result.folds):
            expected = self.x.set_index("sample_id").loc[fold["train_ids"]].to_numpy()
            np.testing.assert_equal(values, expected)
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(result.exploratory_rankings), 15)

    def test_invalid_inputs(self):
        for mutation in ["duplicate", "mismatch", "missing_outcome", "infinite", "text", "single_class"]:
            with self.subTest(mutation=mutation):
                self.setUp()
                if mutation == "duplicate": self.x.loc[0, "sample_id"] = self.x.loc[1, "sample_id"]
                if mutation == "mismatch": self.m.loc[0, "sample_id"] = "other"
                if mutation == "missing_outcome": self.m.loc[0, "outcome"] = None
                if mutation == "infinite": self.x.iloc[0, 1] = np.inf
                if mutation == "text": self.x["bad"] = "not numeric"
                if mutation == "single_class": self.m["outcome"] = "case"
                with self.assertRaises(ValueError): self.run_analysis()

    def test_invalid_configuration(self):
        for kwargs in [{"n_splits": 1}, {"top_k": 100}, {"group_columns": ["absent"]}, {"seed": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.run_analysis(**kwargs)
        self.m["cohort"] = "one"
        with self.assertRaises(ValueError): self.run_analysis(group_columns=["cohort"])

    def test_csv_literal_ids_and_duplicate_header(self):
        from omicsfold.cli import read_csv
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "input.csv"
            p.write_text("sample_id,x\nNA,\n001,2\n")
            frame = read_csv(p)
            self.assertEqual(frame.sample_id.tolist(), ["NA", "001"])
            self.assertTrue(pd.isna(frame.x.iloc[0]))
            p.write_text("sample_id,x,x\n001,1,2\n")
            with self.assertRaises(ValueError): read_csv(p)

    def test_cli_parity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            self.x.to_csv(p / "x.csv", index=False)
            self.m.to_csv(p / "m.csv", index=False)
            cmd = [sys.executable, "-m", "omicsfold", "--features", str(p / "x.csv"),
                   "--orientation", "samples-by-features", "--sample-id-column", "sample_id",
                   "--outcome-column", "outcome", "--input-type", "numeric",
                   "--metadata", str(p / "m.csv"), "--positive-label", "case", "--output", str(p / "out")]
            subprocess.run(cmd, check=True, capture_output=True)
            actual = pd.read_csv(p / "out/predictions.csv")
            pd.testing.assert_frame_equal(actual, self.run_analysis().predictions)
            self.assertEqual(json.loads((p / "out/metrics.json").read_text()), self.run_analysis().metrics)
            second = subprocess.run(cmd, capture_output=True)
            self.assertEqual(second.returncode, 2)


if __name__ == "__main__":
    unittest.main()
