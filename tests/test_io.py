"""Synthetic CSV/TSV validation fixtures only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from omicsfold import load_data


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.x = self.path / "x.csv"
        self.m = self.path / "m.tsv"
        self.x.write_text("id,geneA,geneB\n001,1,2\n002,3,4\n", encoding="utf-8")
        self.m.write_text("id\tstatus\tpatient\tcohort\n002\tcontrol\tP2\tC2\n001\tcase\tP1\tC1\n", encoding="utf-8")

    def load(self, **kwargs):
        options = dict(orientation="samples-by-features", sample_id_column="id",
                       outcome_column="status", positive_label="case", input_type="numeric")
        options.update(kwargs)
        return load_data(self.x, self.m, **options)

    def test_exact_alignment_and_metadata_separation(self):
        result = self.load(group_column="patient", cohort_column="cohort")
        self.assertEqual(result.features.sample_id.tolist(), ["001", "002"])
        self.assertEqual(result.metadata.sample_id.tolist(), ["001", "002"])
        self.assertEqual(result.metadata.outcome.tolist(), ["case", "control"])
        self.assertEqual(result.features.columns.tolist(), ["sample_id", "geneA", "geneB"])
        self.assertEqual(result.summary()["cohort_column"], "cohort")

    def test_transpose_and_mixed_formats(self):
        expected = self.load().features
        self.x = self.path / "x.tsv"
        self.x.write_text("feature_id\t001\t002\ngeneA\t1\t3\ngeneB\t2\t4\n")
        actual = self.load(orientation="features-by-samples").features
        pd.testing.assert_frame_equal(actual, expected, check_names=False)

    def test_unmatched_reports_both_sides(self):
        self.x.write_text("id,geneA\n001,1\n003,2\n")
        with self.assertRaisesRegex(ValueError, r"matrix-only=\['003'\]; metadata-only=\['002'\]"):
            self.load()
        self.x.write_text("id,geneA\n001,1\n002 ,2\n")
        with self.assertRaisesRegex(ValueError, "002 "):
            self.load()

    def test_duplicate_ids_and_features(self):
        for text, message in (("id,A\n001,1\n001,2\n", r"Duplicate matrix sample IDs: \['001'\]"),
                              ("id,A,A\n001,1,2\n002,2,3\n", r"Duplicate column/feature names.*\['A'\]")):
            self.x.write_text(text)
            with self.assertRaisesRegex(ValueError, message):
                self.load()
        self.x.write_text("feature_id,001,002\nA,1,2\nA,3,4\n")
        with self.assertRaisesRegex(ValueError, "Duplicate feature names"):
            self.load(orientation="features-by-samples")
        self.x.write_text("id,A\n001,1\n002,2\n")
        self.m.write_text("id\tstatus\n001\tcase\n001\tcontrol\n")
        with self.assertRaisesRegex(ValueError, r"Duplicate metadata sample IDs: \['001'\]"):
            self.load()

    def test_outcomes_and_metadata_features(self):
        for text in ("id\tstatus\n001\tcase\n002\t\n", "id\tstatus\n001\tcase\n002\tcase\n"):
            self.m.write_text(text)
            with self.assertRaises(ValueError): self.load()
        self.setUp_metadata()
        self.x.write_text("id,status\n001,1\n002,2\n")
        with self.assertRaisesRegex(ValueError, "cannot be matrix features"):
            self.load()

    def setUp_metadata(self):
        self.m.write_text("id\tstatus\n001\tcase\n002\tcontrol\n")

    def test_numeric_missing_and_invalid_values(self):
        for kind in ("numeric", "expression"):
            self.x.write_text("id,A,B\n001,,NA\n002,-2,3.5\n")
            result = self.load(input_type=kind)
            self.assertEqual(result.summary()["missing_feature_values"], 2)
            self.assertEqual(result.features.A.iloc[1], -2)
            for bad in ("bad", "inf", "-inf"):
                self.x.write_text(f"id,A\n001,{bad}\n002,2\n")
                with self.assertRaises(ValueError): self.load(input_type=kind)

    def test_counts(self):
        self.assertEqual(self.load(input_type="counts").summary()["input_type"], "counts")
        for bad in ("-1", "1.5", "inf", "NaN", "", "bad", "9007199254740992.1"):
            self.x.write_text(f"id,A\n001,{bad}\n002,1\n")
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "counts must be finite, nonnegative integers"):
                self.load(input_type="counts")
        self.x.write_text("id,A,B\n001,0,0\n002,1,0\n")
        with self.assertRaisesRegex(ValueError, "Zero library size.*001"):
            self.load(input_type="counts")

    def test_required_roles_and_malformed_tables(self):
        for options in ({"orientation": "guess"}, {"input_type": "auto"}, {"positive_label": "absent"},
                        {"group_column": "absent"}, {"outcome_column": "id"}):
            with self.subTest(options=options), self.assertRaises(ValueError): self.load(**options)
        self.x.write_text("id,A\n001,1,2\n")
        with self.assertRaisesRegex(ValueError, "Wrong number of fields"): self.load()

    def test_validate_cli(self):
        cmd = [sys.executable, "-m", "omicsfold", "validate", "--features", str(self.x),
               "--metadata", str(self.m), "--orientation", "samples-by-features",
               "--sample-id-column", "id", "--outcome-column", "status",
               "--positive-label", "case", "--input-type", "counts"]
        run = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["samples"], 2)
        missing_flag = subprocess.run(cmd[:-2], capture_output=True, text=True)
        self.assertEqual(missing_flag.returncode, 2)
        self.assertIn("--input-type", missing_flag.stderr)
        self.x.write_text("id,A\n001,0\n002,1\n")
        invalid = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("Zero library size", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
