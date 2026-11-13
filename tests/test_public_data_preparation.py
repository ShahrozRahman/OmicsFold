"""Synthetic preparation and independent-audit fixtures; no public downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from omicsfold import benchmark, load_data

EXAMPLE = Path(__file__).parents[1] / "examples/public_data"


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preparation = module_from_path("public_data_prepare", EXAMPLE / "prepare.py")
with patch.dict("sys.modules", {"prepare": preparation}):
    runner = module_from_path("public_data_runner", EXAMPLE / "run_case_study.py")


def synthetic_raw():
    n = 24
    return pd.DataFrame({"CHROM": ["1"] * n, "POS": [str(i + 1) for i in range(n)],
                         "REF": ["A"] * n, "ALT": ["G"] * n,
                         "SYMBOL": [f"GENE_{i // 2}" for i in range(n)],
                         "CLASS": [str(i % 2) for i in range(n)],
                         "AF_ESP": [str((i + 1) / 1000) for i in range(n)],
                         "AF_EXAC": [str((i + 1) / 2000) for i in range(n)],
                         "AF_TGP": [str((i + 1) / 3000) for i in range(n)],
                         "CLNSIG": ["target-derived annotation excluded"] * n})


class PublicPreparationTests(unittest.TestCase):
    def test_allowlist_ids_gene_group_and_elementwise_transform(self):
        raw = synthetic_raw()
        raw.loc[0, "SYMBOL"] = " gene_0 "
        raw.loc[1, "SYMBOL"] = "nan"
        raw.loc[2, "AF_ESP"] = ""
        features, metadata, audit = preparation.transform_frame(raw)
        self.assertEqual(features.columns.tolist(), ["sample_id", "AF_ESP", "AF_EXAC", "AF_TGP"])
        self.assertEqual(audit["excluded_missing_or_invalid_gene"], 1)
        self.assertEqual(features.sample_id.iloc[0], "1:1:A:G")
        self.assertEqual(metadata.gene.iloc[0], "GENE_0")
        self.assertEqual(metadata.outcome.iloc[0], "recorded_agreement")
        self.assertAlmostEqual(features.AF_ESP.iloc[0], np.log1p(1000))
        self.assertTrue(features.AF_ESP.isna().any())
        changed = raw.copy()
        changed.loc[3:, "AF_ESP"] = "0.9"
        other, _, _ = preparation.transform_frame(changed)
        self.assertEqual(features.AF_ESP.iloc[0], other.AF_ESP.iloc[0])

    def test_invalid_frequency_label_and_duplicate_variant(self):
        for column, value in (("AF_ESP", "-1"), ("AF_ESP", "1.1"), ("AF_EXAC", "inf"),
                              ("AF_TGP", "text"), ("CLASS", "unknown")):
            raw = synthetic_raw()
            raw.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                preparation.transform_frame(raw)
        raw = synthetic_raw()
        raw.loc[1, "POS"] = raw.loc[0, "POS"]
        with self.assertRaisesRegex(ValueError, "unique"):
            preparation.transform_frame(raw)

    def test_checksum_pin_preparation_and_independent_split_audit(self):
        config = json.loads((EXAMPLE / "benchmark.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "synthetic.csv"
            synthetic_raw().to_csv(raw, index=False)
            with self.assertRaisesRegex(ValueError, "checksum"):
                preparation.get_raw(config, raw_csv=raw)
            config["source"]["csv_sha256"] = preparation.sha256(raw)
            prepared = root / "prepared"
            manifest = preparation.prepare(raw, prepared, config)
            self.assertEqual(manifest["inclusion_audit"]["eligible_variants"], 24)
            data = load_data(prepared / "features.tsv", prepared / "metadata.tsv",
                             orientation="samples-by-features", sample_id_column="sample_id", outcome_column="outcome",
                             positive_label="recorded_disagreement", input_type="numeric", group_column="gene")
            result = root / "results"
            benchmark(data, n_splits=3).write(result)
            audit = runner.audit_splits(result, prepared)
            self.assertTrue(audit["all_checks_passed"])
            self.assertEqual(audit["variants"], 24)
            self.assertEqual(sum(row["test_samples"] for row in audit["folds"]), 24)
            splits = pd.read_csv(result / "splits.tsv", sep="\t", dtype=str)
            rows = splits[(splits.fold == "0") & (splits.partition == "test")].index
            training_gene = splits[(splits.fold == "0") & (splits.partition == "train")].group.iloc[0]
            splits.loc[rows[0], "group"] = training_gene
            splits.to_csv(result / "splits.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "gene membership"):
                runner.audit_splits(result, prepared)


if __name__ == "__main__":
    unittest.main()
