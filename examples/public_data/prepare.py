"""Prepare the pinned public ClinVar snapshot; no model is fitted here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
FREQUENCIES = ["AF_ESP", "AF_EXAC", "AF_TGP"]
VARIANT_KEY = ["CHROM", "POS", "REF", "ALT"]


def default_cache():
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return base / "OmicsFold" / "public-data"


def outside_repository(path):
    path = Path(path).expanduser().resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError("Keep raw downloads and prepared public-data work outside the Git repository")
    return path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_raw(config, raw_csv=None, download=False):
    path = outside_repository(raw_csv or default_cache() / "clinvar_conflicting.csv")
    if not path.exists():
        if not download:
            raise ValueError(f"Missing raw CSV: {path}; supply --raw-csv or explicitly use --download")
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = path.parent / "clinvar-conflicting-v2.zip"
        with urllib.request.urlopen(config["source"]["download_url"], timeout=120) as response:
            payload = response.read()
        with archive.open("xb") as handle:
            handle.write(payload)
        with zipfile.ZipFile(archive) as zipped:
            candidates = [n for n in zipped.namelist() if n.endswith("clinvar_conflicting.csv")]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one clinvar_conflicting.csv in the archive")
            content = zipped.read(candidates[0])
        if hashlib.sha256(content).hexdigest() != config["source"]["csv_sha256"]:
            raise ValueError("Downloaded CSV checksum differs from the pinned historical snapshot")
        with path.open("xb") as handle:
            handle.write(content)
    if sha256(path) != config["source"]["csv_sha256"]:
        raise ValueError("Raw CSV checksum differs from the snapshot used by ClinVar Conflict Explorer")
    return path


def transform_frame(raw):
    """Deterministic allowlist, variant IDs and gene groups; synthetic-testable."""
    required = VARIANT_KEY + ["SYMBOL", "CLASS"] + FREQUENCIES
    if any(c not in raw for c in required):
        raise ValueError("Required variant, gene, label or frequency columns are missing")
    if not raw["CLASS"].isin(["0", "1"]).all():
        raise ValueError("CLASS must contain only the recorded binary labels 0 and 1")
    gene = raw.SYMBOL.replace({"": pd.NA, "nan": pd.NA, "NA": pd.NA, "N/A": pd.NA}).astype("string").str.strip().str.upper()
    eligible = gene.str.fullmatch(r"[A-Z0-9_.-]+", na=False)
    data = raw.loc[eligible].copy()
    gene = gene.loc[eligible]
    if data.empty:
        raise ValueError("No variants have usable single gene symbols")
    if data[VARIANT_KEY].isin(["", "nan", "NA"]).any().any() or data.duplicated(VARIANT_KEY).any():
        raise ValueError("Variant keys must be complete and unique")
    if not data.POS.str.fullmatch(r"[0-9]+").all() or (data.POS.astype(int) <= 0).any():
        raise ValueError("Positions must be positive integer strings")
    ids = data[VARIANT_KEY].agg(":".join, axis=1)
    if ids.duplicated().any():
        raise ValueError("Composite variant sample IDs are not unique")
    features = pd.DataFrame({"sample_id": ids})
    for name in FREQUENCIES:
        values = pd.to_numeric(data[name].replace({"": np.nan, "nan": np.nan, "NA": np.nan}), errors="raise")
        if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Allele frequencies must be missing or in [0, 1]: {name}")
        features[name] = np.log1p(values * 1_000_000)
    metadata = pd.DataFrame({"sample_id": ids, "outcome": data.CLASS.map(
        {"0": "recorded_agreement", "1": "recorded_disagreement"}), "gene": gene})
    audit = {"raw_variants": len(raw), "excluded_missing_or_invalid_gene": int((~eligible).sum()),
             "eligible_variants": len(data), "genes": int(gene.nunique()),
             "outcome_counts": metadata.outcome.value_counts().to_dict(),
             "missing_frequencies": {c: int(features[c].isna().sum()) for c in FREQUENCIES}}
    return features.reset_index(drop=True), metadata.reset_index(drop=True), audit


def prepare(raw_csv, output, config):
    output = outside_repository(output)
    if output.exists():
        raise FileExistsError(f"Preparation destination already exists: {output}")
    if config["features"] != FREQUENCIES:
        raise ValueError("This preparation protocol supports exactly the three allowlisted frequency features")
    if sha256(raw_csv) != config["source"]["csv_sha256"]:
        raise ValueError("Unexpected raw CSV checksum")
    raw = pd.read_csv(raw_csv, usecols=VARIANT_KEY + ["SYMBOL", "CLASS"] + FREQUENCIES,
                      dtype=str, keep_default_na=False)
    features, metadata, audit = transform_frame(raw)
    output.mkdir(parents=True, exist_ok=False)
    features.to_csv(output / "features.tsv", sep="\t", index=False)
    metadata.to_csv(output / "metadata.tsv", sep="\t", index=False)
    provenance = {"source": config["source"], "raw_csv_sha256": sha256(raw_csv), "inclusion_audit": audit,
                  "feature_mapping": {c: f"log1p(1000000 * {c})" for c in FREQUENCIES},
                  "sample_id": "GRCh37 CHROM:POS:REF:ALT from the source snapshot; not a new ClinVar accession",
                  "group_mapping": "strip and uppercase SYMBOL; retain a single symbol matching [A-Z0-9_.-]+; no gene remapping or gene-as-feature",
                  "outcome": "Source CLASS=1 is recorded classification disagreement; CLASS=0 is recorded agreement among eligible multiple-submission variants",
                  "excluded_predictors": "All clinical assertions, review metadata, coordinates, IDs, gene, consequence and other annotations",
                  "preprocessing": "Only an element-wise log1p frequency transform here; training-fold median imputation, constant filtering and scaling in installed OmicsFold CLI",
                  "prepared_sha256": {name: sha256(output / name) for name in ("features.tsv", "metadata.tsv")},
                  "software": {"numpy": np.__version__, "pandas": pd.__version__}}
    (output / "preparation.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "benchmark.json")
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw = get_raw(config, args.raw_csv, args.download)
    provenance = prepare(raw, args.output, config)
    print(json.dumps(provenance["inclusion_audit"], indent=2))


if __name__ == "__main__":
    main()
