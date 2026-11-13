"""Inspect two GEO candidates and save a conservative compatibility decision."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import re
import urllib.request

from prepare import default_cache, outside_repository, sha256

SOURCES = {
    "GSE10072": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE10nnn/GSE10072/matrix/GSE10072_series_matrix.txt.gz",
    "GSE7670": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE7nnn/GSE7670/matrix/GSE7670_series_matrix.txt.gz",
}


def inspect(path, accession):
    text = gzip.decompress(path.read_bytes()).decode("utf-8-sig")
    fields, probes = {}, []
    in_table = False
    for row in csv.reader(io.StringIO(text), delimiter="\t"):
        if not row:
            continue
        if row[0] == "!series_matrix_table_begin":
            in_table = True
        elif row[0] == "!series_matrix_table_end":
            in_table = False
        elif in_table and row[0] != "ID_REF":
            probes.append(row[0])
        elif row[0].startswith("!"):
            fields.setdefault(row[0], []).append(row[1:])
    titles = fields["!Sample_title"][0]
    samples = fields["!Sample_geo_accession"][0]
    included, excluded = [], []
    for sample, title in zip(samples, titles):
        if accession == "GSE10072":
            match = re.fullmatch(r"(Lung Tumor|Normal Lung)_(GT[0-9]+)", title)
            outcome = "tumor" if match and match[1] == "Lung Tumor" else "non_tumor"
        else:
            match = re.fullmatch(r"([0-9]+)(N|T): (Adjacent normal|Tumor) part of adenocarcinoma", title)
            outcome = "tumor" if match and match[2] == "T" else "non_tumor"
        if match:
            patient = match[2] if accession == "GSE10072" else match[1]
            included.append({"sample_id": sample, "patient": f"{accession}:{patient}", "outcome": outcome})
        else:
            excluded.append({"sample_id": sample, "title": title})
    descriptions = sorted({v for row in fields.get("!Sample_data_processing", []) for v in row})
    report = {"accession": accession, "url": SOURCES[accession], "sha256": sha256(path),
              "platforms": sorted(set(fields["!Sample_platform_id"][0])), "samples_total": len(samples),
              "included_samples": len(included), "included_outcomes": {label: sum(s["outcome"] == label for s in included) for label in ("tumor", "non_tumor")},
              "included_patient_groups": len({s["patient"] for s in included}),
              "excluded_samples": excluded, "processing_descriptions": descriptions,
              "unique_probe_ids": len(set(probes)), "duplicate_probe_ids": len(probes) - len(set(probes))}
    return report, set(probes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache() / "geo")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = outside_repository(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    reports, sets = [], []
    for accession, url in SOURCES.items():
        path = cache / f"{accession}_series_matrix.txt.gz"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response, path.open("xb") as handle:
                handle.write(response.read())
        report, probes = inspect(path, accession)
        reports.append(report)
        sets.append(probes)
    common = sorted(p for p in sets[0] & sets[1] if not p.startswith("AFFX-"))
    result = {"status": "pending", "cross_cohort_evaluation_performed": False,
              "common_phenotype": "lung adenocarcinoma tissue versus non-tumor/adjacent lung tissue, not healthy-person diagnosis",
              "cohorts": reports, "common_non_control_probes": len(common),
              "feature_mapping": "Exact GPL96 probe-set ID intersection, excluding AFFX- controls; no gene symbol collapse or outcome-based feature selection",
              "checks": {"same_platform": all(r["platforms"] == ["GPL96"] for r in reports),
                         "binary_metadata": all(min(r["included_outcomes"].values()) > 0 for r in reports),
                         "unique_common_features": bool(common) and all(r["duplicate_probe_ids"] == 0 for r in reports),
                         "verified_common_preprocessing": False},
              "reason": "GSE10072 released data use RMA and averaged technical replicates; GSE7670 released data use MAS5 with per-array target 500. A common raw-CEL or validated training-reference preprocessing workflow has not been established. Do not merge, re-quantile-normalize jointly or report cross-cohort performance from these mismatched releases.",
              "next_step": "Reprocess both raw CEL cohorts with a documented compatible workflow that fits any cross-sample parameters on training data only; verify patient independence and probe annotation before releasing cross-cohort results."}
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"], "common_non_control_probes": len(common)}, indent=2))


if __name__ == "__main__":
    main()
