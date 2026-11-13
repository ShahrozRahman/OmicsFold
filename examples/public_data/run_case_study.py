"""Prepare public data, exercise the installed CLI and audit its held-out splits."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sysconfig

import pandas as pd

from prepare import HERE, default_cache, get_raw, outside_repository, prepare, sha256


def installed_cli():
    candidate = shutil.which("omicsfold")
    if candidate:
        return candidate
    scripts = Path(sysconfig.get_path("scripts"))
    for name in ("omicsfold.exe", "omicsfold"):
        if (scripts / name).is_file():
            return str(scripts / name)
    raise RuntimeError('Install the package first: python -m pip install -e ".[dev]"')


def audit_splits(result_dir, prepared_dir):
    splits = pd.read_csv(result_dir / "splits.tsv", sep="\t", dtype=str, keep_default_na=False)
    predictions = pd.read_csv(result_dir / "predictions.tsv", sep="\t", dtype={"sample_id": str})
    metadata = pd.read_csv(prepared_dir / "metadata.tsv", sep="\t", dtype=str)
    expected = set(metadata.sample_id)
    rows = []
    for fold, frame in splits.groupby("fold", sort=True):
        train = frame[frame.partition == "train"]
        test = frame[frame.partition == "test"]
        if (set(train.sample_id) & set(test.sample_id) or set(train.group) & set(test.group)
                or set(frame.sample_id) != expected or frame.sample_id.duplicated().any()):
            raise RuntimeError(f"Invalid ID/gene membership in fold {fold}")
        if train.outcome.nunique() != 2 or test.outcome.nunique() != 2:
            raise RuntimeError(f"Single-class partition in fold {fold}")
        for model in ("prevalence", "logistic_l2"):
            scored = predictions[(predictions.fold == int(fold)) & (predictions.model == model)]
            if set(scored.sample_id) != set(test.sample_id) or scored.sample_id.duplicated().any():
                raise RuntimeError(f"Invalid held-out predictions in fold {fold}")
        rows.append({"fold": int(fold), "train_samples": len(train), "test_samples": len(test),
                     "train_genes": train.group.nunique(), "test_genes": test.group.nunique(),
                     "train_positives": int((train.outcome == "recorded_disagreement").sum()),
                     "test_positives": int((test.outcome == "recorded_disagreement").sum()),
                     "id_overlap": 0, "gene_overlap": 0})
    for model, frame in predictions.groupby("model"):
        if len(frame) != len(metadata) or set(frame.sample_id) != expected or frame.sample_id.duplicated().any():
            raise RuntimeError(f"Expected exactly one held-out score per variant for {model}")
    return {"variants": len(metadata), "genes": int(metadata.gene.nunique()),
            "positive_prevalence": float((metadata.outcome == "recorded_disagreement").mean()),
            "folds": rows, "all_checks_passed": True}


def publish_report(work, publication, config, preparation, audit):
    """Publish a compact derived report; full variant-level outputs stay in work."""
    publication = Path(publication)
    if publication.exists():
        raise FileExistsError(f"Publication destination already exists: {publication}")
    result_dir = work / "results"
    metrics = pd.read_csv(result_dir / "metrics.tsv", sep="\t")
    means = metrics[metrics.scope == "mean"].set_index("model")
    baseline, model = means.loc["prevalence"], means.loc["logistic_l2"]
    escape = lambda value: html.escape(str(value), quote=True)
    context = f'''<section><h2>Public-data case study: source, meaning and reproduction</h2>
<p>Source: Kevin Arvai, Genetic Variant Classifications, Kaggle dataset kevinarvai/clinvar-conflicting version 2 (CC0: Public Domain). Original resource: NCBI ClinVar GRCh37 VCF downloaded by the dataset author on 2018-04-07. This is the same checksum-pinned snapshot used by ClinVar Conflict Explorer.</p>
<p>Dataset URL: {escape(config['source']['url'])}. Raw CSV SHA-256: {escape(preparation['raw_csv_sha256'])}.</p>
<p>Each sample is a variant, not a patient. The outcome is recorded classification disagreement in this historical multiple-submission snapshot. It does not label pathogenicity, disease, or disagreements arising in the future. Source categories combine benign/likely benign, VUS, and pathogenic/likely pathogenic; disagreement spans these categories.</p>
<p>{audit['variants']:,} variants across {audit['genes']:,} gene groups were evaluated. Only three population frequency inputs are used, transformed independently with log1p(1,000,000 × frequency). Gene is a splitting constraint, never a predictor. Clinical assertions and review information are excluded. All five folds have both classes and zero train/test gene or ID overlap.</p>
<p>Mean average precision: model {model.average_precision:.6f}, prevalence baseline {baseline.average_precision:.6f}, absolute difference {model.average_precision - baseline.average_precision:.6f}. Mean ROC AUC: model {model.roc_auc:.6f}, baseline {baseline.roc_auc:.6f}. This is a modest internal ranking signal, not clinical validation. These estimates are sample-level, so genes with more variants contribute more observations. There is no temporal or external-cohort validation.</p>
<p>Cross-cohort expression evaluation remains <strong>pending</strong>: GSE10072 and GSE7670 share GPL96 probes and compatible tissue labels after exclusions, but released RMA and MAS5 values do not have a verified common preprocessing workflow. No expression cross-cohort scores are reported.</p>
<p>From the OmicsFold repository after local installation, reproduce with <code>python examples/public_data/run_case_study.py --download --work-dir /path/to/new/outside-repository-work</code>. Alternatively provide <code>--raw-csv /path/to/clinvar_conflicting.csv</code>. The script verifies the pinned checksum, prepares tables, invokes the installed <code>omicsfold validate</code> and <code>omicsfold benchmark</code> commands, and independently audits splits. Use an unused work directory. Raw downloads and full variant-level results remain outside Git.</p>
<p>This compact publication includes the HTML, fold metrics, exploratory coefficients, preparation provenance and split audit. Full predictions.tsv, splits.tsv and run.json are generated under the external work directory's results folder. The installed CLI report is otherwise unchanged; this case-study context is added by the reproduction script.</p></section>'''
    document = (result_dir / "report.html").read_text(encoding="utf-8").replace("</main>", context + "</main>")
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    summary = {key: run[key] for key in ("report_schema_version", "configuration", "input_checksums", "package_versions", "class_balance", "metric_summary")}
    summary["publication_note"] = "Compact summary; full run.json and variant-level outputs are reproduced outside Git"
    publication.mkdir(parents=True, exist_ok=False)
    (publication / "report.html").write_text(document, encoding="utf-8")
    for name in ("metrics.tsv", "features.tsv"):
        shutil.copyfile(result_dir / name, publication / name)
    for name, payload in (("preparation.json", preparation), ("split_audit.json", audit), ("run_summary.json", summary)):
        (publication / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest = {"full_cli_output_sha256": {file.name: sha256(file) for file in sorted(result_dir.iterdir())},
                "published_sha256": {file.name: sha256(file) for file in sorted(publication.iterdir())},
                "note": "Published HTML adds source/interpretation context; full CLI output checksums refer to the original artifacts"}
    (publication / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "benchmark.json")
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=default_cache() / "clinvar-case-study")
    parser.add_argument("--publish-dir", type=Path, help="New compact derived report directory; defaults to WORK/publication")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    work = outside_repository(args.work_dir)
    if work.exists():
        raise FileExistsError(f"Case-study work directory already exists: {work}; choose a new --work-dir")
    if args.publish_dir and args.publish_dir.exists():
        raise FileExistsError(f"Publication destination already exists: {args.publish_dir}")
    raw = get_raw(config, args.raw_csv, args.download)
    preparation = prepare(raw, work / "prepared", config)
    common = ["--features", "prepared/features.tsv", "--metadata", "prepared/metadata.tsv"]
    benchmark = config["benchmark"]
    for key in ("orientation", "sample_id_column", "outcome_column", "positive_label", "input_type", "group_column"):
        common.extend(["--" + key.replace("_", "-"), str(benchmark[key])])
    cli = installed_cli()
    validation_command = [cli, "validate", *common]
    benchmark_command = [cli, "benchmark", *common]
    for key in ("split", "folds", "seed", "dataset_label"):
        benchmark_command.extend(["--" + key.replace("_", "-"), str(benchmark[key])])
    for key, default in (("weighting", "sample"), ("bootstrap_repeats", 0)):
        benchmark_command.extend(["--" + key.replace("_", "-"), str(benchmark.get(key, default))])
    benchmark_command.extend(["--output", "results"])
    subprocess.run(validation_command, cwd=work, check=True)
    subprocess.run(benchmark_command, cwd=work, check=True)
    audit = audit_splits(work / "results", work / "prepared")
    audit["cli_entry_point"] = "installed omicsfold console script"
    audit["configuration_sha256"] = sha256(args.config)
    audit["preparation_script_sha256"] = sha256(HERE / "prepare.py")
    audit["runner_script_sha256"] = sha256(HERE / "run_case_study.py")
    audit["benchmark_command"] = ["omicsfold", *benchmark_command[1:]]
    (work / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    publish_report(work, args.publish_dir or work / "publication", config, preparation, audit)
    print(json.dumps({"output": str(work / "results"), "inclusion_audit": preparation["inclusion_audit"],
                      "split_audit": audit}, indent=2))


if __name__ == "__main__":
    main()
